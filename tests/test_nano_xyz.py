#!/usr/bin/env python3
"""
Comprehensive test suite for Nano XYZ model.

Tests all major functionality including:
- Model creation and basic inference
- Feedforward network implementation
- DCA (Dynamic Context Allocation)
- Encoder-decoder architectures
- Attention mechanisms
- Quantization support
"""

import torch
import pytest

# Set seeds for reproducible tests
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

from nano_xyz.configuration_nano import NanoConfig
from nano_xyz.model import NanoModel, NanoEncoder, NanoDecoder
from nano_xyz.modeling_nano import NanoForCausalLM, NanoEncoderModel, NanoDecoderModel
from nano_xyz.cache_utils import get_past_key_values_length


@pytest.fixture
def tiny_config():
    """Create a tiny config for fast testing."""
    return NanoConfig.from_preset("decoder_tiny")


@pytest.fixture
def dca_config():
    """Create DCA-enabled config."""
    config = NanoConfig.from_preset("decoder_tiny")
    config.use_dca = True
    config.dca_attention_budget = 0.5
    return config


@pytest.fixture
def encoder_config():
    """Create encoder config."""
    return NanoConfig.from_preset("encoder_decoder_tiny")


@pytest.fixture
def decoder_config():
    """Create decoder config."""
    return NanoConfig.from_preset("decoder_tiny")


@pytest.fixture
def sample_inputs(tiny_config):
    """Create sample inputs for testing."""
    batch_size = 2
    seq_len = 32
    return {
        "input_ids": torch.randint(0, tiny_config.vocab_size, (batch_size, seq_len)),
        "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
        "batch_size": batch_size,
        "seq_len": seq_len,
    }


class TestNanoXYZ:
    """Test suite for Nano XYZ model functionality."""

    @pytest.mark.parametrize("config_fixture", ["tiny_config", "dca_config"])
    def test_model_creation(self, request, config_fixture):
        """Test basic model creation with different configs."""
        config = request.getfixturevalue(config_fixture)
        model = NanoModel(config)
        assert model is not None
        assert len(model.layers) == config.n_layer

        # Check DCA-specific attributes if enabled
        if hasattr(config, 'use_dca') and config.use_dca:
            # DCA allocator should be on the self_attn module of the layers
            assert 'dca_allocator' in model.layers[0].self_attn._modules
            assert model.layers[0].self_attn.dca_allocator is not None

    @pytest.mark.parametrize("config_fixture,input_type", [
        ("tiny_config", "short"),
        ("dca_config", "medium"),
        ("tiny_config", "batch"),
    ])
    def test_model_forward(self, request, config_fixture, input_type):
        """Test basic model forward pass with different inputs."""
        config = request.getfixturevalue(config_fixture)
        model = NanoModel(config)
        model.eval()

        # Create appropriate inputs based on type
        if input_type == "short":
            batch_size, seq_len = 1, 5
        elif input_type == "medium":
            batch_size, seq_len = 2, 20
        elif input_type == "batch":
            batch_size, seq_len = 3, 10

        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        assert outputs.last_hidden_state.shape == (batch_size, seq_len, config.n_embd)
        assert not torch.isnan(outputs.last_hidden_state).any()
        assert not torch.isinf(outputs.last_hidden_state).any()

    def test_model_forward_without_attention_mask(self, tiny_config):
        """Test model forward pass without explicit attention mask."""
        model = NanoModel(tiny_config)
        model.eval()

        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, tiny_config.vocab_size, (batch_size, seq_len))

        with torch.no_grad():
            outputs = model(input_ids)

        assert outputs.last_hidden_state.shape == (batch_size, seq_len, tiny_config.n_embd)
        assert outputs.past_key_values is None


    def test_dca_functionality(self, dca_config):
        """Test Dynamic Context Allocation."""
        model = NanoModel(dca_config)
        # DCA allocator should be on the self_attn module of the layers
        assert 'dca_allocator' in model.layers[0].self_attn._modules
        assert model.layers[0].self_attn.dca_allocator is not None

        # Test with sequence longer than local window
        batch_size, seq_len = 1, 50  # Longer than local window
        input_ids = torch.randint(0, dca_config.vocab_size, (batch_size, seq_len))

        model.eval()
        with torch.no_grad():
            outputs = model(input_ids)

        # Should complete without errors
        assert outputs.last_hidden_state.shape == (batch_size, seq_len, dca_config.n_embd)

        # Test DCA produces different attention patterns than dense
        dense_config = NanoConfig.from_preset("decoder_tiny")
        dense_config.use_dca = False
        dense_model = NanoModel(dense_config)

        with torch.no_grad():
            dca_outputs = model(input_ids)
            dense_outputs = dense_model(input_ids)

        # Outputs should be different (DCA changes attention patterns)
        assert not torch.allclose(dca_outputs.last_hidden_state, dense_outputs.last_hidden_state, atol=1e-3)

    def test_encoder_decoder(self, encoder_config, decoder_config):
        """Test encoder-decoder architecture."""
        # Modify configs as needed
        encoder_config.is_decoder = False
        encoder_config.add_cross_attention = False
        decoder_config.is_decoder = True
        decoder_config.add_cross_attention = True

        encoder = NanoEncoder(encoder_config)
        decoder = NanoDecoder(decoder_config)

        # Test encoder
        batch_size, src_len = 2, 10
        src_ids = torch.randint(0, encoder_config.vocab_size, (batch_size, src_len))
        encoder_outputs = encoder(src_ids)

        # Test decoder with cross-attention
        tgt_len = 8
        tgt_ids = torch.randint(0, decoder_config.vocab_size, (batch_size, tgt_len))
        decoder_outputs = decoder(
            tgt_ids,
            encoder_outputs=encoder_outputs.last_hidden_state,
            output_hidden_states=True,
        )

        hidden_states = decoder_outputs["hidden_states"]
        assert hidden_states.shape == (batch_size, tgt_len, decoder_config.n_embd)

    def test_quantization(self, tiny_config):
        """Test quantization support."""
        config = tiny_config
        config.quantization_config = {
            "method": "bnb",
            "bits": 4,
            "quant_type": "nf4",
            "double_quant": True,
        }

        # This test mainly checks that quantization config is accepted
        # Actual quantization requires bitsandbytes to be installed
        try:
            model = NanoModel(config)
        except RuntimeError as exc:
            if "BitsAndBytes" in str(exc) or "bitsandbytes" in str(exc):
                pytest.skip(f"Skipping quantization test: {exc}")
            raise
        assert model is not None

    def test_kv_caching(self, tiny_config):
        """Test KV caching for efficient generation."""
        model = NanoModel(tiny_config)

        batch_size, seq_len = 1, 5
        input_ids = torch.randint(0, tiny_config.vocab_size, (batch_size, seq_len))

        # First forward pass
        outputs1 = model(input_ids, use_cache=True)
        assert outputs1.past_key_values is not None

        # Second forward pass with cached KV
        new_token = torch.randint(0, tiny_config.vocab_size, (batch_size, 1))
        outputs2 = model(new_token, past_key_values=outputs1.past_key_values, use_cache=True)

        assert outputs2.last_hidden_state.shape == (batch_size, 1, tiny_config.n_embd)

    def test_encoder_decoder_model(self):
        """Test NanoEncoderDecoderModel wrapper."""
        from nano_xyz.modeling_nano import NanoEncoderDecoderModel

        config = NanoConfig.from_preset("encoder_decoder_tiny")
        model = NanoEncoderDecoderModel(config)

        # Test that components exist
        assert hasattr(model, 'encoder')
        assert hasattr(model, 'decoder')
        assert hasattr(model, 'get_encoder')
        assert hasattr(model, 'get_decoder')

        # Test basic forward pass
        batch_size, src_len, tgt_len = 2, 8, 6
        input_ids = torch.randint(0, config.vocab_size, (batch_size, src_len))
        decoder_input_ids = torch.randint(0, config.vocab_size, (batch_size, tgt_len))

        encoder_outputs = model.encoder(input_ids)
        decoder_outputs = model.decoder(
            decoder_input_ids,
            encoder_outputs=encoder_outputs.last_hidden_state
        )

        assert encoder_outputs.last_hidden_state.shape == (batch_size, src_len, config.n_embd)
        assert decoder_outputs["logits"].shape == (batch_size, tgt_len, config.vocab_size)


class TestRegression:
    """Regression tests for specific bugs and issues."""

    def test_config_parsing_validation(self):
        """Test that config parsing properly validates keys and types."""
        # Skip this test as train_hf module has been moved to scripts/
        pytest.skip("Config parsing validation test skipped - train_hf module moved to scripts/")

    def test_encoder_runtime_safety(self):
        """Test that encoder validates inputs at runtime."""
        from nano_xyz.model import NanoEncoder
        from nano_xyz.configuration_nano import NanoConfig

        config = NanoConfig(block_size=512, n_embd=128, n_head=8, n_layer=2)
        encoder = NanoEncoder(config)

        # Valid input should work
        valid_input = torch.randint(0, 1000, (2, 32))
        result = encoder(valid_input)
        assert hasattr(result, 'last_hidden_state')
        assert hasattr(result, 'hidden_states')

        # Invalid input types should fail
        with pytest.raises(TypeError):
            encoder("not a tensor")

        # Invalid shapes should fail
        with pytest.raises(ValueError):
            encoder(torch.randint(0, 1000, (2, 3, 4)))  # 3D tensor

        # Invalid dtypes should fail
        with pytest.raises(ValueError):
            encoder(torch.randn(2, 32))  # Float tensor

        # Test with longer sequence (should work if within block_size)
        longer_input = torch.randint(0, 1000, (2, 128))
        longer_result = encoder(longer_input)
        assert longer_result.last_hidden_state.shape == (2, 128, 128)

    def test_decoder_runtime_safety(self):
        """Test that decoder validates inputs at runtime."""
        from nano_xyz.model import NanoDecoder
        from nano_xyz.configuration_nano import NanoConfig

        config = NanoConfig(
            block_size=512, n_embd=128, n_head=8, n_layer=2,
            is_decoder=True, add_cross_attention=True
        )
        decoder = NanoDecoder(config)

        # Valid inputs should work
        input_ids = torch.randint(0, 1000, (2, 16))
        encoder_outputs = torch.randn(2, 10, 128)  # Mock encoder outputs
        result = decoder(input_ids, encoder_outputs=encoder_outputs)
        assert isinstance(result, dict)
        assert "logits" in result

        # Invalid input types should fail
        with pytest.raises(TypeError):
            decoder("not a tensor")

        # Invalid shapes should fail
        with pytest.raises(ValueError):
            decoder(torch.randint(0, 1000, (2, 3, 4)))  # 3D tensor

        # Invalid encoder outputs shape should fail
        with pytest.raises(ValueError):
            decoder(input_ids, encoder_outputs=torch.randn(2, 10, 64))  # Wrong hidden size

    def test_dca_generation_warning(self):
        """Test that DCA generation produces appropriate warnings."""
        from nano_xyz.configuration_nano import NanoConfig
        from nano_xyz.model import NanoModel
        import warnings

        config = NanoConfig(
            use_dca=True,
            dca_enable_generation=True,
            block_size=512,
            n_embd=64,
            n_head=4,
            n_layer=2,
            dca_local_window=256
        )
        model = NanoModel(config)

        # Create inputs that will trigger generation mode
        input_ids = torch.randint(0, 1000, (1, 10))

        # First forward pass to establish cache
        with torch.no_grad():
            result1 = model(input_ids, use_cache=True)

        # Second forward pass should trigger warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with torch.no_grad():
                _ = model(torch.randint(0, 1000, (1, 1)), past_key_values=result1.past_key_values, use_cache=True)

            assert len(w) >= 1
            assert any("experimental" in str(warning.message).lower() for warning in w)
            assert any("dca_enable_generation=False" in str(warning.message) for warning in w)

    def test_quantization_api_fallback(self):
        """Test that quantization gracefully handles API differences."""
        from nano_xyz.quantization import LinearFactory

        # Test with no quantization config
        factory = LinearFactory()
        layer = factory.create_linear(64, 32)
        assert layer is not None

        # Test with bnb config (may skip if not installed)
        bnb_config = {"method": "bnb", "bits": 4, "quant_type": "nf4"}
        factory = LinearFactory(bnb_config)

        try:
            layer = factory.create_linear(64, 32)
            # If we get here, quantization worked
            assert layer is not None
        except RuntimeError as e:
            # Should be a clear error message
            assert "BitsAndBytes" in str(e) or "bitsandbytes" in str(e)

    def test_model_output_consistency(self):
        """Test that model outputs are consistent across different code paths."""
        from nano_xyz.modeling_nano import NanoEncoderModel, NanoDecoderModel

        enc_config = NanoConfig(n_embd=64, n_head=4, n_layer=2, block_size=512, vocab_size=1000)
        dec_config = NanoConfig(n_embd=64, n_head=4, n_layer=2, block_size=512, vocab_size=1000,
                               is_decoder=True, add_cross_attention=True)

        encoder = NanoEncoderModel(enc_config)
        decoder = NanoDecoderModel(dec_config)

        input_ids = torch.randint(0, 1000, (2, 16))

        # Encoder should return BaseModelOutputWithPast
        enc_output = encoder(input_ids)
        assert hasattr(enc_output, 'last_hidden_state')
        assert enc_output.last_hidden_state.shape == (2, 16, 64)

        # Decoder should return CausalLMOutputWithCrossAttentions
        dec_output = decoder(input_ids, encoder_hidden_states=enc_output.last_hidden_state)
        assert hasattr(dec_output, 'logits')
        assert hasattr(dec_output, 'hidden_states')
        assert dec_output.logits.shape == (2, 16, 1000)

    def test_gradient_checkpointing(self):
        """Test gradient checkpointing for memory efficiency."""
        from nano_xyz.configuration_nano import NanoConfig
        config = NanoConfig(use_activation_checkpointing=True)

        model = NanoModel(config)

        # Should work in training mode with checkpointing
        model.train()
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        # This mainly tests that the checkpointing code path works
        outputs = model(input_ids)
        assert outputs.last_hidden_state.shape == (batch_size, seq_len, config.n_embd)


class TestNanoForCausalLM:
    """Test the causal LM wrapper."""

    @pytest.fixture
    def lm_config(self):
        """Configuration for causal LM testing."""
        return NanoConfig(
            n_layer=2,
            n_head=4,
            n_embd=128,
            block_size=512,
            dca_local_window=256,
            n_exp=4,
            vocab_size=1000,
        )

    def test_causal_lm_creation(self, lm_config):
        """Test causal LM model creation."""
        model = NanoForCausalLM(lm_config)
        assert model is not None
        assert hasattr(model, 'lm_head')

    def test_causal_lm_forward(self, lm_config):
        """Test causal LM forward pass."""
        model = NanoForCausalLM(lm_config)
        model.eval()

        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, lm_config.vocab_size, (batch_size, seq_len))

        with torch.no_grad():
            outputs = model(input_ids)

        # Should return logits for language modeling
        assert outputs.logits.shape == (batch_size, seq_len, lm_config.vocab_size)


if __name__ == "__main__":
    # Run basic tests without pytest
    print("Running Nano XYZ comprehensive tests...")

    # Create test config
    config = NanoConfig(
        n_layer=2,
        n_head=4,
        n_embd=128,
        block_size=512,
        dca_local_window=256,
        vocab_size=1000,
    )

    # Test model creation
    print("Testing model creation...")
    model = NanoModel(config)
    print("PASS: Model created successfully")

    # Test basic inference
    print("Testing basic inference...")
    model.eval()
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    with torch.no_grad():
        outputs = model(input_ids)

    assert outputs.last_hidden_state.shape == (batch_size, seq_len, config.n_embd)
    print("PASS: Basic inference works")


    # Test DCA
    print("Testing DCA functionality...")
    config.use_dca = True
    config.dca_local_window = 5
    model = NanoModel(config)

    seq_len = 20  # Longer than local window
    input_ids = torch.randint(0, config.vocab_size, (1, seq_len))

    with torch.no_grad():
        outputs = model(input_ids)

    assert outputs.last_hidden_state.shape == (1, seq_len, config.n_embd)
    print("PASS: DCA functionality works")


def test_cache_utility_functions():
        """Test cache utility functions with different cache formats."""
        # Test None input
        assert get_past_key_values_length(None) == 0

        # Test empty list
        assert get_past_key_values_length([]) == 0

        # Test legacy tuple format (transformers format: [batch, num_heads, seq_len, head_dim])
        mock_k = torch.randn(2, 4, 5, 8)  # [batch, num_heads, seq_len, head_dim]
        mock_v = torch.randn(2, 4, 5, 8)
        legacy_cache = [(mock_k, mock_v)]
        assert get_past_key_values_length(legacy_cache) == 5

        # Test multiple layers
        legacy_cache_multi = [(mock_k, mock_v), (mock_k, mock_v), (mock_k, mock_v)]
        assert get_past_key_values_length(legacy_cache_multi) == 5

        # Test invalid input raises TypeError
        with pytest.raises(TypeError, match="Unsupported past_key_values type"):
            get_past_key_values_length("invalid")


def test_encoder_decoder_components():
        """Test basic encoder-decoder components."""
        # Create encoder-decoder config
        config = NanoConfig(
            vocab_size=1000,
            n_embd=128,
            n_head=8,
            n_layer=2,
            block_size=512,
            is_encoder_decoder=True
        )

        # Test encoder
        encoder = NanoEncoder(config)
        input_ids = torch.randint(0, 1000, (2, 32))  # batch_size=2, seq_len=32
        attention_mask = torch.ones(2, 32)

        encoder_outputs = encoder(input_ids, attention_mask)
        encoder_hidden = encoder_outputs.last_hidden_state
        assert encoder_hidden.shape == (2, 32, 128)

        # Test decoder
        decoder = NanoDecoder(config)
        decoder_input_ids = torch.randint(0, 1000, (2, 16))  # shorter target sequence

        decoder_outputs = decoder(
            decoder_input_ids,
            encoder_outputs=encoder_hidden,
            encoder_attention_mask=attention_mask
        )

        assert decoder_outputs['logits'].shape == (2, 16, 1000)


def test_encoder_decoder_models():
        """Test NanoEncoderModel and NanoDecoderModel wrappers."""
        # Create separate configs for encoder and decoder
        encoder_config = NanoConfig(
            vocab_size=1000,
            n_embd=128,
            n_head=8,
            n_layer=2,
            block_size=512,
            is_decoder=False,
            add_cross_attention=False
        )

        decoder_config = NanoConfig(
            vocab_size=1000,
            n_embd=128,
            n_head=8,
            n_layer=2,
            block_size=512,
            is_decoder=True,
            add_cross_attention=True
        )

        # Create model wrappers
        encoder_model = NanoEncoderModel(encoder_config)
        decoder_model = NanoDecoderModel(decoder_config)

        # Test encoder
        input_ids = torch.randint(0, 1000, (2, 32))
        attention_mask = torch.ones(2, 32)

        encoder_outputs = encoder_model(input_ids=input_ids, attention_mask=attention_mask)
        assert encoder_outputs.last_hidden_state.shape == (2, 32, 128)

        # Test decoder with cross-attention
        decoder_input_ids = torch.randint(0, 1000, (2, 16))
        decoder_attention_mask = torch.ones(2, 16)

        decoder_outputs = decoder_model(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            encoder_attention_mask=attention_mask,
            output_hidden_states=True,
        )

        assert decoder_outputs.hidden_states.shape == (2, 16, 128)
        assert decoder_outputs.logits.shape == (2, 16, 1000)
