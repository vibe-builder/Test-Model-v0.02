"""
Tests for HuggingFace ecosystem integration.

Tests cover:
- AutoModel loading and registration
- Model serialization/deserialization
- Config serialization/deserialization
- Cross-platform compatibility
"""

import pytest
import torch
import tempfile
import os
from unittest.mock import patch
from nano_xyz import NanoConfig, NanoForCausalLM, NanoModel, NanoEncoderModel, NanoDecoderModel


@pytest.fixture
def tiny_config():
    """Create a tiny config for fast testing."""
    return NanoConfig.from_preset("decoder_tiny")


@pytest.fixture
def encoder_config():
    """Create encoder config."""
    return NanoConfig.from_preset("encoder_decoder_tiny")


@pytest.fixture
def decoder_config():
    """Create decoder config."""
    return NanoConfig.from_preset("decoder_tiny")


@pytest.fixture
def tiny_model(tiny_config):
    """Create a tiny model for testing."""
    tiny_config.use_torch_compile = False  # Disable for testing
    return NanoForCausalLM(tiny_config)


@pytest.fixture
def encoder_model(encoder_config):
    """Create encoder model."""
    encoder_config.use_torch_compile = False
    return NanoEncoderModel(encoder_config)


@pytest.fixture
def decoder_model(decoder_config):
    """Create decoder model."""
    decoder_config.use_torch_compile = False
    return NanoDecoderModel(decoder_config)


@pytest.mark.parametrize("config_fixture", ["tiny_config", "encoder_config", "decoder_config"])
def test_config_serialization(request, config_fixture):
    """Test that NanoConfig can be serialized and deserialized."""
    config = request.getfixturevalue(config_fixture)

    # Serialize to dict
    config_dict = config.to_dict()
    assert isinstance(config_dict, dict)
    assert 'model_type' in config_dict
    assert config_dict['model_type'] == 'nano'

    # Test HF config method
    hf_config = config.to_hf_config()
    assert isinstance(hf_config, dict)
    assert hf_config['model_type'] == 'nano'

    # Test round-trip serialization
    new_config = NanoConfig(**config_dict)
    assert new_config.n_layer == config.n_layer
    assert new_config.n_head == config.n_head
    assert new_config.n_embd == config.n_embd


def test_config_from_dict(tiny_config):
    """Test creating config from dictionary."""
    config_dict = tiny_config.to_dict()

    # Create new config from dict
    new_config = NanoConfig(**config_dict)

    # Verify all attributes match
    assert new_config.n_layer == tiny_config.n_layer
    assert new_config.n_head == tiny_config.n_head
    assert new_config.n_embd == tiny_config.n_embd
    assert new_config.vocab_size == tiny_config.vocab_size


@pytest.mark.parametrize("model_fixture", ["tiny_model", "encoder_model", "decoder_model"])
def test_automodel_compatibility(request, model_fixture):
    """Test AutoModel loading compatibility."""
    model = request.getfixturevalue(model_fixture)

    # Test that model has required HF attributes
    assert hasattr(model, 'config')
    assert hasattr(model.config, 'model_type')
    assert model.config.model_type == 'nano'

    # Test forward pass
    input_ids = torch.randint(0, model.config.vocab_size, (1, 5))
    with torch.no_grad():
        outputs = model(input_ids)
        assert outputs is not None


def test_automodel_registration():
    """Test that Nano models are properly registered with AutoModel."""
    try:
        from transformers import AutoConfig

        # Test that we can create config locally
        config = NanoConfig(vocab_size=1000, n_layer=2, n_head=4, n_embd=128)
        assert config.model_type == 'nano'

        # Test that AutoConfig can instantiate our config class
        auto_config = AutoConfig.for_model('nano', vocab_size=1000, n_layer=2, n_head=4, n_embd=128)
        assert isinstance(auto_config, NanoConfig)
        assert auto_config.model_type == 'nano'

    except ImportError:
        pytest.skip("transformers not available")


@pytest.mark.parametrize("model_fixture", ["tiny_model", "encoder_model"])
def test_model_save_load_hf_format(request, model_fixture):
    """Test saving and loading models in HF format."""
    model = request.getfixturevalue(model_fixture)

    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = os.path.join(temp_dir, "test_model")

        # Save model
        model.save_pretrained(model_path)

        # Verify files exist
        assert os.path.exists(os.path.join(model_path, "config.json"))
        assert os.path.exists(os.path.join(model_path, "pytorch_model.bin")) or \
               os.path.exists(os.path.join(model_path, "model.safetensors"))

        # Load model
        loaded_model = type(model).from_pretrained(model_path)

        # Verify loaded model is equivalent
        assert loaded_model.config.n_layer == model.config.n_layer
        assert loaded_model.config.n_head == model.config.n_head
        assert loaded_model.config.n_embd == model.config.n_embd

        # Test forward pass equivalence
        input_ids = torch.randint(0, model.config.vocab_size, (1, 5))
        with torch.no_grad():
            original_output = model(input_ids)
            loaded_output = loaded_model(input_ids)

        # Compare appropriate output attributes
        if hasattr(original_output, 'logits') and hasattr(loaded_output, 'logits'):
            torch.testing.assert_close(original_output.logits, loaded_output.logits)
        elif hasattr(original_output, 'last_hidden_state') and hasattr(loaded_output, 'last_hidden_state'):
            torch.testing.assert_close(original_output.last_hidden_state, loaded_output.last_hidden_state)
        else:
            # At least ensure the outputs have the same structure
            assert type(original_output) == type(loaded_output)


def test_model_serialization(tiny_model):
    """Test saving and loading model."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "test_model")

        # Save model
        tiny_model.save_pretrained(model_path)

        # Verify files were created
        assert os.path.exists(os.path.join(model_path, "config.json"))
        assert os.path.exists(os.path.join(model_path, "pytorch_model.bin")) or \
               os.path.exists(os.path.join(model_path, "model.safetensors"))

        # Load model
        loaded_model = NanoForCausalLM.from_pretrained(model_path)

        # Verify it's equivalent
        assert loaded_model.config.n_embd == tiny_model.config.n_embd
        assert loaded_model.config.n_layer == tiny_model.config.n_layer

        # Test forward pass
        input_ids = torch.randint(0, tiny_model.config.vocab_size, (1, 5))
        with torch.no_grad():
            orig_output = tiny_model(input_ids)
            loaded_output = loaded_model(input_ids)

        # Outputs should be close (allowing for small numerical differences)
        torch.testing.assert_close(orig_output.logits, loaded_output.logits, rtol=1e-4, atol=1e-4)


def test_config_presets():
    """Test that all config presets work."""
    presets = ["decoder_tiny", "decoder_small", "decoder_medium"]

    for preset in presets:
        config = NanoConfig.from_preset(preset)
        assert isinstance(config, NanoConfig)

        # Should be able to create model
        config.use_torch_compile = False
        model = NanoForCausalLM(config)
        assert isinstance(model, NanoForCausalLM)


def test_model_type_registration():
    """Test that model type is properly registered."""
    config = NanoConfig.from_preset("decoder_tiny")

    # Model type should be 'nano'
    assert config.model_type == "nano"

    # Should be registered with AutoConfig
    try:
        from transformers import AutoConfig
        # model_type is already in config.to_dict(), so don't pass it again
        config_dict = config.to_dict()
        registered_config = AutoConfig.for_model(**config_dict)
        assert isinstance(registered_config, NanoConfig)
    except ImportError:
        pytest.skip("AutoConfig not available")


def test_generation_config_serialization(tiny_model):
    """Test that generation config is properly handled."""
    # Model should have generation config
    assert hasattr(tiny_model, 'generation_config')

    # Should be able to save with generation config
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "test_model")

        # Save
        tiny_model.save_pretrained(model_path)

        # Load
        loaded_model = NanoForCausalLM.from_pretrained(model_path)

        # Generation config should be preserved
        assert hasattr(loaded_model, 'generation_config')


def test_quantization_config_serialization():
    """Test that quantization config serialization works."""
    from nano_xyz import QuantizationConfig

    # Create config with quantization
    config = NanoConfig.from_preset("decoder_tiny")
    quant_config = QuantizationConfig(method="torchao", bits=8)
    config.quantization_config = quant_config

    # Should serialize without errors
    config_dict = config.to_dict()
    assert 'quantization_config' in config_dict

    # Should be able to recreate
    new_config = NanoConfig(**config_dict)
    assert new_config.quantization_config is not None


def test_model_forward_after_save_load(tiny_model):
    """Test that model works correctly after save/load cycle."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "test_model")

        # Save and load
        tiny_model.save_pretrained(model_path)
        loaded_model = NanoForCausalLM.from_pretrained(model_path)

        # Test various input shapes
        for seq_len in [1, 5, 10]:
            input_ids = torch.randint(0, tiny_model.config.vocab_size, (1, seq_len))

            with torch.no_grad():
                orig_output = tiny_model(input_ids)
                loaded_output = loaded_model(input_ids)

            # Should produce same results
            torch.testing.assert_close(orig_output.logits, loaded_output.logits, rtol=1e-4, atol=1e-4)


def test_config_edge_cases():
    """Test config handling of edge cases."""
    # Test with None quantization config
    config = NanoConfig.from_preset("decoder_tiny")
    config.quantization_config = None

    # Should serialize without errors
    config_dict = config.to_dict()
    assert config_dict.get('quantization_config') is None

    # Should be able to recreate
    new_config = NanoConfig(**config_dict)
    assert new_config.quantization_config is None


def test_hf_encoder_decoder_components():
    """Test individual HF-compatible encoder and decoder models."""
    # Create encoder config (standard model)
    encoder_config = NanoConfig(
        vocab_size=1000,
        n_embd=128,
        n_head=8,
        n_layer=2,
        block_size=512,
    )

    # Create decoder config (decoder with cross-attention)
    decoder_config = NanoConfig(
        vocab_size=1000,
        n_embd=128,
        n_head=8,
        n_layer=2,
        block_size=512,
        is_decoder=True,
        add_cross_attention=True,
    )

    # Test encoder model
    encoder = NanoEncoderModel(encoder_config)
    input_ids = torch.randint(0, 1000, (2, 32))
    attention_mask = torch.ones(2, 32)

    encoder_outputs = encoder(input_ids, attention_mask)
    assert encoder_outputs.last_hidden_state.shape == (2, 32, 128)

    # Test decoder model
    decoder = NanoDecoderModel(decoder_config)
    decoder_input_ids = torch.randint(0, 1000, (2, 16))

    decoder_outputs = decoder(
        decoder_input_ids,
        encoder_hidden_states=encoder_outputs.last_hidden_state,
        encoder_attention_mask=attention_mask
    )

    assert decoder_outputs.logits.shape == (2, 16, 1000)


def test_encoder_decoder_config():
    """Test encoder-decoder config setup (skipped - not implemented)."""
    pytest.skip("Encoder-decoder configuration not implemented for Nano models")
