"""Tests for model components."""

import pytest
import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import nano_xyz
from nano_xyz.model import ModelSettings, ModelArchitecture, RotaryPositionEmbedding, RopeWithYaRN, LCRBlock, GTRBlock


class TestModelComponents:
    """Test model components."""

    def test_forward_backward(self):
        """Test forward and backward pass."""
        config = ModelSettings(n_layer=2, n_head=4, n_embd=64, block_size=32)
        model = ModelArchitecture(config)

        # Create test input
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        # Forward pass
        logits, loss = model(input_ids, targets=input_ids)
        assert logits.shape == (batch_size, seq_len, config.vocab_size)
        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0

        # Backward pass
        loss.backward()
        assert any(p.grad is not None for p in model.parameters())

    def test_kv_cache(self):
        """Test KV cache functionality."""
        config = ModelSettings(n_layer=2, n_head=4, n_embd=64, block_size=32, max_cache_len=16)
        model = ModelArchitecture(config)

        batch_size, seq_len = 1, 10
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        # First forward pass with cache
        result1 = model(input_ids, use_cache=True)
        logits1, loss1, kv_cache = result1
        assert len(kv_cache) > 0  # Should have cache for transformer layers

        # Second forward pass with cache
        new_tokens = torch.randint(0, config.vocab_size, (batch_size, 5))
        result2 = model(new_tokens, use_cache=True, past_key_values=kv_cache)
        logits2, loss2, kv_cache2 = result2

        # Cache should be updated
        assert len(kv_cache2) == len(kv_cache)

    def test_rope_shapes(self):
        """Test RoPE with different input shapes."""
        rope = RotaryPositionEmbedding(dim=8, max_seq_len=10)

        # Test 4D input
        x_4d = torch.rand(2, 2, 4, 8)
        output_4d = rope(x_4d)
        assert output_4d.shape == (2, 2, 4, 8)

        # Test 3D input (should reshape)
        x_3d = torch.rand(2, 4, 8)
        output_3d = rope(x_3d)
        assert output_3d.shape == (2, 4, 8)

    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config
        config = ModelSettings(max_cache_len=64, block_size=128)
        assert config.max_cache_len == 64

        # Invalid config - max_cache_len > block_size
        with pytest.raises(ValueError, match="max_cache_len.*cannot exceed block_size"):
            ModelSettings(max_cache_len=128, block_size=64)

    @pytest.mark.parametrize("use_amp", [True, False])
    def test_mixed_precision(self, use_amp):
        """Test mixed precision training."""
        if use_amp and not torch.cuda.is_available():
            pytest.skip("CUDA required for AMP testing")

        config = ModelSettings(
            n_layer=2, n_head=4, n_embd=64, block_size=32,
            dtype='float16' if use_amp else 'float32'
        )
        model = ModelArchitecture(config)

        device = torch.device('cuda' if use_amp else 'cpu')
        model = model.to(device)

        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len)).to(device)
        targets = input_ids.clone()

        # Forward pass
        logits, loss = model(input_ids, targets=targets)
        assert logits.device == device
        assert loss.device == device

        # Backward pass
        loss.backward()
        assert any(p.grad is not None for p in model.parameters())

    def test_yarn_offsets_cache(self):
        """Test YaRN frequency offsets with KV cache functionality."""
        # Create test model config with YaRN
        config = ModelSettings(
            n_layer=2, n_head=4, n_embd=64, block_size=32,
            use_yarn=True, yarn_orig_ctx=2048, yarn_target_ctx=4096,
            max_cache_len=16
        )
        model = ModelArchitecture(config)

        batch_size, seq_len = 1, 10
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        # First forward pass (prefill) - should work with YaRN enabled
        result1 = model(input_ids, use_cache=True)
        assert len(result1) == 3  # Should return logits, loss, kv_cache
        logits1, _, kv_cache = result1
        assert logits1.shape == (batch_size, 1, config.vocab_size)  # Last token only by default

        # Second forward pass with cache (decode) - should work
        new_tokens = torch.randint(0, config.vocab_size, (batch_size, 5))
        result2 = model(new_tokens, use_cache=True, past_key_values=kv_cache)
        assert len(result2) == 3
        logits2, _, kv_cache2 = result2
        assert logits2.shape == (batch_size, 1, config.vocab_size)  # Last token only

        # Verify cache is updated
        assert len(kv_cache2) == len(kv_cache)

        # Test that YaRN is actually being used (model should have rope attribute)
        assert hasattr(model, 'rope')
        assert model.rope.enabled == True
        assert model.rope.target_ctx == 4096

    def test_sliding_window_with_sinks(self):
        """Test sliding window attention with attention sinks enabled."""
        # Create config with sliding window and attention sinks
        config = ModelSettings(
            n_layer=2, n_head=4, n_embd=64, block_size=32,
            sliding_window=8, use_attention_sinks=True, attention_sink_size=2,
            max_cache_len=16
        )
        model = ModelArchitecture(config)

        # Create input longer than sliding window
        batch_size, seq_len = 1, 20  # Longer than sliding window
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        # Forward pass - attention should work with sinks accessible
        logits, loss = model(input_ids, return_full_logits=True)
        assert logits.shape == (batch_size, seq_len, config.vocab_size)

        # Test with KV cache (simulate generation scenario)
        result = model(input_ids, use_cache=True, return_full_logits=False)
        logits_cached, _, kv_cache = result

        # Check that attention sinks are present in KV cache
        # First layer should have transformer
        transformer_layer_found = False
        for layer in model.transformer.layers:
            if hasattr(layer, 'attention') and hasattr(layer.attention, 'attention_sink_k'):
                transformer_layer_found = True
                # Sink K should be initialized and have correct shape
                sink_k = layer.attention.attention_sink_k
                sink_v = layer.attention.attention_sink_v
                expected_n_kv_groups = layer.attention.n_kv_heads  # Use actual value from layer
                assert sink_k.shape == (1, expected_n_kv_groups, config.attention_sink_size, layer.attention.head_size)
                assert sink_v.shape == (1, expected_n_kv_groups, config.attention_sink_size, layer.attention.head_size)
                break
        assert transformer_layer_found, "Should have transformer layer with attention sinks"

        # Verify sliding window mask allows sink access
        # This is tested implicitly by successful forward pass, but we can check mask properties
        attn_layer = None
        for layer in model.transformer.layers:
            if hasattr(layer, 'attention'):
                attn_layer = layer.attention
                break

        if attn_layer and hasattr(attn_layer, 'sliding_window_mask'):
            # Mask should have finite values for sink positions (first columns)
            sink_cols = attn_layer.sliding_window_mask[:, :config.attention_sink_size]
            assert torch.isfinite(sink_cols).all(), "Sink positions should not be masked"

    def test_fused_unfused_qkv_parity(self):
        """Test numerical parity between fused and unfused QKV projections."""
        batch_size, seq_len = 2, 8

        # Test with fused attention
        config_fused = ModelSettings(
            n_layer=1, n_head=4, n_embd=64, block_size=16,
            use_fused_attention=True
        )
        model_fused = ModelArchitecture(config_fused)

        # Test with unfused attention
        config_unfused = ModelSettings(
            n_layer=1, n_head=4, n_embd=64, block_size=16,
            use_fused_attention=False
        )
        model_unfused = ModelArchitecture(config_unfused)

        # Copy weights to ensure fair comparison
        # Copy embeddings and output projection
        model_unfused.transformer.token_embeddings.load_state_dict(
            model_fused.transformer.token_embeddings.state_dict()
        )
        model_unfused.language_model_head.load_state_dict(
            model_fused.language_model_head.state_dict()
        )

        # Copy transformer layer weights (attention and MLP)
        fused_layer = None
        unfused_layer = None
        for layer in model_fused.transformer.layers:
            if hasattr(layer, 'attention'):
                fused_layer = layer
                break
        for layer in model_unfused.transformer.layers:
            if hasattr(layer, 'attention'):
                unfused_layer = layer
                break

        if fused_layer and unfused_layer:
            # Copy attention weights - need to convert fused to separate
            fused_qkv = fused_layer.attention.qkv_projection.weight.data
            n_head = config_fused.n_head
            n_kv_heads = config_fused.n_kv_groups
            head_size = config_fused.n_embd // n_head

            # Split fused weights back to separate
            query_size = n_head * head_size
            kv_size = n_kv_heads * head_size

            unfused_layer.attention.query_projection.weight.data = fused_qkv[:query_size]
            unfused_layer.attention.key_projection.weight.data = fused_qkv[query_size:query_size + kv_size]
            unfused_layer.attention.value_projection.weight.data = fused_qkv[query_size + kv_size:]

            # Copy biases if present
            if hasattr(fused_layer.attention.qkv_projection, 'bias') and fused_layer.attention.qkv_projection.bias is not None:
                fused_bias = fused_layer.attention.qkv_projection.bias.data
                unfused_layer.attention.query_projection.bias.data = fused_bias[:query_size]
                unfused_layer.attention.key_projection.bias.data = fused_bias[query_size:query_size + kv_size]
                unfused_layer.attention.value_projection.bias.data = fused_bias[query_size + kv_size:]

            # Copy output projection
            unfused_layer.attention.output_projection.load_state_dict(
                fused_layer.attention.output_projection.state_dict()
            )

            # Copy MLP weights
            unfused_layer.pre_attention_norm.load_state_dict(fused_layer.pre_attention_norm.state_dict())
            unfused_layer.feed_forward.load_state_dict(fused_layer.feed_forward.state_dict())
            unfused_layer.pre_mlp_norm.load_state_dict(fused_layer.pre_mlp_norm.state_dict())

        # Create identical input
        input_ids = torch.randint(0, config_fused.vocab_size, (batch_size, seq_len))
        targets = input_ids.clone()

        # Forward pass with both models
        logits_fused, loss_fused = model_fused(input_ids, targets=targets)
        logits_unfused, loss_unfused = model_unfused(input_ids, targets=targets)

        # Assert numerical parity
        torch.testing.assert_close(logits_fused, logits_unfused, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(loss_fused, loss_unfused, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("n_layer", [3, 6, 8, 12, 15])
    def test_reasoning_block_indices(self, n_layer):
        """Test LCR/GTR block placement and non-overlap for various depths."""
        # Test with reasoning blocks enabled
        config = ModelSettings(
            n_layer=n_layer, n_head=4, n_embd=64, block_size=32,
            use_lcr=True, use_gtr=True, gtr_num_tokens=4
        )
        model = ModelArchitecture(config)

        lcr_indices = []
        gtr_indices = []
        transformer_indices = []

        for i, layer in enumerate(model.transformer.layers):
            if isinstance(layer, LCRBlock):
                lcr_indices.append(i)
            elif isinstance(layer, GTRBlock):
                gtr_indices.append(i)
            else:
                transformer_indices.append(i)

        # Check placement logic
        if n_layer >= 6:
            # LCR should be at layer 4 if depth allows
            if n_layer >= 6:
                assert 4 in lcr_indices, f"LCR should be at layer 4 for n_layer={n_layer}, got {lcr_indices}"

            # GTR placement depends on depth
            if n_layer >= 12:
                assert 9 in gtr_indices, f"GTR should be at layer 9 for n_layer={n_layer}, got {gtr_indices}"
            elif n_layer >= 6:
                expected_gtr = n_layer - 3
                assert expected_gtr in gtr_indices, f"GTR should be at layer {expected_gtr} for n_layer={n_layer}, got {gtr_indices}"

            # Ensure no overlap
            all_reasoning_indices = set(lcr_indices + gtr_indices)
            assert len(all_reasoning_indices) == len(lcr_indices) + len(gtr_indices), \
                f"Overlap detected: LCR={lcr_indices}, GTR={gtr_indices}"

            # Ensure reasoning blocks don't exceed total layers
            assert max(all_reasoning_indices) < n_layer, f"Reasoning block index exceeds n_layer: {max(all_reasoning_indices)} >= {n_layer}"
        else:
            # For shallow models, reasoning should be disabled
            assert len(lcr_indices) == 0, f"LCR should be disabled for n_layer={n_layer}"
            assert len(gtr_indices) == 0, f"GTR should be disabled for n_layer={n_layer}"

        # Ensure all layers are accounted for
        total_layers = len(lcr_indices) + len(gtr_indices) + len(transformer_indices)
        assert total_layers == n_layer, f"Layer count mismatch: {total_layers} != {n_layer}"

    def test_gtr_padding_mask(self):
        """Test GTR padding mask application prevents noise from padded tokens."""
        # Create config with GTR enabled
        config = ModelSettings(
            n_layer=6, n_head=4, n_embd=64, block_size=16,
            use_gtr=True, gtr_num_tokens=4
        )
        model = ModelArchitecture(config)

        batch_size, seq_len = 2, 8
        # Create input with some padding (attention_mask with 0s)
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        # Attention mask: first sequence has padding at end
        attention_mask = torch.tensor([
            [1, 1, 1, 1, 1, 0, 0, 0],  # First 5 tokens real, last 3 padding
            [1, 1, 1, 1, 1, 1, 1, 1]   # All real tokens
        ], dtype=torch.long)

        # Forward pass with attention mask
        logits_masked, loss_masked = model(input_ids, attention_mask=attention_mask)

        # Forward pass without attention mask (all tokens treated as real)
        logits_unmasked, loss_unmasked = model(input_ids)

        # Outputs should differ due to padding mask application
        # The first sequence's padding positions should not influence GTR outputs
        assert not torch.allclose(logits_masked, logits_unmasked, atol=1e-6), \
            "Outputs should differ when padding mask is applied"

        # Loss should also differ (padding tokens ignored in loss computation)
        assert not torch.allclose(loss_masked, loss_unmasked, atol=1e-6), \
            "Loss should differ when padding mask is applied"

        # Test that GTR layer exists and processed the mask
        gtr_layer_found = False
        for layer in model.transformer.layers:
            if isinstance(layer, GTRBlock):
                gtr_layer_found = True
                break
        assert gtr_layer_found, "GTR layer should be present in the model"
