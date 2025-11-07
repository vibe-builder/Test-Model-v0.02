"""Regression tests for P0/P1 fixes."""

import pytest
import torch
import sys
import os
import tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import nano_xyz
from nano_xyz.model import ModelSettings, ModelArchitecture
from nano_xyz.generator import TextGenerator
from nano_xyz.checkpoint import CheckpointManager


class TestRegressionFixes:
    """Regression tests for critical fixes."""

    def test_checkpoint_gtr_num_heads_regression(self):
        """Regression test for checkpoint AttributeError with gtr_num_heads."""
        config = ModelSettings(n_layer=2, n_head=4, n_embd=64, block_size=32, use_gtr=True)
        model = ModelArchitecture(config)

        # Create dummy optimizer and scheduler
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)

        with tempfile.TemporaryDirectory() as tmpdir:
            # This should not raise AttributeError: 'ModelSettings' object has no attribute 'gtr_num_heads'
            checkpoint_path = CheckpointManager.save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=1,
                step=100,
                loss=0.5,
                checkpoint_dir=tmpdir,
                filename='test_regression.pt'
            )

            # Should be able to load without issues
            loaded_checkpoint = CheckpointManager.load_checkpoint(
                checkpoint_path, model=model, optimizer=optimizer, scheduler=scheduler
            )

            # Verify gtr_num_heads is not in saved config
            assert 'gtr_num_heads' not in loaded_checkpoint['config']

            print("Checkpoint gtr_num_heads regression test passed!")

    def test_generator_early_stop_regression(self):
        """Regression test for generator NameError with undefined 'i' variable."""
        config = ModelSettings(n_layer=2, n_head=4, n_embd=64, block_size=32)
        model = ModelArchitecture(config)
        generator = TextGenerator(model=model)

        input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)

        # Test max_new_tokens=0 early return (should not crash)
        result = generator.generate(input_ids, max_new_tokens=0)
        assert torch.equal(result, input_ids)

        # Test normal generation with stop token (should not crash with NameError)
        result = generator.generate(input_ids, max_new_tokens=3, stop_token=0)
        assert isinstance(result, torch.Tensor)

        print("Generator early stop regression test passed!")

    def test_attention_sink_mask_regression(self):
        """Regression test for sliding window + attention sinks interaction."""
        config = ModelSettings(
            n_layer=2, n_head=4, n_embd=64, block_size=32,
            sliding_window=8, use_attention_sinks=True, attention_sink_size=2
        )
        model = ModelArchitecture(config)

        # Should initialize without AttributeError
        input_ids = torch.randint(0, config.vocab_size, (1, 20))

        # Forward pass should work
        logits, loss = model(input_ids)
        assert logits.shape == (1, 1, config.vocab_size)  # Last token by default

        # With KV cache should also work
        result = model(input_ids, use_cache=True)
        assert len(result) == 3

        print("Attention sink mask regression test passed!")

    def test_gtr_padding_mask_regression(self):
        """Regression test for GTR padding mask application."""
        config = ModelSettings(n_layer=3, n_head=4, n_embd=64, block_size=16, use_gtr=True)
        model = ModelArchitecture(config)

        # Create input with padding
        batch_size, seq_len = 2, 8
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        attention_mask = torch.tensor([
            [1, 1, 1, 1, 1, 0, 0, 0],  # First sequence has padding
            [1, 1, 1, 1, 1, 1, 1, 1]   # Second sequence is full
        ], dtype=torch.long)

        # Should work without errors
        logits_masked, loss_masked = model(input_ids, attention_mask=attention_mask)
        logits_unmasked, loss_unmasked = model(input_ids)

        # Results should be different due to padding mask
        assert not torch.allclose(logits_masked, logits_unmasked, atol=1e-6)

        print("GTR padding mask regression test passed!")

    def test_generator_prefill_truncation_regression(self):
        """Regression test for generator prefill truncation with YaRN."""
        config = ModelSettings(
            n_layer=2, n_head=4, n_embd=64, block_size=32,
            use_yarn=True, yarn_orig_ctx=2048, yarn_target_ctx=4096
        )
        model = ModelArchitecture(config)
        generator = TextGenerator(model=model)

        # Create input longer than block_size but within yarn_target_ctx
        long_input = torch.randint(0, config.vocab_size, (1, config.block_size + 10))
        assert long_input.size(1) > config.block_size
        assert long_input.size(1) <= config.yarn_target_ctx

        # Should gracefully handle prefill and allow generation
        result = generator.generate(long_input, max_new_tokens=2)
        max_context = config.yarn_target_ctx if config.use_yarn else config.block_size
        if getattr(config, "max_cache_len", None):
            max_context = max(max_context, config.max_cache_len)
        expected = min(long_input.size(1) + 2, max_context)
        assert result.shape[1] == expected

        print("Generator prefill truncation regression test passed!")
