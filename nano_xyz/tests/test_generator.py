"""Tests for generator components."""

import pytest
import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import nano_xyz
from nano_xyz.model import ModelSettings, ModelArchitecture
from nano_xyz.generator import TextGenerator


class TestTextGenerator:
    """Test text generation components."""

    @pytest.fixture
    def model_and_generator(self):
        """Create a small model and generator for testing."""
        config = ModelSettings(n_layer=2, n_head=4, n_embd=64, block_size=32)
        model = ModelArchitecture(config)
        generator = TextGenerator(model=model)
        return model, generator

    def test_empty_input_handling(self, model_and_generator):
        """Test handling of empty input."""
        model, generator = model_and_generator

        # Empty tensor
        empty_input = torch.empty(1, 0, dtype=torch.long)
        result = generator.generate(empty_input, max_new_tokens=5)
        assert result.shape[1] == 0  # Should return empty

    def test_input_truncation(self, model_and_generator):
        """Test input truncation for long sequences."""
        model, generator = model_and_generator

        # Create input longer than block_size
        long_input = torch.randint(0, generator.config.vocab_size, (1, generator.config.block_size + 10))
        result = generator.generate(long_input, max_new_tokens=5)

        max_context = generator.config.yarn_target_ctx if generator.config.use_yarn else generator.config.block_size
        if getattr(generator.config, "max_cache_len", None):
            max_context = max(max_context, generator.config.max_cache_len)
        expected = min(long_input.size(1), max_context) + 5
        assert result.shape[1] == expected

    def test_generation_basic(self, model_and_generator):
        """Test basic generation functionality."""
        model, generator = model_and_generator

        # Simple input
        input_ids = torch.randint(0, generator.config.vocab_size, (1, 5))
        result = generator.generate(input_ids, max_new_tokens=3)

        assert result.shape[0] == 1  # batch size
        assert result.shape[1] == 8  # original + generated tokens

    def test_parameter_validation(self, model_and_generator):
        """Test parameter validation."""
        model, generator = model_and_generator

        input_ids = torch.randint(0, generator.config.vocab_size, (1, 5))

        # Invalid temperature
        with pytest.raises(ValueError, match="temperature must be positive"):
            generator.generate(input_ids, max_new_tokens=5, temperature=0)

        # Invalid top_k
        with pytest.raises(ValueError, match="top_k must be positive"):
            generator.generate(input_ids, max_new_tokens=5, top_k=-1)

        # Invalid top_p
        with pytest.raises(ValueError, match="top_p must be in"):
            generator.generate(input_ids, max_new_tokens=5, top_p=1.5)

    def test_kv_cache_limits(self, model_and_generator):
        """Test KV cache size limiting."""
        model, generator = model_and_generator

        # Generate many tokens to trigger cache limiting
        input_ids = torch.randint(0, generator.config.vocab_size, (1, 5))
        result = generator.generate(input_ids, max_new_tokens=50)

        # Should complete without memory issues
        assert result.shape[1] == 55

    def test_iterative_generation_short_context(self):
        """Ensure iterative KV cache works for short contexts."""
        torch.manual_seed(0)
        config = ModelSettings(
            n_layer=2,
            n_head=4,
            n_embd=64,
            block_size=32,
            max_cache_len=32,
            use_yarn=True,
            yarn_orig_ctx=32,
            yarn_target_ctx=32,
        )
        model = ModelArchitecture(config)
        generator = TextGenerator(model=model)

        input_ids = torch.randint(0, config.vocab_size, (1, 8))
        result = generator.generate(input_ids, max_new_tokens=6)

        assert result.shape == (1, input_ids.size(1) + 6)
        assert result.shape[1] <= max(config.max_cache_len, config.yarn_target_ctx)

    def test_iterative_generation_long_context(self):
        """Ensure KV cache trims for long generations beyond cache window."""
        torch.manual_seed(1)
        config = ModelSettings(
            n_layer=2,
            n_head=4,
            n_embd=64,
            block_size=96,
            max_cache_len=64,
            use_yarn=True,
            yarn_orig_ctx=32,
            yarn_target_ctx=96,
        )
        model = ModelArchitecture(config)
        generator = TextGenerator(model=model)

        input_ids = torch.randint(0, config.vocab_size, (1, 48))  # exceeds block_size
        result = generator.generate(input_ids, max_new_tokens=80)

        expected_max_context = max(config.max_cache_len, config.yarn_target_ctx)
        # Total tokens should be clipped to context window
        assert result.shape[0] == 1
        assert result.shape[1] == expected_max_context

    def test_generation_long_sequence_regression(self):
        """Regression test for extended sequence generation (inspired by bug repro script)."""
        torch.manual_seed(2)
        config = ModelSettings(
            n_layer=2,
            n_head=4,
            n_embd=64,
            block_size=96,
            n_kv_groups=2,
            max_cache_len=64,
            use_yarn=True,
            yarn_orig_ctx=32,
            yarn_target_ctx=96,
        )
        model = ModelArchitecture(config)
        generator = TextGenerator(model=model)

        token_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        generated = generator.generate(token_ids, max_new_tokens=35, temperature=1.0)

        max_context = config.yarn_target_ctx if config.use_yarn else config.block_size
        if config.max_cache_len:
            max_context = max(max_context, config.max_cache_len)
        expected = min(token_ids.size(1) + 35, max_context)
        assert generated.shape == (1, expected)

    @pytest.mark.parametrize("temperature,top_k,top_p", [
        (1.0, None, None),
        (0.5, 10, None),
        (1.0, None, 0.9),
        (0.8, 20, 0.95),
    ])
    def test_sampling_parameters(self, model_and_generator, temperature, top_k, top_p):
        """Test different sampling parameters."""
        model, generator = model_and_generator

        input_ids = torch.randint(0, generator.config.vocab_size, (1, 5))
        result = generator.generate(
            input_ids,
            max_new_tokens=5,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )

        assert result.shape[1] == 10

    def test_generator_sampling_guardrails(self):
        """Test generator sampling guardrails: temperature clamping, stop token early exit."""
        config = ModelSettings(n_layer=2, n_head=4, n_embd=64, block_size=32)
        model = ModelArchitecture(config)
        generator = TextGenerator(model=model)

        input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)

        # Test temperature clamping (should clamp to [0.1, 5.0])
        # Test with temperature too low
        result_low = generator.generate(input_ids, max_new_tokens=2, temperature=0.01)
        assert result_low.shape[1] == 5  # input + generated

        # Test with temperature too high
        result_high = generator.generate(input_ids, max_new_tokens=2, temperature=10.0)
        assert result_high.shape[1] == 5

        # Test max_new_tokens = 0 (early return guard)
        result_zero = generator.generate(input_ids, max_new_tokens=0)
        assert torch.equal(result_zero, input_ids)

        # Test stop token early exit (using token 0 as stop, which might be generated)
        # This tests that the method doesn't crash with undefined variable
        result_stop = generator.generate(input_ids, max_new_tokens=3, stop_token=0)
        # Should complete without NameError crash
        assert isinstance(result_stop, torch.Tensor)
        assert result_stop.shape[0] == 1

        print("Generator sampling guardrails test passed!")

    def test_generate_batch_respects_stop_tokens(self, monkeypatch):
        """Ensure generate_batch forwards per-sequence stop tokens and slices outputs correctly."""
        config = ModelSettings(n_layer=2, n_head=4, n_embd=64, block_size=16)
        model = ModelArchitecture(config)
        generator = TextGenerator(model=model)

        invoked_stop_tokens = []

        def fake_generate(self, token_ids, max_new_tokens, temperature=1.0, top_k=None, top_p=None, stop_token=None):
            invoked_stop_tokens.append(stop_token)
            batch_size, seq_len = token_ids.shape
            new_vals = torch.full((batch_size, max_new_tokens), 9, dtype=token_ids.dtype, device=token_ids.device)
            return torch.cat([token_ids, new_vals], dim=1)

        monkeypatch.setattr(TextGenerator, "generate", fake_generate, raising=False)

        seq_a = torch.tensor([1, 2, 3], dtype=torch.long)
        seq_b = torch.tensor([4, 5], dtype=torch.long)
        stop_tokens = [42, None]

        outputs = generator.generate_batch(
            [seq_a, seq_b],
            max_new_tokens=3,
            temperature=0.8,
            top_k=5,
            top_p=0.9,
            stop_tokens=stop_tokens,
            pad_token_id=0
        )

        assert invoked_stop_tokens == stop_tokens
        assert len(outputs) == 2
        for out in outputs:
            assert torch.equal(out, torch.tensor([9, 9, 9], dtype=torch.long))

    def test_long_prompt_preserves_chronology(self):
        """Ensure prompts longer than block_size are consumed in original order."""
        config = ModelSettings(
            n_layer=2,
            n_head=4,
            n_embd=64,
            block_size=4,
            max_cache_len=16,
            use_yarn=True,
            yarn_orig_ctx=4,
            yarn_target_ctx=16,
        )
        model = ModelArchitecture(config)
        generator = TextGenerator(model=model)

        prompt = torch.arange(0, 8, dtype=torch.long, device=generator.device).unsqueeze(0)
        generated = generator.generate(prompt, max_new_tokens=1)

        assert torch.equal(
            generated[:, : prompt.size(1)].to("cpu"),
            prompt.to("cpu")
        )

    def test_max_new_tokens_zero_processes_overflow(self):
        """Ensure max_new_tokens=0 still ingests the entire prompt."""
        config = ModelSettings(
            n_layer=2,
            n_head=4,
            n_embd=64,
            block_size=4,
            max_cache_len=32,
            use_yarn=True,
            yarn_orig_ctx=4,
            yarn_target_ctx=32,
        )
        model = ModelArchitecture(config)
        generator = TextGenerator(model=model)

        prompt = torch.arange(0, 10, dtype=torch.long, device=generator.device).unsqueeze(0)
        result = generator.generate(prompt, max_new_tokens=0)

        assert result.shape[1] == prompt.size(1)
