"""Tests for processor components."""

import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import nano_xyz
from nano_xyz.processor import TextProcessor


class TestTextProcessor:
    """Test text processing components."""

    def test_tokenizer_initialization(self):
        """Test tokenizer initialization and special tokens."""
        processor = TextProcessor(tokenizer_name="gpt2")

        # Check vocab size
        assert processor.vocab_size > 0

        # Check special tokens (should have defaults)
        assert processor.eos_token_id is not None
        assert processor.pad_token_id == 0  # Default pad token

    def test_encode_decode(self):
        """Test encoding and decoding."""
        processor = TextProcessor(tokenizer_name="gpt2")

        text = "Hello world!"
        tokens = processor.encode(text)
        decoded = processor.decode(tokens)

        assert isinstance(tokens, list)
        assert len(tokens) > 0
        assert isinstance(decoded, str)

    def test_token_clamping(self):
        """Test token clamping warnings."""
        processor = TextProcessor(tokenizer_name="gpt2")

        # Create tokens that are out of range
        invalid_tokens = [0, 1, 999999, processor.vocab_size + 1]

        with pytest.warns(UserWarning, match="Clamped.*tokens"):
            _ = processor.decode(invalid_tokens)

        # Test excessive clamping (>5%)
        # We can't easily test this without mocking, so skip for now
        pass

    def test_special_tokens_defaults(self):
        """Test that special tokens have safe defaults."""
        processor = TextProcessor(tokenizer_name="gpt2")

        # Should expose an EOS token even if tokenizer lacks one
        assert processor.eos_token_id is not None

        # Test with a mock tokenizer that has no eos_token
        # This is hard to test without creating a custom tokenizer, so skip for now
        pass
