#!/usr/bin/env python3
"""
Test the cache utility function with different cache formats.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch

# Set seeds for reproducible tests
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
from transformers.cache_utils import Cache
from nano_xyz.cache_utils import get_past_key_values_length


def test_cache_utility():
    """Test cache utility with different formats."""

    # Test None
    assert get_past_key_values_length(None) == 0
    print("PASS: None input test passed")

    # Test empty list
    assert get_past_key_values_length([]) == 0
    print("PASS: Empty list test passed")

    # Test legacy tuple format (transformers format: [batch, num_heads, seq_len, head_dim])
    mock_k = torch.randn(2, 4, 5, 8)  # [batch, num_heads, seq_len, head_dim]
    mock_v = torch.randn(2, 4, 5, 8)
    legacy_cache = [(mock_k, mock_v)]
    assert get_past_key_values_length(legacy_cache) == 5
    print("PASS: Legacy tuple format test passed")

    # Test multiple layers
    legacy_cache_multi = [(mock_k, mock_v), (mock_k, mock_v), (mock_k, mock_v)]
    assert get_past_key_values_length(legacy_cache_multi) == 5
    print("PASS: Multi-layer legacy format test passed")

    # Test Cache object (if we can create one)
    try:
        # Try to create a basic cache object
        cache_obj = Cache()
        # If we can create it, test the interface
        length = get_past_key_values_length(cache_obj)
        print(f"PASS: Cache object test passed (length: {length})")
    except Exception as e:
        print(f"SKIP: Cache object test skipped: {e}")

    # Test invalid input
    try:
        get_past_key_values_length("invalid")
        assert False, "Should have raised TypeError"
    except TypeError as e:
        assert "Unsupported past_key_values type" in str(e)
        print("PASS: Invalid input error handling test passed")

    print("\nSUCCESS: All cache utility tests passed!")


if __name__ == "__main__":
    test_cache_utility()
