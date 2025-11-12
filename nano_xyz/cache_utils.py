"""
Cache utilities for robust KV cache handling.

Provides a clean interface for cache introspection that works across different
cache implementations (Cache objects, legacy tuples, etc.).
"""

from typing import Optional, Union, Tuple, List
import torch
from transformers.cache_utils import Cache


def get_past_key_values_length(
    past_key_values: Optional[Union[Cache, List[Tuple[torch.Tensor, torch.Tensor]]]]
) -> int:
    """
    Get the sequence length from past key values, handling different cache formats.

    This function provides a clean, robust interface for cache introspection that
    works across different HuggingFace cache implementations:

    1. Modern Cache objects (with get_seq_length() method)
    2. Legacy tuple format (past_key_values[0][0].size(-2))
    3. DynamicCache or other cache types

    Args:
        past_key_values: Cache object, list of tuples, or None

    Returns:
        int: Number of tokens in the past key values (0 if None or empty)

    Raises:
        TypeError: If past_key_values is of an unexpected type
    """
    if past_key_values is None:
        return 0

    # Modern Cache interface (preferred)
    if isinstance(past_key_values, Cache):
        try:
            return past_key_values.get_seq_length()
        except (AttributeError, NotImplementedError):
            # Fallback to seen_tokens attribute if get_seq_length not available
            # Note: seen_tokens is deprecated in favor of cache_position
            if hasattr(past_key_values, 'seen_tokens') and past_key_values.seen_tokens is not None:
                return int(past_key_values.seen_tokens)
            # Final fallback - assume empty
            return 0

    # Legacy tuple format: List[Tuple[Tensor, Tensor]] or Tuple[Tensor, Tensor]
    if isinstance(past_key_values, (list, tuple)):
        if len(past_key_values) == 0:
            return 0

        first_layer = past_key_values[0]

        # Handle direct tuple of tensors: (k, v)
        if isinstance(first_layer, torch.Tensor):
            # Direct tuple format: (k, v) where k and v are tensors
            if first_layer.dim() >= 2:
                return int(first_layer.size(-2))  # Sequence dimension
            return 0

        # Standard format: [(k1, v1), (k2, v2), ...]
        if isinstance(first_layer, (list, tuple)) and len(first_layer) >= 1:
            first_key = first_layer[0]
            if isinstance(first_key, torch.Tensor) and first_key.dim() >= 2:
                return int(first_key.size(-2))  # Sequence dimension

        # Handle case where first_layer might be a Cache object
        if isinstance(first_layer, Cache):
            try:
                return first_layer.get_seq_length()
            except (AttributeError, NotImplementedError):
                if hasattr(first_layer, 'seen_tokens'):
                    return int(first_layer.seen_tokens)
                return 0

    # Unknown format - raise descriptive error
    raise TypeError(
        f"Unsupported past_key_values type: {type(past_key_values)}. "
        f"Expected Cache object or List[Tuple[Tensor, Tensor]], got {past_key_values}"
    )
