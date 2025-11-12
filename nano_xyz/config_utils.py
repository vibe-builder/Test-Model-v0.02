"""
Configuration utilities for robust config serialization and wrapper handling.

Provides shared functionality for handling HuggingFace config serialization,
particularly around quantization configs and encoder-decoder compatibility.
"""

from typing import Dict, Any, Optional


class ConfigWrapper:
    """
    Dict-like wrapper for config objects to provide EncoderDecoderConfig compatibility.

    This wrapper allows config dictionaries to be used in places where HF's
    EncoderDecoderConfig expects objects with dict-like interfaces (pop, get, etc.).
    """

    def __init__(self, config_dict: Dict[str, Any]):
        """
        Initialize wrapper with a config dictionary.

        Args:
            config_dict: Configuration dictionary to wrap
        """
        self._dict = config_dict.copy()

    def pop(self, key: str, *args):
        """Dict-like pop method."""
        if args:
            return self._dict.pop(key, args[0])
        else:
            return self._dict.pop(key)

    def get(self, key: str, default=None):
        """Dict-like get method."""
        return self._dict.get(key, default)

    def __getitem__(self, key: str):
        """Dict-like item access."""
        return self._dict[key]

    def __contains__(self, key: str) -> bool:
        """Dict-like containment check."""
        return key in self._dict

    def keys(self):
        """Dict-like keys method."""
        return self._dict.keys()

    def __repr__(self) -> str:
        """String representation."""
        return f"ConfigWrapper({self._dict})"


def normalize_quantization_config_for_serialization(
    quantization_config: Optional[Any]
) -> Dict[str, Any]:
    """
    Normalize quantization config for safe serialization.

    Handles the common pattern where quantization_config might be None or empty,
    ensuring consistent serialization behavior across different config classes.

    Args:
        quantization_config: The quantization config to normalize

    Returns:
        Dict suitable for serialization (empty dict if None/empty)
    """
    if not quantization_config:
        return {}
    return quantization_config if isinstance(quantization_config, dict) else {}


def drop_empty_quant_config(result: Dict[str, Any], quant_config: Optional[Any]) -> Dict[str, Any]:
    """
    Drop empty quantization config from serialization result.

    This helper handles the common pattern of removing quantization_config from
    the serialized dict when it's None or empty, ensuring clean serialization.

    Args:
        result: The serialization result dict to modify
        quant_config: The quantization config value

    Returns:
        Modified result dict with quantization_config properly handled
    """
    if not quant_config:
        result.pop("quantization_config", None)
    else:
        result["quantization_config"] = quant_config
    return result


def handle_quantization_config_serialization(
    config_instance,
    quantization_config_attr: str = "quantization_config"
) -> Dict[str, Any]:
    """
    Handle quantization config serialization with proper None handling.

    This is a common pattern in HF configs where quantization_config needs special
    handling during serialization to avoid issues with None values.

    Args:
        config_instance: The config instance with quantization_config attribute
        quantization_config_attr: Name of the quantization config attribute

    Returns:
        Properly serialized config dict
    """
    # Get original quantization config
    original_quant_config = getattr(config_instance, quantization_config_attr, {})

    # Temporarily set to empty dict if None to avoid serialization issues
    if not original_quant_config:
        setattr(config_instance, quantization_config_attr, {})

    # Call parent serialization
    result = super(type(config_instance), config_instance).to_dict()

    # Restore original value
    setattr(config_instance, quantization_config_attr, original_quant_config)

    # Clean up result using the focused helper
    return drop_empty_quant_config(result, original_quant_config)
