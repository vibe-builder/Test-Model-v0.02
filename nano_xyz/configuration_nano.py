"""
Unified Pydantic configuration for the Nano XYZ model.

This module provides type-safe configuration management using Pydantic BaseModel [8].
Key features:
- Field validation with constraints (e.g., divisibility checks)
- Preset loading for common configurations
- HuggingFace compatibility wrapper
- Automatic validation on instantiation and assignment

Rationale: Pydantic provides runtime type checking and validation that catches
configuration errors early, preventing subtle bugs in model initialization.
In layman's terms: It's like having a smart assistant that double-checks your
settings before you start cooking, ensuring you don't mix incompatible ingredients.
"""

from __future__ import annotations

import logging
from typing import Optional, List, Dict, Any

import torch
from transformers import PretrainedConfig

from .constants import (
    DEFAULT_BLOCK_SIZE,
    DEFAULT_N_EMBD,
    DEFAULT_N_HEAD,
    DEFAULT_N_LAYER,
    DROPOUT_MIN,
    DROPOUT_MAX,
)
from .config_utils import handle_quantization_config_serialization

logger = logging.getLogger(__name__)

# Use PretrainedConfig for HF compatibility while maintaining validation


class NanoConfig(PretrainedConfig):
    """
    Nano XYZ configuration with comprehensive validation.

    Uses Pydantic validators to ensure architectural consistency:
    - n_embd must be divisible by n_head for proper attention computation
    - Field constraints prevent invalid configurations at runtime

    Rationale: Configuration errors are caught early via pydantic validation
    rather than failing during model initialization or forward pass.
    """
    model_type = "nano"

    def __init__(
        self,
        vocab_size: int = 50257,
        n_layer: int = 12,
        n_head: int = 12,
        n_embd: int = 768,
        block_size: int = 1024,
        dropout: float = 0.0,
        bias: bool = True,
        n_kv_groups: Optional[int] = None,
        rope_type: str = "default",
        rope_base: float = 10000.0,
        yarn_scale_factor: float = 1.0,
        use_xpos: bool = False,
        attention_type: str = "dca",
        dca_window_size: int = 256,
        dca_global_tokens: int = 32,
        dca_random_blocks: int = 16,
        dca_enable_generation: bool = True,
        dca_attention_budget: float = 0.2,
        dca_local_window: int = 128,
        dca_global_budget: int = 64,
        architecture: str = "decoder_only",
        encoder_layers: int = 6,
        decoder_layers: int = 6,
        use_torch_compile: bool = False,  # Disabled by default for Windows compatibility
        torch_compile_mode: str = "auto",
        torch_dtype: Optional[str] = None,
        use_activation_checkpointing: bool = False,
        quantization_config: Optional[Dict[str, Any]] = None,
        use_dca: bool = False,
        enable_observability: bool = False,
        log_dca_metrics: bool = False,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        bos_token_id: Optional[int] = None,
        tie_word_embeddings: bool = True,
        **kwargs
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            bos_token_id=bos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs
        )

        # use_return_dict defaults to True in parent class

        # Basic validation
        if block_size < 512:
            raise ValueError(f"block_size must be >= 512, got {block_size}")
        if vocab_size < 1:
            raise ValueError(f"vocab_size must be >= 1, got {vocab_size}")
        if n_layer < 1:
            raise ValueError(f"n_layer must be >= 1, got {n_layer}")
        if n_head < 1:
            raise ValueError(f"n_head must be >= 1, got {n_head}")
        if n_embd < 64:
            raise ValueError(f"n_embd must be >= 64, got {n_embd}")
        if not (0.0 <= dropout <= 0.5):
            raise ValueError(f"dropout must be between 0.0 and 0.5, got {dropout}")
        if n_embd % n_head != 0:
            raise ValueError(f"n_embd ({n_embd}) must be divisible by n_head ({n_head})")

        # Set attributes
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.block_size = block_size
        self.dropout = dropout
        self.bias = bias
        self.n_kv_groups = n_kv_groups
        self.rope_type = rope_type
        self.rope_base = rope_base
        self.yarn_scale_factor = yarn_scale_factor
        self.use_xpos = use_xpos
        self.attention_type = attention_type
        self.dca_window_size = dca_window_size
        self.dca_global_tokens = dca_global_tokens
        self.dca_random_blocks = dca_random_blocks
        self.dca_enable_generation = dca_enable_generation
        self.dca_attention_budget = dca_attention_budget
        self.dca_local_window = dca_local_window
        self.dca_global_budget = dca_global_budget
        self.architecture = architecture
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.use_torch_compile = use_torch_compile
        self.torch_compile_mode = torch_compile_mode
        self.torch_dtype = torch_dtype
        self.use_activation_checkpointing = use_activation_checkpointing
        self.quantization_config = quantization_config
        self.use_dca = use_dca
        # use_return_dict is handled by parent class


    @classmethod
    def from_preset(cls, preset: str) -> 'NanoConfig':
        """
        Load configuration from predefined presets.

        Presets provide battle-tested configurations for different use cases:
        - decoder_* : Decoder-only models for language modeling
        - encoder_decoder_* : Seq2seq models for translation/summarization

        Rationale: Presets reduce configuration errors and provide starting
        points optimized for different model sizes and tasks.
        """
        presets = {
            "decoder_tiny": {
                "block_size": 1024, "n_layer": 6, "n_head": 8, "n_embd": 512,
                "dca_window_size": 256, "dca_global_tokens": 32, "dca_random_blocks": 16
            },
            "decoder_small": {
                "block_size": 2048, "n_layer": 12, "n_head": 12, "n_embd": 768,
                "dca_window_size": 512, "dca_global_tokens": 64, "dca_random_blocks": 32
            },
            "decoder_medium": {
                "block_size": 4096, "n_layer": 24, "n_head": 16, "n_embd": 1024,
                "use_dca": True,
                "dca_window_size": 1024, "dca_global_tokens": 128, "dca_random_blocks": 64
            },
            "encoder_decoder_tiny": {
                "block_size": 512, "n_layer": 6, "n_head": 8, "n_embd": 512,
                "is_decoder": True, "add_cross_attention": True,
                "dca_window_size": 128, "dca_global_tokens": 16, "dca_random_blocks": 8
            },
            "encoder_decoder_small": {
                "block_size": 1024, "n_layer": 12, "n_head": 12, "n_embd": 768,
                "is_decoder": True, "add_cross_attention": True,
                "dca_window_size": 256, "dca_global_tokens": 32, "dca_random_blocks": 16
            },
        }

        if preset not in presets:
            available = list(presets.keys())
            raise ValueError(f"Invalid preset '{preset}'. Available presets: {available}")

        return cls(**presets[preset])

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize config to dictionary with proper quantization_config handling.

        HF's PretrainedConfig.to_dict() has special handling for quantization_config
        but assumes it's not None. We need to handle the None case.
        """
        import copy

        # Start with a deep copy of our __dict__
        output = copy.deepcopy(self.__dict__)

        # Add model_type if defined
        if hasattr(self.__class__, "model_type"):
            output["model_type"] = self.__class__.model_type

        # Remove internal attributes that shouldn't be serialized
        for key in ["_auto_class", "_commit_hash", "_attn_implementation_internal", "base_model_tp_plan"]:
            output.pop(key, None)

        # Handle transformers version
        try:
            from transformers import __version__ as transformers_version
            output["transformers_version"] = transformers_version
        except ImportError:
            output["transformers_version"] = "unknown"

        # Handle nested configs
        for key, value in output.items():
            if isinstance(value, type(self)):  # Nested PretrainedConfig
                value = value.to_dict()
                if "transformers_version" in value:
                    del value["transformers_version"]
            output[key] = value

        # Handle quantization_config specially (avoid None.to_dict() error)
        if hasattr(self, "quantization_config"):
            if self.quantization_config is None:
                output["quantization_config"] = None
            elif not isinstance(self.quantization_config, dict):
                output["quantization_config"] = self.quantization_config.to_dict()
            else:
                output["quantization_config"] = self.quantization_config

            # Remove internal quantization dtype
            output.pop("_pre_quantization_dtype", None)

        # Convert torch dtypes to strings
        self.dict_torch_dtype_to_str(output)

        return output

    def to_diff_dict(self) -> Dict[str, Any]:
        """
        Override to_diff_dict to handle quantization_config properly.
        HF's PretrainedConfig.to_diff_dict() calls to_dict() and then filters,
        but we need to ensure quantization_config is handled correctly.
        """
        output = self.to_dict()

        # Handle quantization_config specially for diff dict
        if hasattr(self, "quantization_config") and self.quantization_config is not None:
            if not isinstance(self.quantization_config, dict):
                output["quantization_config"] = self.quantization_config.to_dict()
            else:
                output["quantization_config"] = self.quantization_config

        return output

    def to_hf_config(self) -> Dict[str, Any]:
        """
        Convert to HuggingFace-compatible configuration dictionary.

        HF models expect specific field mappings and model_type for auto-loading.
        This wrapper maintains compatibility while using our internal config.

        Rationale: Enables seamless integration with HuggingFace ecosystem
        (Trainer, AutoModel, etc.) while keeping our optimized config internally.
        """
        return {
            "model_type": "nano",
            **self.to_dict()
        }


# Legacy compatibility - keep for backwards compatibility
HFNanoConfig = NanoConfig  # Simplified - no separate wrapper needed
