# Nano XYZ Quantization Support
# ================================

import logging
from typing import Optional, Any, Dict, Literal, Union
from pydantic import BaseModel, Field, field_validator
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Check for optional dependencies
try:
    import bitsandbytes as bnb
    BNB_AVAILABLE = True
except ImportError:
    BNB_AVAILABLE = False
    logger.warning("bitsandbytes not available. 4-bit/8-bit quantization disabled.")

try:
    import torch.ao.quantization as ao
    AO_AVAILABLE = True
except ImportError:
    AO_AVAILABLE = False
    logger.warning("torch.ao.quantization not available. Dynamic quantization disabled.")


class QuantizationConfig(BaseModel):
    """
    Configuration for model quantization with torchao support.

    Supports multiple quantization backends and bit precisions for different
    trade-offs between model size, inference speed, and accuracy.

    Extended to support torchao dynamic quantization for optimal consumer hardware performance.

    Rationale: Different quantization methods work better for different use cases.
    Dynamic quantization with torchao provides 2-4x inference speedup on consumer GPUs
    while maintaining accuracy through runtime calibration.
    """
    method: str = Field("torchao", description="Quant backend: torchao, bnb")
    bits: int = Field(8, ge=4, le=8, description="Bit width for weights/activations")
    quant_type: str = Field("int8_dyn_act_int4_weight", description="TorchAO quant type: int8_dyn_act_int4_weight, float8_dyn_act_float8_weight")
    group_size: int = Field(32, ge=16, le=128, description="Group size for per-group quant")
    calibration_samples: int = Field(100, ge=10, description="Samples for dynamic calibration")

    # Legacy fields for backward compatibility
    compute_dtype: Optional[str] = None  # Compute dtype override
    double_quant: bool = False  # Nested quantization for bnb

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return self.dict()

    class Config:
        validate_assignment = True

    @field_validator('method')
    @classmethod
    def validate_method(cls, v):
        """
        Validate quantization method.

        Ensures only supported methods are used, with torchao as the primary
        method for dynamic quantization on consumer hardware.
        """
        if v not in ["torchao", "bnb"]:
            raise ValueError("Unsupported method; use torchao for dynamic quant")
        return v

    @field_validator('bits')
    @classmethod
    def validate_bits(cls, v):
        """
        Validate bit width.

        TorchAO supports 4-bit and 8-bit dynamic quantization.
        """
        return v

    @field_validator('quant_type')
    @classmethod
    def validate_quant_type(cls, v, info):
        """
        Validate quantization type based on method.

        Ensures only supported quantization types for the selected method are used.
        """
        method = info.data.get('method', 'torchao')

        if method == "torchao":
            supported_types = ["int8_dyn_act_int4_weight", "float8_dyn_act_float8_weight"]
            if v not in supported_types:
                raise ValueError(f"TorchAO quant_type must be one of: {supported_types}")
        elif method == "bnb":
            supported_types = ["nf4", "fp4"]
            if v not in supported_types:
                raise ValueError(f"bnb quant_type must be one of: {supported_types}")
        else:
            # For unknown methods, allow any string
            pass

        return v


class LinearFactory:
    """
    Factory for creating quantized linear layers.

    Dispatches to appropriate quantization backend based on configuration.
    Provides unified interface regardless of underlying quantization method.

    Rationale: Abstraction layer allows swapping quantization methods without
    changing model code, enabling easy experimentation and optimization.
    """

    def __init__(self, quant_config: Optional[Union[QuantizationConfig, Dict[str, Any]]] = None):
        """
        Initialize factory with quantization configuration.

        Args:
            quant_config: Quantization settings as object or dict. If None, uses standard Linear.
        """
        # Convert dict to QuantizationConfig if needed
        if isinstance(quant_config, dict):
            quant_config = QuantizationConfig(**quant_config)
        self.quant_config = quant_config

        # Validate backend availability
        if quant_config and quant_config.method == "bnb" and not BNB_AVAILABLE:
            raise ImportError("bitsandbytes required for bnb quantization")
        if quant_config and quant_config.method == "torchao" and not AO_AVAILABLE:
            raise ImportError("torchao required for torchao quantization")

    def create_linear(self, in_features: int, out_features: int, bias: bool = True) -> nn.Module:
        """
        Create linear layer with appropriate quantization.

        Args:
            in_features: Input dimension
            out_features: Output dimension
            bias: Whether to include bias term

        Returns:
            Quantized or standard linear layer
        """
        if self.quant_config is None:
            # No quantization - standard linear layer
            return nn.Linear(in_features, out_features, bias=bias)

        method = self.quant_config.method

        if method == "bnb":
            return self._create_bnb_linear(in_features, out_features, bias)
        elif method == "torchao":
            # For torchao, defer quantization to full model quantization
            # to avoid issues with weight initialization on quantized tensors
            return nn.Linear(in_features, out_features, bias=bias)
        else:
            raise ValueError(f"Unsupported quantization method: {method}")

    def _create_bnb_linear(self, in_features: int, out_features: int, bias: bool) -> nn.Module:
        """
        Create bitsandbytes quantized linear layer.

        bitsandbytes provides highly optimized 4-bit and 8-bit quantization
        with excellent performance on modern GPUs.
        """
        if self.quant_config.bits == 4:
            # 4-bit quantization with NF4 or FP4
            return bnb.nn.Linear4bit(
                in_features,
                out_features,
                bias=bias,
                quant_type=self.quant_config.quant_type,
                compute_dtype=getattr(torch, self.quant_config.compute_dtype) if self.quant_config.compute_dtype else None,
                compress_statistics=self.quant_config.double_quant,
            )
        elif self.quant_config.bits == 8:
            # 8-bit quantization
            return bnb.nn.Linear8bitLt(
                in_features,
                out_features,
                bias=bias,
                has_fp16_weights=False,  # Use int8 weights
                threshold=6.0,  # Outlier threshold
            )
        else:
            raise ValueError(f"bnb quantization supports 4-bit or 8-bit, got {self.quant_config.bits}")

    def _create_torchao_linear(self, in_features: int, out_features: int, bias: bool) -> nn.Module:
        """
        Create torchao quantized linear layer with dynamic quantization.

        TorchAO provides modern quantization with dynamic activation-weight quantization,
        optimized for consumer hardware and compatible with torch.compile.

        Supports both int8_dyn_act_int4_weight and float8_dyn_act_float8_weight configurations.
        """
        # Import torchao quantization
        try:
            import torchao.quantization as ao_quant
        except ImportError:
            raise ImportError("torchao required for torchao quantization")

        # Start with standard linear layer
        linear = nn.Linear(in_features, out_features, bias=bias)

        # Select quantization configuration based on quant_type
        if self.quant_config.quant_type == "int8_dyn_act_int4_weight":
            qconfig = ao_quant.Int8DynamicActivationInt4WeightConfig()
        elif self.quant_config.quant_type == "float8_dyn_act_float8_weight":
            qconfig = ao_quant.Float8DynamicActivationFloat8WeightConfig()
        else:
            raise ValueError(f"Unsupported torchao quant_type: {self.quant_config.quant_type}")

        # Apply quantization using quantize_ (modifies in-place and returns None)
        try:
            ao_quant.quantize_(linear, qconfig)
            # quantize_ modifies linear in-place, so we can return it directly
        except Exception as e:
            # Fallback to standard linear if quantization fails
            print(f"Warning: TorchAO quantization failed ({e}), using standard linear")
            return linear

        # Perform dynamic calibration to set scales
        # Generate calibration data
        calibration_data = torch.randn(
            self.quant_config.calibration_samples,
            in_features,
            dtype=torch.float32
        )

        # Run forward pass to calibrate (activations are quantized dynamically)
        try:
            with torch.no_grad():
                linear(calibration_data)
        except Exception as e:
            print(f"Warning: Calibration failed ({e}), proceeding without")

        return linear

    def create_embedding(self, num_embeddings: int, embedding_dim: int) -> nn.Module:
        """Create an embedding layer.

        Args:
            num_embeddings: Number of embeddings
            embedding_dim: Dimension of each embedding

        Returns:
            Standard nn.Embedding layer (quantization not typically applied to embeddings)
        """
        return nn.Embedding(num_embeddings, embedding_dim)
