"""
Nano XYZ - Adaptive Computation Transformer with Dynamic Context Management.

Solves the universal LLM scaling problem: How to handle arbitrarily long contexts
without exponential memory/compute costs while maintaining efficiency.

Key Innovation: Dynamic Context Allocation (DCA)
- Intelligently allocates attention budget based on content importance
- Efficient feedforward networks with SwiGLU activation
- Maintains full context awareness without fixed window limitations
- Scales to millions of tokens while staying computationally efficient

Features:
- Feedforward Networks: Efficient SwiGLU-based feedforward layers
- Dynamic Attention: Budgeted attention allocation based on token significance
- Memory-Efficient Scaling: Handles 100K+ tokens on consumer GPUs
- Quantization-Aware: Optimized for 4-bit/8-bit deployment
- Research Platform: Extensible architecture for novel attention mechanisms

Modules:
- model.py: Core DCA architecture with adaptive attention and feedforward networks
- attention_utils.py: Advanced attention mechanisms with budget allocation
- configuration_nano.py: DCA-specific configuration with scaling parameters
- modeling_nano.py: HF-compatible interfaces with DCA generation methods
- quantization.py: Quantization support optimized for DCA efficiency
- scripts/train_hf.py: Training infrastructure with DCA-aware optimization
"""

__version__ = "1.0.0"

from .configuration_nano import NanoConfig
from .quantization import QuantizationConfig
from .base import NanoPreTrainedModel
from .model import NanoModel, RotaryEmbedding, SwiGLU, TransformerLayer, ModelOutput, NanoEncoder, NanoDecoder
from .modeling_nano import NanoForCausalLM, NanoEncoderModel, NanoDecoderModel
from .attention_utils import process_attention_mask

__all__ = [
    "NanoModel",
    "NanoPreTrainedModel",
    "RotaryEmbedding",
    "SwiGLU",
    "TransformerLayer",
    "ModelOutput",
    "NanoEncoder",
    "NanoDecoder",
    "NanoConfig",
    "QuantizationConfig",
    "NanoForCausalLM",
    "NanoEncoderModel",
    "NanoDecoderModel",
    "process_attention_mask",
]

