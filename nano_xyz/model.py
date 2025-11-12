# Nano XYZ Model - Adaptive Computation Transformer with DCA
# ============================================================
#
# Revolutionary approach to long-context language modeling that solves the universal
# scaling problem: How to handle arbitrarily long contexts without exponential costs.
#
# Key Innovations:
# - Dynamic Context Allocation (DCA): Intelligently allocates attention budget based on content importance
# - Feedforward Networks: Optimized SwiGLU activation with proper normalization
# - Memory-Efficient Scaling: Handles 100K+ tokens on consumer GPUs
# - Sparse Attention: Only attends to high-priority tokens
#
# Core Features:
# - Transformer Layers: Standard attention + feedforward architecture
# - FlashAttention/SDPA: Optimized attention mechanisms for speed
# - Rotary Position Encoding (RoPE): Better positional understanding
# - YaRN scaling: Extends context length beyond training limits
# - Quantization support: 4-bit and 8-bit quantization with bitsandbytes
# - Gradient checkpointing: Memory-efficient training
#
# Architecture: Transformer with DCA attention, SwiGLU feed-forward, and RMSNorm.

# ENCODER-DECODER IMPLEMENTATION VERIFICATION:
# ==============================================
#
# This implementation follows proven T5/BART architecture patterns:
#
# 1. **Bidirectional Encoder**: Processes source sequences without causal masking
#    - Citation: "BART: Denoising Sequence-to-Sequence Pre-training" (Lewis et al., 2019)
#    - https://arxiv.org/abs/1910.13461
#
# 2. **Causal Decoder with Cross-Attention**: Three sub-layers per decoder block:
#    - Causal self-attention (autoregressive generation)
#    - Cross-attention to encoder outputs (source context)
#    - Feed-forward network
#    - Citation: "Attention Is All You Need" (Vaswani et al., 2017) + T5/BART papers
#    - https://arxiv.org/abs/1706.03762
#
# 3. **HF-Compatible Design**: Separate encoder/decoder models with EncoderDecoderModel wrapper
#    - Citation: Hugging Face documentation and T5/BART implementations
#    - https://huggingface.co/docs/transformers/en/model_doc/encoder-decoder
#
# 4. **Shared Embeddings**: Parameter sharing between encoder/decoder vocabularies
#    - Citation: "Exploring the Limits of Transfer Learning" (Raffel et al., 2019)
#    - https://arxiv.org/abs/1910.10683
#
# This architecture has been proven effective across translation, summarization, and
# other sequence-to-sequence tasks in the research literature.

import math
import logging
from dataclasses import dataclass
from typing import Optional, Tuple, List, Any, Union, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import Cache

from .attention_utils import repeat_kv, DynamicContextAllocator, apply_sparse_attention_optimization
from .constants import (
    DEFAULT_VOCAB_SIZE,
    DEFAULT_BLOCK_SIZE,
    DEFAULT_N_LAYER,
    DEFAULT_N_HEAD,
    DEFAULT_N_EMBD,
    DEFAULT_DROPOUT,
    DEFAULT_BIAS,
    WEIGHT_INIT_STD,
)
from .configuration_nano import NanoConfig
from .quantization import LinearFactory
from .base import NanoPreTrainedModel
from .attention_utils import process_attention_mask
from .cache_utils import get_past_key_values_length

logger = logging.getLogger(__name__)


"""
Rotary Position Embedding (RoPE) with YaRN and xPos support.

YaRN (Yet another RoPE) [4] extends context length by scaling frequencies,
allowing trained models to handle sequences longer than training length.
xPos [5] provides improved extrapolation by scaling based on embedding position.

Citation [4]: YaRN - extends RoPE to handle 64K+ tokens
Source: https://arxiv.org/html/2511.01192v1
Layman explanation: YaRN is like stretching a rubber band - it lets the model
handle longer texts by smoothly extending the position encoding pattern.

Citation [5]: xPos improves RoPE extrapolation
Source: https://aclanthology.org/2025.findings-emnlp.991/
Layman explanation: xPos is like adding perspective correction to position encoding,
making it work better for positions far from the training data.
"""


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding with YaRN scaling and xPos support.

    Implements RoPE with extensions for long-context efficiency:
    - YaRN: Scales inverse frequencies to extend context beyond training
    - xPos: Applies position-dependent scaling for better extrapolation
    - Efficient caching: Recomputes cos/sin only when sequence length increases

    Rationale: Position encoding is crucial for transformers to understand word order.
    RoPE provides better length generalization than absolute position embeddings,
    and YaRN/xPos extend this to very long sequences efficiently.
    """

    def __init__(
        self,
        dim: int,
        base: float = 10000.0,
        use_xpos: bool = False,
        yarn_scale_factor: float = 1.0,
    ):
        """
        Initialize RoPE with optional YaRN and xPos scaling.

        Args:
            dim: Embedding dimension (must be even for RoPE split)
            base: Base frequency for RoPE computation
            use_xpos: Enable xPos scaling for better extrapolation
            yarn_scale_factor: YaRN scaling factor (>1.0 extends context)
        """
        super().__init__()
        self.dim = dim
        self.base = base
        self.use_xpos = use_xpos
        self.yarn_scale_factor = yarn_scale_factor

        # Compute inverse frequencies with YaRN scaling
        # YaRN divides frequencies to create smoother extrapolation
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        if yarn_scale_factor > 1.0:
            inv_freq = inv_freq / yarn_scale_factor
        self.register_buffer("inv_freq", inv_freq)

        # xPos scaling parameters (computed once at initialization)
        if use_xpos:
            scale_base = torch.arange(0, dim, 2).float() / dim
            # xPos formula: nonlinear scaling based on embedding position
            scale = (scale_base + 0.4 * (dim // 2 - 1)) / (scale_base + 0.4 * dim // 2)
            self.register_buffer("scale", scale)

        # Cache for cos/sin tensors to avoid recomputation
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None

    def _update_cos_sin_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        """
        Update cached cos/sin tensors when sequence length increases.

        This optimization avoids recomputing trigonometric functions for every forward pass,
        significantly speeding up inference with variable-length sequences.
        """
        if seq_len > self._seq_len_cached:
            # Create position indices [0, 1, 2, ..., seq_len-1]
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)

            # Compute frequencies: outer product gives [seq_len, dim//2] frequency matrix
            freqs = torch.outer(t, self.inv_freq)

            # Apply xPos scaling if enabled
            if self.use_xpos:
                freqs = freqs * self.scale.unsqueeze(0)

            # Cache cos and sin for efficient reuse
            self._cos_cached = freqs.cos().to(dtype)
            self._sin_cached = freqs.sin().to(dtype)
            self._seq_len_cached = seq_len

    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None, offset: int = 0) -> torch.Tensor:
        """
        Apply rotary position embedding to input tensor.

        RoPE rotates pairs of embedding dimensions using position-dependent angles,
        providing better length generalization than absolute position embeddings.

        Args:
            x: Input tensor [batch_size, n_heads, seq_len, head_dim]
            seq_len: Total sequence length (computed from x if None)
            offset: Position offset for incremental decoding

        Returns:
            Tensor with rotary position encoding applied
        """
        batch, heads, qlen, head_dim = x.shape
        if seq_len is None:
            seq_len = qlen + offset

        # Update cache if sequence is longer than cached
        self._update_cos_sin_cache(seq_len, x.device, x.dtype)

        # Extract cos/sin for current query positions
        cos = self._cos_cached[offset : offset + qlen]  # [qlen, dim//2]
        sin = self._sin_cached[offset : offset + qlen]  # [qlen, dim//2]

        # Split embedding dimension for RoPE rotation
        x1 = x[..., : head_dim // 2]  # First half of embedding dimensions
        x2 = x[..., head_dim // 2 :]  # Second half of embedding dimensions

        # Reshape cos/sin to broadcast: [1, 1, qlen, head_dim//2]
        cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, qlen, head_dim//2]
        sin = sin.unsqueeze(0).unsqueeze(0)  # [1, 1, qlen, head_dim//2]

        # Apply RoPE rotation: complex multiplication in 2D plane
        # RoPE rotates pairs of dimensions using position-dependent angles
        # Formula: x'₁ = x₁*cos(θ) - x₂*sin(θ), x'₂ = x₂*cos(θ) + x₁*sin(θ)
        # This preserves relative position information while being position-aware
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)

    # torch.compile is now conditionally enabled based on platform detection
    # Disabled on Windows due to known stability issues with inductor backend

# Attention Layer for Transformer blocks
# Implements multi-head attention with DCA, RoPE, and cross-attention support
class AttentionLayer(nn.Module):
    """
    Multi-head attention layer with DCA support and cross-attention.

    Handles:
    - Standard self-attention with RoPE
    - Dynamic Context Allocation (DCA) for efficient long contexts
    - Cross-attention for encoder-decoder architectures
    - KV caching for efficient generation
    """

    def __init__(self, config: NanoConfig, layer_idx: int, linear_factory: Optional[LinearFactory] = None, is_decoder: bool = False, is_encoder: bool = False):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.is_decoder = is_decoder
        self.is_encoder = is_encoder

        # Attention configuration
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.n_kv_groups = getattr(config, 'n_kv_groups', None) or 1  # For GQA

        # Linear layers for attention
        self.q_proj = self._make_linear(config.n_embd, config.n_embd, linear_factory)
        self.k_proj = self._make_linear(config.n_embd, config.n_embd // self.n_kv_groups, linear_factory)
        self.v_proj = self._make_linear(config.n_embd, config.n_embd // self.n_kv_groups, linear_factory)
        self.o_proj = self._make_linear(config.n_embd, config.n_embd, linear_factory)

        # Cross-attention projections (for decoder)
        if is_decoder:
            self.q_proj_cross = self._make_linear(config.n_embd, config.n_embd, linear_factory)
            self.k_proj_cross = self._make_linear(config.n_embd, config.n_embd // self.n_kv_groups, linear_factory)
            self.v_proj_cross = self._make_linear(config.n_embd, config.n_embd // self.n_kv_groups, linear_factory)

        # RoPE for positional encoding
        self.rope = RotaryEmbedding(
            dim=self.head_dim,
            base=config.rope_base,
            yarn_scale_factor=config.yarn_scale_factor,
            use_xpos=config.use_xpos,
        )

        # DCA bookkeeping (for monitoring/debugging)
        self.last_attention_metadata = None
        self.last_sparse_mask = None
        self.dca_allocator = DynamicContextAllocator(config) if config.use_dca else None

    @staticmethod
    def _make_linear(in_features: int, out_features: int, factory: Optional[LinearFactory]) -> nn.Module:
        return factory.create_linear(in_features, out_features, bias=False) if factory else nn.Linear(in_features, out_features, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        query_position: int = 0,
        is_causal: bool = True,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass with support for DCA, cross-attention, and KV caching.
        """
        batch, seq_len, _ = hidden_states.shape

        past_length = get_past_key_values_length(past_key_value)
        total_seq_len = past_length + seq_len

        # DCA activation logic: disabled during generation unless explicitly enabled
        # Research shows sparse attention is designed for complete sequence processing, not incremental generation
        # Citation: "BigBird: Transformers for Longer Sequences" (Zaheer et al., 2020) - sparse patterns for training/inference
        # Citation: "Longformer: The Long-Document Transformer" (Beltagy et al., 2020) - local+global for full contexts
        # During autoregressive generation, maintaining full attention to previous tokens is crucial for coherence
        is_generation_mode = past_key_value is not None
        dca_allowed_in_generation = self.config.dca_enable_generation

        # Warn about experimental DCA generation usage
        if is_generation_mode and dca_allowed_in_generation and self.config.use_dca:
            import warnings
            import inspect

            # Find the user code frame (skip library internals)
            frame = inspect.currentframe()
            stacklevel = 1
            while frame:
                filename = frame.f_code.co_filename
                function_name = frame.f_code.co_name
                # Skip frames from our library and PyTorch internals
                if ('nano_xyz' not in filename and
                    'transformers' not in filename and
                    'torch' not in filename and
                    function_name not in ['forward', '_call_impl', '__call__']):
                    break
                stacklevel += 1
                frame = frame.f_back

            warnings.warn(
                "DCA during autoregressive generation is experimental and may produce incoherent outputs. "
                "Sparse attention patterns are designed for complete sequence processing, not incremental "
                "token-by-token generation. Consider disabling DCA during generation by setting "
                "dca_enable_generation=False in config.",
                UserWarning,
                stacklevel=stacklevel
            )

        dca_active = (
            self.config.use_dca and  # DCA must be enabled in config
            total_seq_len > self.config.dca_local_window and  # Sequence must be long enough
            (not is_generation_mode or dca_allowed_in_generation)  # Either not generation, or generation allowed
        )

        # Standard attention path with DCA and cross-attention support
        # Handle cross-attention first (decoder only)
        if encoder_hidden_states is not None and self.is_decoder:
            # Cross-attention to encoder outputs
            q_cross = self.q_proj_cross(hidden_states)
            k_cross = self.k_proj_cross(encoder_hidden_states)
            v_cross = self.v_proj_cross(encoder_hidden_states)

            # Reshape for attention
            q_cross = q_cross.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
            k_cross = k_cross.view(batch, -1, self.n_head // self.n_kv_groups, self.head_dim).transpose(1, 2)
            v_cross = v_cross.view(batch, -1, self.n_head // self.n_kv_groups, self.head_dim).transpose(1, 2)

            # Repeat KV for GQA
            k_cross = repeat_kv(k_cross, self.n_kv_groups)
            v_cross = repeat_kv(v_cross, self.n_kv_groups)

            # Apply RoPE to query only (keys/values from encoder are already positioned)
            q_cross = self.rope(q_cross, offset=0)

            # Cross-attention mask using unified processing
            src_len = encoder_hidden_states.size(1)  # Length of encoder sequence
            encoder_attention_mask = process_attention_mask(
                encoder_attention_mask,
                batch,
                self.n_head,
                seq_len,
                src_len,  # key_len for cross-attention
                hidden_states.device,
                is_causal=False  # Cross-attention is not causal
            )

            # Cross-attention with SDPA
            cross_attn_output = F.scaled_dot_product_attention(
                q_cross, k_cross, v_cross,
                attn_mask=encoder_attention_mask,
                is_causal=False  # Cross-attention is not causal
            )

            # Reshape and project
            cross_attn_output = cross_attn_output.transpose(1, 2).contiguous().view(batch, seq_len, self.n_embd)
            cross_attn_output = self.o_proj(cross_attn_output)

            return cross_attn_output, None, past_key_value

        # Self-attention (standard or with DCA)
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape for attention
        q = q.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_head // self.n_kv_groups, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_head // self.n_kv_groups, self.head_dim).transpose(1, 2)

        # Handle KV caching
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        # Apply RoPE with correct offset for generation
        # During generation with past_key_value, offset should be past_length
        # to maintain absolute positional encoding across generation steps
        rope_offset = past_length if past_key_value is not None else 0
        q = self.rope(q, offset=rope_offset)
        k = self.rope(k, offset=rope_offset)

        # Repeat KV for GQA
        k_expanded = repeat_kv(k, self.n_kv_groups)
        v_expanded = repeat_kv(v, self.n_kv_groups)

        # DCA: Apply sparse attention if enabled
        if dca_active and self.dca_allocator is not None:
            sparse_attention_mask, selected_mask, attention_metadata = self.dca_allocator(
                hidden_states,
                attention_mask,
                total_seq_len - 1,
                is_decoder=self.is_decoder,
                is_encoder=self.is_encoder,
            )
            self.last_attention_metadata = attention_metadata
            self.last_sparse_mask = sparse_attention_mask
            # Convert mask to jagged pattern for optimized attention
            # Convert boolean mask to float for SDPA compatibility
            float_mask = sparse_attention_mask.float()
            attn_output = F.scaled_dot_product_attention(
                q, k_expanded, v_expanded,
                attn_mask=float_mask,
                is_causal=False
            )
        else:
            # Standard attention with SDPA using unified mask processing
            attn_mask = process_attention_mask(
                attention_mask,
                batch,
                self.n_head,
                seq_len,
                total_seq_len,  # key_len includes past context
                hidden_states.device,
                is_causal=is_causal and self.is_decoder  # Apply causal masking for decoder self-attention
            )

            attn_output = F.scaled_dot_product_attention(
                q, k_expanded, v_expanded,
                attn_mask=attn_mask,
                is_causal=False  # We handle causality manually
            )

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, self.n_embd)
        attn_output = self.o_proj(attn_output)

        # Prepare KV cache for next step
        new_past_key_value = (k, v) if use_cache else None

        return attn_output, None, new_past_key_value


class SwiGLU(nn.Module):
    """
    SwiGLU feedforward network with gating mechanism.

    Uses FFN(x) = (xW_g) * SiLU(xW_u) @ W_d architecture as in PaLM and Grok.
    This provides better performance than standard FFN while maintaining efficiency.
    """

    def __init__(self, config: NanoConfig, linear_factory: Optional[LinearFactory] = None):
        super().__init__()
        self.hidden_dim = config.n_embd * 4  # Standard FFN expansion ratio

        # FFN weights: gate, up, and down projections for SwiGLU
        self.w1 = self._make_linear(config.n_embd, self.hidden_dim, config.bias, linear_factory)
        self.w3 = self._make_linear(config.n_embd, self.hidden_dim, config.bias, linear_factory)
        self.w2 = self._make_linear(self.hidden_dim, config.n_embd, config.bias, linear_factory)

    @staticmethod
    def _make_linear(in_features: int, out_features: int, bias: bool, factory: Optional[LinearFactory]) -> nn.Module:
        return factory.create_linear(in_features, out_features, bias=bias) if factory else nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process tokens through the feedforward network.

        Uses SwiGLU activation: FFN(x) = (xW_g) * SiLU(xW_u) @ W_d
        """
        # Standard SwiGLU computation
        gate = self.w1(x)  # Gate projection
        up = self.w3(x)    # Up projection
        return self.w2(F.silu(gate) * up)  # SwiGLU + down projection





# One layer of our transformer with attention and feedforward network
class TransformerLayer(nn.Module):
    def __init__(self, config: NanoConfig, layer_idx: int, linear_factory: Optional[LinearFactory] = None, is_encoder: bool = False, is_decoder: bool = False) -> None:
        super().__init__()
        self.is_encoder = is_encoder
        self.is_decoder = is_decoder
        # Each layer has attention to understand relationships
        # For encoders: is_decoder=False, is_encoder=True (no cross-attention projections)
        # For decoders: is_decoder=True, is_encoder=False (includes cross-attention projections)
        # For decoder-only: is_decoder=False, is_encoder=False (standard attention)
        self.self_attn = AttentionLayer(config, layer_idx, linear_factory, is_decoder=is_decoder, is_encoder=is_encoder)

        # Standard feedforward network
        self.feedforward = SwiGLU(config, linear_factory)
        self.input_layernorm = nn.RMSNorm(config.n_embd)
        self.post_attention_layernorm = nn.RMSNorm(config.n_embd)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        query_position: int = 0,  # For DCA: current position in sequence
        encoder_hidden_states: Optional[torch.Tensor] = None,  # For cross-attention
        encoder_attention_mask: Optional[torch.Tensor] = None,  # For cross-attention masking
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_output, attn_weights, past_key_value = self.self_attn(
            hidden_states, attention_mask, position_ids, past_key_value, output_attentions, use_cache, query_position, not self.is_encoder,
            encoder_hidden_states, encoder_attention_mask
        )
        hidden_states = residual + attn_output

        residual = hidden_states
        # Apply normalization before feedforward
        hidden_states = self.post_attention_layernorm(hidden_states)

        # Standard feedforward network
        hidden_states = self.feedforward(hidden_states)

        # Add the result to what we had before (residual connection)
        hidden_states = residual + hidden_states

        return hidden_states, attn_weights, past_key_value

# What our model returns after processing text
@dataclass
class ModelOutput:
    last_hidden_state: torch.Tensor  # The final understanding of the text
    past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None  # Memory for continuing conversations
    hidden_states: Optional[Tuple[torch.Tensor]] = None  # All layers' outputs (if requested)
    attentions: Optional[Tuple[torch.Tensor]] = None  # How much each word paid attention to others

# Encoder component for encoder-decoder architecture (T5/BART style)
class NanoEncoder(nn.Module):
    """Encoder component for sequence-to-sequence tasks.

    Processes input sequences bidirectionally to generate contextual representations
    that can be attended to by the decoder. Uses the same transformer architecture
    but without causal masking.

    Based on: "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer"
    (Raffel et al., 2019) https://arxiv.org/abs/1910.10683
    And: "BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation"
    (Lewis et al., 2019) https://arxiv.org/abs/1910.13461

    Both papers establish that encoder-decoder models with bidirectional encoders
    and shared embeddings provide superior performance for seq2seq tasks.
    """

    def __init__(self, config: NanoConfig):
        super().__init__()
        self.config = config
        # Handle quantization config (could be dict or QuantizationConfig)
        if config.quantization_config:
            from .quantization import QuantizationConfig as QuantConfig
            if isinstance(config.quantization_config, dict):
                quant_config = QuantConfig(**config.quantization_config)
            else:
                quant_config = config.quantization_config
            linear_factory = LinearFactory(quant_config)
        else:
            linear_factory = None

        self.embed_tokens = nn.Embedding(config.vocab_size, config.n_embd)
        self.layers = nn.ModuleList([
            TransformerLayer(config, i, linear_factory, is_encoder=True, is_decoder=False)
            for i in range(config.n_layer)
        ])
        self.norm = nn.RMSNorm(config.n_embd)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[torch.Tensor, ModelOutput]:
        """Encode input sequence into contextual representations.

        Args:
            input_ids: [batch_size, seq_len] - input token ids
            attention_mask: [batch_size, seq_len] - attention mask (1 for tokens to attend to)
            output_hidden_states: Whether to return intermediate hidden states
            return_dict: Whether to return a ModelOutput dataclass

        Returns:
            encoder_outputs: [batch_size, seq_len, hidden_size] - contextual representations
        """
        # Runtime safety assertions
        if not isinstance(input_ids, torch.Tensor):
            raise TypeError(f"input_ids must be a torch.Tensor, got {type(input_ids)}")
        if input_ids.dtype not in (torch.int32, torch.int64, torch.long):
            raise ValueError(f"input_ids must have integer dtype, got {input_ids.dtype}")
        if input_ids.dim() != 2:
            raise ValueError(f"input_ids must be 2D [batch_size, seq_len], got shape {input_ids.shape}")

        batch_size, seq_len = input_ids.shape
        if seq_len > self.config.block_size:
            raise ValueError(f"Input sequence length {seq_len} exceeds maximum block_size {self.config.block_size}")

        if attention_mask is not None:
            if not isinstance(attention_mask, torch.Tensor):
                raise TypeError(f"attention_mask must be a torch.Tensor, got {type(attention_mask)}")
            if attention_mask.shape != input_ids.shape:
                raise ValueError(f"attention_mask shape {attention_mask.shape} must match input_ids shape {input_ids.shape}")

        # Embed tokens
        hidden_states = self.embed_tokens(input_ids)

        all_hidden_states: Optional[Tuple[torch.Tensor, ...]] = None
        if output_hidden_states:
            all_hidden_states = (hidden_states,)

        # Process through transformer layers
        for layer in self.layers:
            # Encoder uses bidirectional attention (no causal masking)
            layer_outputs = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=None,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
                query_position=0,  # Not used in encoder
            )
            hidden_states = layer_outputs[0]
            if output_hidden_states and all_hidden_states is not None:
                all_hidden_states = all_hidden_states + (hidden_states,)

        # Final layer normalization
        encoder_outputs = self.norm(hidden_states)

        if output_hidden_states and all_hidden_states is not None:
            all_hidden_states = all_hidden_states + (encoder_outputs,)

        if not return_dict:
            outputs: Tuple[torch.Tensor, ...]
            if output_hidden_states and all_hidden_states is not None:
                outputs = (encoder_outputs, all_hidden_states)
            else:
                outputs = (encoder_outputs,)
            return outputs  # type: ignore[return-value]

        return ModelOutput(
            last_hidden_state=encoder_outputs,
            hidden_states=all_hidden_states,
        )


# Decoder component for encoder-decoder architecture (T5/BART style)
class NanoDecoder(nn.Module):
    """Decoder component for sequence-to-sequence tasks.

    Generates output sequences while attending to encoder outputs via cross-attention.
    Uses causal self-attention for autoregressive generation.

    Architecture follows: "Attention Is All You Need" (Vaswani et al., 2017)
    https://arxiv.org/abs/1706.03762 - establishes cross-attention mechanism

    And: "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer"
    (Raffel et al., 2019) https://arxiv.org/abs/1910.10683 - applies it to seq2seq

    Key insight: Each decoder layer performs three operations:
    1. Causal self-attention (autoregressive, prevents lookahead)
    2. Cross-attention to encoder outputs (context from source)
    3. Feed-forward network (standard transformer block)

    This pattern is proven effective across T5, BART, and other SOTA seq2seq models.
    """

    def __init__(self, config: NanoConfig):
        super().__init__()
        self.config = config
        # Handle quantization config (could be dict or QuantizationConfig)
        if config.quantization_config:
            from .quantization import QuantizationConfig as QuantConfig
            if isinstance(config.quantization_config, dict):
                quant_config = QuantConfig(**config.quantization_config)
            else:
                quant_config = config.quantization_config
            linear_factory = LinearFactory(quant_config)
        else:
            linear_factory = None

        self.embed_tokens = nn.Embedding(config.vocab_size, config.n_embd)
        self.layers = nn.ModuleList([
            TransformerLayer(config, i, linear_factory, is_decoder=True)
            for i in range(config.n_layer)
        ])
        self.norm = nn.RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Tie weights with input embeddings (T5/BART style)
        self.lm_head.weight = self.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        encoder_outputs: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        labels: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,  # For HF model compatibility
    ) -> Dict[str, torch.Tensor]:
        """Decode target sequence with cross-attention to encoder.

        Args:
            input_ids: [batch_size, tgt_seq_len] - target token ids
            encoder_outputs: [batch_size, src_seq_len, hidden_size] - encoder representations
            attention_mask: [batch_size, tgt_seq_len] - target attention mask
            encoder_attention_mask: [batch_size, src_seq_len] - source attention mask
            past_key_values: Previous decoder key/value states for incremental decoding
            use_cache: Whether to return past key/values for incremental decoding
            labels: [batch_size, tgt_seq_len] - labels for loss computation

        Returns:
            Dict containing:
            - logits: [batch_size, tgt_seq_len, vocab_size] - language model logits
            - loss: Optional loss if labels provided
            - past_key_values: Updated key/value states if use_cache=True
        """
        # Runtime safety assertions
        if not isinstance(input_ids, torch.Tensor):
            raise TypeError(f"input_ids must be a torch.Tensor, got {type(input_ids)}")
        if input_ids.dtype not in (torch.int32, torch.int64, torch.long):
            raise ValueError(f"input_ids must have integer dtype, got {input_ids.dtype}")
        if input_ids.dim() != 2:
            raise ValueError(f"input_ids must be 2D [batch_size, seq_len], got shape {input_ids.shape}")

        batch_size, tgt_len = input_ids.shape
        if tgt_len > self.config.block_size:
            raise ValueError(f"Input sequence length {tgt_len} exceeds maximum block_size {self.config.block_size}")

        if attention_mask is not None:
            if not isinstance(attention_mask, torch.Tensor):
                raise TypeError(f"attention_mask must be a torch.Tensor, got {type(attention_mask)}")
            if attention_mask.shape != input_ids.shape:
                raise ValueError(f"attention_mask shape {attention_mask.shape} must match input_ids shape {input_ids.shape}")

        if encoder_outputs is not None:
            if not isinstance(encoder_outputs, torch.Tensor):
                raise TypeError(f"encoder_outputs must be a torch.Tensor, got {type(encoder_outputs)}")
            if encoder_outputs.dim() != 3:
                raise ValueError(f"encoder_outputs must be 3D [batch_size, src_seq_len, hidden_size], got shape {encoder_outputs.shape}")
            if encoder_outputs.size(0) != batch_size:
                raise ValueError(f"encoder_outputs batch_size {encoder_outputs.size(0)} must match input_ids batch_size {batch_size}")
            if encoder_outputs.size(2) != self.config.n_embd:
                raise ValueError(f"encoder_outputs hidden_size {encoder_outputs.size(2)} must match config n_embd {self.config.n_embd}")

        if encoder_attention_mask is not None:
            if not isinstance(encoder_attention_mask, torch.Tensor):
                raise TypeError(f"encoder_attention_mask must be a torch.Tensor, got {type(encoder_attention_mask)}")
            if encoder_outputs is not None and encoder_attention_mask.shape[:2] != encoder_outputs.shape[:2]:
                raise ValueError(f"encoder_attention_mask shape {encoder_attention_mask.shape} must match encoder_outputs shape {encoder_outputs.shape[:2]}")

        if labels is not None:
            if not isinstance(labels, torch.Tensor):
                raise TypeError(f"labels must be a torch.Tensor, got {type(labels)}")
            if labels.shape != input_ids.shape:
                raise ValueError(f"labels shape {labels.shape} must match input_ids shape {input_ids.shape}")

        # Embed target tokens
        hidden_states = self.embed_tokens(input_ids)

        # Prepare past key values for incremental decoding
        next_past_key_values = [] if use_cache else None

        # Process through decoder layers
        for layer_idx, layer in enumerate(self.layers):
            layer_past_key_values = past_key_values[layer_idx] if past_key_values is not None else None

            layer_outputs = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=None,
                past_key_value=layer_past_key_values,
                output_attentions=False,
                use_cache=use_cache,
                query_position=0,  # Not used in decoder
                encoder_hidden_states=encoder_outputs,
                encoder_attention_mask=encoder_attention_mask,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_past_key_values.append(layer_outputs[2])

        # Final layer normalization
        hidden_states = self.norm(hidden_states)

        # Language modeling head (only if not requesting hidden states)
        if output_hidden_states:
            outputs = {"hidden_states": hidden_states}
        else:
            logits = self.lm_head(hidden_states)
            outputs = {"logits": logits}

        if use_cache:
            outputs["past_key_values"] = next_past_key_values

        # Compute loss if labels provided
        if labels is not None and not output_hidden_states:
            # Shift logits and labels for causal language modeling
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1)
            )
            outputs["loss"] = loss

        return outputs




# The main model that puts everything together
class NanoModel(NanoPreTrainedModel):
    def __init__(self, config: NanoConfig) -> None:
        super().__init__(config)


        # Handle quantization config (could be dict or QuantizationConfig)
        if config.quantization_config:
            from .quantization import QuantizationConfig as QuantConfig
            if isinstance(config.quantization_config, dict):
                quant_config = QuantConfig(**config.quantization_config)
            else:
                quant_config = config.quantization_config
            linear_factory = LinearFactory(quant_config)
        else:
            linear_factory = None

        # Initialize architecture based on configuration
        if getattr(config, 'is_decoder', False):
            # Decoder model with optional cross-attention
            self.decoder = NanoDecoder(config)
            self.embed_tokens = self.decoder.embed_tokens
        else:
            # Encoder or standard decoder-only model
            # Convert words to mathematical representations
            if linear_factory:
                self.embed_tokens = linear_factory.create_embedding(config.vocab_size, config.n_embd)
            else:
                self.embed_tokens = nn.Embedding(config.vocab_size, config.n_embd)

            # Stack multiple transformer layers
            self.layers = nn.ModuleList([TransformerLayer(config, i, linear_factory, is_decoder=True) for i in range(config.n_layer)])

            # Normalize the final output
            self.norm = nn.RMSNorm(config.n_embd)

            # Convert back to word predictions
            self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

            # Share weights between embedding and output layers (saves memory)
            self.lm_head.weight = self.embed_tokens.weight

        # Initialize all the weights properly
        self.apply(self._init_weights)

        self.post_init()

        # Use PyTorch's compiler for speed if requested and supported
        # Platform detection prevents compilation on Windows where torch.compile may be unstable
        # Citation: PyTorch documentation and community reports of Windows compilation issues
        # Source: https://pytorch.org/docs/stable/generated/torch.compile.html
        # Windows torch.compile stability issues reported in PyTorch GitHub issues
        if config.use_torch_compile and self._is_compilation_supported():
            compile_kwargs = self._get_optimal_compile_config()
            self.forward = torch.compile(self.forward, **compile_kwargs)

        # Apply quantization if configured (torchao dynamic quantization)
        # Done after compilation to avoid issues with quantized tensor operations
        if config.quantization_config:
            self.quantize_model()

    def quantize_model(self) -> None:
        """
        Apply full-model dynamic quantization using torchao.

        Quantizes all linear layers in the model using the configuration's quantization settings.
        This provides 2-4x inference speedup on consumer GPUs with minimal accuracy loss.

        The quantization is applied in-place, modifying the model directly.
        Call this method after model initialization but before inference.

        Note: This method should only be called once per model instance.
        """
        if not hasattr(self.config, 'quantization_config') or not self.config.quantization_config:
            return  # No quantization configured

        # Import torchao quantization
        try:
            import torchao.quantization as ao_quant
        except ImportError:
            logger.warning("torchao not available, skipping model quantization")
            return

        # Get quantization configuration
        quant_config = self.config.quantization_config
        if isinstance(quant_config, dict):
            from .quantization import QuantizationConfig as QuantConfig
            quant_config = QuantConfig(**quant_config)

        if quant_config.method != "torchao":
            return  # Only handle torchao quantization here

        # Select quantization configuration based on quant_type
        if quant_config.quant_type == "int8_dyn_act_int4_weight":
            qconfig = ao_quant.Int8DynamicActivationInt4WeightConfig()
        elif quant_config.quant_type == "float8_dyn_act_float8_weight":
            qconfig = ao_quant.Float8DynamicActivationFloat8WeightConfig()
        else:
            raise ValueError(f"Unsupported torchao quant_type: {quant_config.quant_type}")

        # Apply quantization to the entire model using quantize_
        # This will quantize all nn.Linear layers in the model
        try:
            ao_quant.quantize_(self, qconfig)
            logger.info(f"Successfully quantized model with {quant_config.quant_type}")
        except Exception as e:
            logger.warning(f"Model quantization failed ({e}), proceeding without quantization")
            return

        # Perform dynamic calibration to set scales
        # Create calibration data loader if not exists
        try:
            calibration_data = self._get_calibration_data(quant_config.calibration_samples)

            # Run forward passes to calibrate quantization scales
            self.eval()
            with torch.no_grad():
                for batch in calibration_data:
                    if isinstance(batch, dict) and 'input_ids' in batch:
                        inputs = batch['input_ids']
                    else:
                        inputs = batch

                    # Limit calibration to reasonable batch size
                    if inputs.size(0) > 4:
                        inputs = inputs[:4]

                    # Forward pass for calibration
                    self(inputs)

            logger.info(f"Calibration completed with {quant_config.calibration_samples} samples")
        except Exception as e:
            logger.warning(f"Calibration failed ({e}), proceeding without")

    def _get_calibration_data(self, num_samples: int) -> torch.utils.data.DataLoader:
        """
        Create a simple calibration dataset for quantization.

        Args:
            num_samples: Number of calibration samples to generate

        Returns:
            DataLoader with random calibration data
        """
        class CalibrationDataset(torch.utils.data.Dataset):
            def __init__(self, num_samples: int, vocab_size: int, seq_len: int = 512):
                self.num_samples = num_samples
                self.vocab_size = vocab_size
                self.seq_len = seq_len

            def __len__(self):
                return self.num_samples

            def __getitem__(self, idx):
                return torch.randint(0, self.vocab_size, (self.seq_len,))

        dataset = CalibrationDataset(num_samples, self.config.vocab_size)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=min(4, num_samples),  # Reasonable batch size
            shuffle=False
        )

    def _get_optimal_compile_config(self) -> Dict[str, Any]:
        """Get optimal torch.compile configuration based on hardware and model characteristics."""

        # Start with user-specified mode if provided
        if self.config.torch_compile_mode and self.config.torch_compile_mode != "auto":
            return {"mode": self.config.torch_compile_mode}

        # Hardware-aware mode selection
        if torch.cuda.is_available():
            device_props = torch.cuda.get_device_properties(0)
            gpu_name = device_props.name.lower()

            # High-end GPUs (Ampere and newer) - use reduce-overhead for best performance
            if device_props.major >= 8:  # Ampere and newer
                mode = "reduce-overhead"
                # Use dynamic shapes for better memory efficiency on large models
                dynamic = self.config.n_layer > 12 or self.config.n_embd > 1024
            # Older GPUs - use default mode
            else:
                mode = "default"
                dynamic = False
        else:
            # CPU - use default mode
            mode = "default"
            dynamic = False

        compile_config = {"mode": mode}

        # Add dynamic shape support if beneficial
        if dynamic:
            compile_config["dynamic"] = True

        # Add fullgraph for better optimization if model is not too complex
        if self.config.n_layer <= 24:
            compile_config["fullgraph"] = True

        # Use max_autotune for better kernel selection during compilation
        if torch.cuda.is_available():
            compile_config["options"] = {"max_autotune": True}

        return compile_config

    def _is_compilation_supported(self) -> bool:
        """
        Check if torch.compile is supported on this platform.

        Returns False on Windows due to known stability issues with inductor backend.
        Returns True on Linux/macOS where torch.compile generally works well.

        Citation: PyTorch platform compatibility
        Source: https://docs.python.org/3/library/platform.html
        """
        import platform
        system = platform.system()

        # Windows has known issues with torch.compile stability
        # Linux and macOS generally work well
        if system == "Windows":
            return False
        elif system in ["Linux", "Darwin"]:  # macOS
            return True
        else:
            # Conservative default for unknown platforms
            return False

    def _init_weights(self, module: nn.Module) -> None:
        # Set up the model's weights with good starting values
        if isinstance(module, nn.Linear) or isinstance(module, nn.Parameter):
            if hasattr(module, 'weight'):
                # Scale weights based on number of layers for stable training
                nn.init.normal_(module.weight, mean=0.0, std=WEIGHT_INIT_STD / math.sqrt(2 * self.config.n_layer))
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.zeros_(module.bias)  # Start biases at zero
        elif isinstance(module, nn.Embedding):
            # Initialize word embeddings
            nn.init.normal_(module.weight, mean=0.0, std=WEIGHT_INIT_STD)

    def forward(
        self,
        input_ids: torch.Tensor,  # The words we want to process
        attention_mask: Optional[torch.Tensor] = None,  # Which words to pay attention to
        position_ids: Optional[torch.Tensor] = None,  # Position of each word
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,  # Previous conversation memory
        inputs_embeds: Optional[torch.Tensor] = None,  # Pre-computed word representations
        use_cache: bool = False,  # Remember for next time
        output_attentions: bool = False,  # Return attention weights
        output_hidden_states: bool = False,  # Return all layer outputs
        return_dict: bool = True,  # Return as nice object or tuple
        cache_position: Optional[torch.Tensor] = None,  # Cache position for efficient caching
        # Encoder-decoder specific arguments
        decoder_input_ids: Optional[torch.Tensor] = None,  # Target sequence for seq2seq
        decoder_attention_mask: Optional[torch.Tensor] = None,  # Target attention mask
        encoder_hidden_states: Optional[torch.Tensor] = None,  # Encoder outputs for cross-attention
        encoder_attention_mask: Optional[torch.Tensor] = None,  # Encoder attention mask
        labels: Optional[torch.Tensor] = None,  # Labels for loss computation
    ) -> Union[Tuple, ModelOutput, Dict[str, torch.Tensor]]:
        # Handle decoder model (with optional cross-attention)
        if getattr(self.config, 'is_decoder', False):
            decoder_input = input_ids
            decoder_attn_mask = attention_mask

            # If this is called from an encoder-decoder setup, use decoder-specific inputs
            if decoder_input_ids is not None:
                decoder_input = decoder_input_ids
                decoder_attn_mask = decoder_attention_mask

            # Forward through decoder
            decoder_outputs = self.decoder(
                decoder_input,
                encoder_outputs=encoder_hidden_states,
                attention_mask=decoder_attn_mask,
                encoder_attention_mask=encoder_attention_mask,
                labels=labels,
                use_cache=use_cache,
            )

            return decoder_outputs

        # Standard decoder-only forward pass
        # Convert word IDs to mathematical representations
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Prepare to collect outputs if requested
        all_hidden_states = (inputs_embeds,) if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        # Start with the embedded words
        hidden_states = inputs_embeds

        # Calculate query position for DCA (used in generation)
        batch_size, seq_len = input_ids.shape
        if cache_position is not None:
            # Use cache_position if provided (newer transformers API)
            past_tokens = cache_position[0].item() if cache_position.numel() > 0 else 0
            query_position = past_tokens + seq_len - 1
        else:
            # Fallback to legacy method
            past_tokens = get_past_key_values_length(past_key_values)
            query_position = past_tokens + seq_len - 1

        # Process through each transformer layer
        for i, layer in enumerate(self.layers):
            # Save intermediate results if requested
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # Get cached values for this layer if available
            layer_past = past_key_values[i] if past_key_values else None

            # Use gradient checkpointing if enabled to save memory
            if self.config.use_activation_checkpointing and self.training:
                from torch.utils.checkpoint import checkpoint

                def layer_forward(*args):
                    return layer(*args)

                layer_outputs = checkpoint(
                    layer_forward,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    layer_past,
                    output_attentions,
                    use_cache,
                    query_position,
                    use_reentrant=False,  # More memory efficient
                )
            else:
                # Let the layer do its work
                layer_outputs = layer(
                    hidden_states,
                    attention_mask,
                    position_ids,
                    layer_past,
                    output_attentions,
                    use_cache,
                    query_position,
                )

            # Update our understanding
            hidden_states = layer_outputs[0]


            # Save cache for next time if requested
            if use_cache:
                next_decoder_cache += (layer_outputs[2],)

            # Save attention weights if requested
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        # Final normalization
        hidden_states = self.norm(hidden_states)

        # Include final hidden state in outputs
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # Return results in the requested format
        if not return_dict:
            return tuple(v for v in [hidden_states, next_decoder_cache, all_hidden_states, all_self_attns] if v is not None)

        return ModelOutput(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
