# Nano GPT Model
# ===============
#
# Transformer-based language model implementation with:
# - RMSNorm: Root Mean Square Layer Normalization
# - RoPE: Rotary Position Embedding
# - SwiGLU: Swish-Gated Linear Unit activation
# - GQA: Grouped Query Attention (optional)
# - Top-p sampling: Nucleus sampling for text generation

import math
import logging
import contextlib
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Union, Any

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .attention_utils import build_attention_mask, repeat_kv

# Constants
DEFAULT_VOCAB_SIZE = 50257  # Default GPT-2 vocab size (auto-detected from tokenizer in practice)
DEFAULT_BLOCK_SIZE = 1024
DEFAULT_N_LAYER = 12  # Default depth is 12 layers for ~100M params (with LCR/GTR optional).
DEFAULT_N_HEAD = 12
DEFAULT_N_EMBD = 768
DEFAULT_DROPOUT = 0.0
DEFAULT_BIAS = True

WEIGHT_INIT_STD = 0.02

logger = logging.getLogger(__name__)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Reference: https://arxiv.org/abs/1910.07467
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        """Initialize RMS normalization.

        Args:
            dim: Dimensionality of the input tensor
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization to input tensor.

        Args:
            x: Input tensor of shape (..., dim)

        Returns:
            Normalized tensor with same shape as input
        """
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x * self.weight / rms


NormalizationLayer = RMSNorm


class RotaryPositionEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE).

    Encodes position information using rotation matrices.

    Reference: https://arxiv.org/abs/2104.09864
    """

    def __init__(self, dim: int, max_seq_len: int, base: int = 10000) -> None:
        """Initialize RoPE.

        Args:
            dim: Dimensionality of the embeddings
            max_seq_len: Maximum sequence length to precompute (required, no default)
            base: Base for the frequency computation
        """
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.initial_max_seq_len = max_seq_len
        self.base = base

        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self._build_rope_cache()

    def _build_rope_cache(self, *, device: Optional[torch.device] = None) -> None:
        """Precompute rotation matrices for all positions and dimensions."""
        device = device or self.inv_freq.device
        inv_freq = self.inv_freq.to(device)
        self.inv_freq = inv_freq
        t = torch.arange(self.max_seq_len, device=device, dtype=inv_freq.dtype)
        freqs = torch.einsum('i,j->ij', t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        self.register_buffer('cos_cache', cos, persistent=False)
        self.register_buffer('sin_cache', sin, persistent=False)

    def _maybe_extend_rope_cache(self, target_seq_len: int, device: torch.device) -> None:
        if target_seq_len <= self.max_seq_len:
            if device is not None:
                if self.cos_cache.device != device:
                    self.cos_cache = self.cos_cache.to(device)
                    self.sin_cache = self.sin_cache.to(device)
                if self.inv_freq.device != device:
                    self.inv_freq = self.inv_freq.to(device)
            return
        logger.info(
            "Extending RoPE cache from %d to %d positions", self.max_seq_len, target_seq_len
        )
        old_max = self.max_seq_len
        self.max_seq_len = target_seq_len
        inv_freq = self.inv_freq.to(device)
        self.inv_freq = inv_freq
        t = torch.arange(old_max, target_seq_len, device=device, dtype=inv_freq.dtype)
        freqs = torch.einsum('i,j->ij', t, inv_freq)
        cos_append = freqs.cos()
        sin_append = freqs.sin()
        if self.cos_cache.device != device:
            self.cos_cache = self.cos_cache.to(device)
            self.sin_cache = self.sin_cache.to(device)
        self.cos_cache = torch.cat([self.cos_cache, cos_append], dim=0)
        self.sin_cache = torch.cat([self.sin_cache, sin_append], dim=0)

    def forward(self, x: torch.Tensor, seq_start: int = 0) -> torch.Tensor:
        """Apply rotary position embedding to input tensor.

        Args:
            x: Input tensor of shape (batch, n_heads, seq_len, head_dim)
            seq_start: Starting position offset (for incremental decoding)

        Returns:
            Tensor with rotary position embedding applied (same shape as input)
        """
        # Handle shape mismatch: add dimension if needed for single-head testing
        original_shape = x.shape
        if x.dim() == 3:
            x = x.unsqueeze(1)  # Add head dimension: (B, T, D) -> (B, 1, T, D)

        # Fix: RoPE receives (B, H, T, D) after transpose in attention, not (B, T, H, D)
        assert x.dim() == 4, f"Expected 4D tensor (B, H, T, D), got {x.shape}"
        batch_size, n_heads, seq_len, head_dim = x.shape
        logger.debug(f"RoPE input shape: (B={batch_size}, H={n_heads}, T={seq_len}, D={head_dim}), seq_start={seq_start}")

        positions = torch.arange(seq_start, seq_start + seq_len, device=x.device, dtype=torch.long)

        max_pos = seq_start + seq_len
        if max_pos > self.max_seq_len:
            if max_pos > self.initial_max_seq_len:
                logger.info(
                    "RoPE extrapolating beyond initial cache (%d -> %d)",
                    self.initial_max_seq_len,
                    max_pos,
                )
            self._maybe_extend_rope_cache(max_pos, x.device)
            positions = torch.arange(seq_start, seq_start + seq_len, device=x.device, dtype=torch.long)

        cos = self.cos_cache[positions]
        sin = self.sin_cache[positions]

        # Reshape for broadcasting: (seq_len, head_dim // 2) -> (1, 1, seq_len, head_dim // 2)
        # Simplified approach while maintaining correct shape
        cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim//2)
        sin = sin.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim//2)

        # x shape: (B, H, T, D)
        x_even = x[..., ::2]  # (B, H, T, D//2)
        x_odd = x[..., 1::2]  # (B, H, T, D//2)

        # DeepSeek-V3 style: Use complex multiplication for RoPE
        # Create complex tensors: x_complex = x_even + 1j * x_odd
        x_complex = torch.complex(x_even, x_odd)  # (B, H, T, D//2)

        # Create rotation complex: rot = cos + 1j * sin
        rot_complex = torch.complex(cos, sin)  # (1, 1, T, D//2)

        # Apply rotation: result = x_complex * rot_complex
        result_complex = x_complex * rot_complex

        # Extract real and imaginary parts
        output = torch.empty_like(x)
        output[..., ::2] = result_complex.real
        output[..., 1::2] = result_complex.imag

        if not torch.isfinite(output).all():
            logger.warning("Detected non-finite values in RoPE output; clamping to [-1e4, 1e4].")
            output = torch.clamp(output, min=-1e4, max=1e4)

        # Restore original shape if we added a dimension
        if len(original_shape) == 3:
            output = output.squeeze(1)  # Remove head dimension: (B, 1, T, D) -> (B, T, D)

        return output


def dynamic_rope_update(func):
    """Decorator that auto-extends RoPE caches when sequences exceed current limits."""

    def wrapper(self, x: torch.Tensor, *args, **kwargs):
        start = kwargs.get("start", 0)
        seq_len = x.shape[-2]
        end_pos = start + seq_len
        current_limit = getattr(self, "target_ctx", getattr(self, "max_seq_len", end_pos))
        if end_pos > current_limit and hasattr(self, "resize_cache"):
            self.resize_cache(end_pos)
        return func(self, x, *args, **kwargs)

    return wrapper


class RopeWithYaRN:
    """
    Rotary Position Embedding with YaRN scaling for extended context windows.

    YaRN (Yet Another RoPE) extends RoPE's context window by scaling frequencies,
    enabling transformers to handle sequences much longer than their training length.

    How it works:
    - Standard RoPE: frequencies decrease exponentially with position
    - YaRN: stretches frequencies by (target_ctx/orig_ctx)^alpha, then optionally
      adjusts magnitude by beta factor

    Key features:
    - Backward compatible: falls back to standard RoPE when disabled
    - Efficient: caches frequency computations per sequence start position
    - Configurable: separate original/target contexts, alpha/beta parameters

    Usage:
        rope = RopeWithYaRN(
            dim=head_dim,
            base=10000,
            orig_ctx=2048,    # Training context length
            target_ctx=8192,  # Extended context for generation
            alpha=1.0,        # Frequency scaling exponent
            beta=1.0          # Magnitude adjustment (keep 1.0)
        )

        # Apply to Q and K before attention
        q_rotated = rope.apply_rope(q, start_pos)
        k_rotated = rope.apply_rope(k, start_pos)

    References:
    - RoPE: https://arxiv.org/abs/2104.09864
    - YaRN: https://arxiv.org/abs/2309.00071
    """

    def __init__(self, dim, base=10000.0, orig_ctx=2048, target_ctx=8192,
                 alpha=1.0, beta=1.0, enabled=True, device=None, dtype=None):
        self.dim = dim
        self.base = base
        self.orig_ctx = orig_ctx
        self.target_ctx = target_ctx
        self.alpha = alpha
        self.beta = beta
        self.enabled = enabled
        self.device = device
        self.dtype = dtype

        # If not enabled, fall back to standard RoPE
        if not self.enabled:
            # Create a standard RotaryPositionEmbedding instead
            self.rope_fallback = RotaryPositionEmbedding(self.dim, self.orig_ctx, self.base)
        else:
            self.rope_fallback = None

    def _frequencies(
        self,
        seq_len: int,
        start: int = 0,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        # Resolve runtime device/dtype without mutating shared state unnecessarily
        device = device or self.device or torch.device("cpu")
        target_dtype = dtype or self.dtype or torch.float32

        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2, device=device, dtype=torch.float32) / self.dim)
        )

        # YaRN scaling (stretch)
        if self.enabled and self.target_ctx > self.orig_ctx:
            scale = (self.target_ctx / self.orig_ctx) ** self.alpha
            inv_freq = inv_freq / scale

            # optional magnitude tweak
            inv_freq = inv_freq * (1.0 / max(self.beta, 1e-6))

        # Defensive check for sequence length exceeding target context
        if self.enabled and start + seq_len > self.target_ctx:
            logger.warning(f"Sequence length {start + seq_len} exceeds YaRN target context {self.target_ctx}. "
                          f"Consider increasing yarn_target_ctx for better performance.")

        positions = torch.arange(start, start + seq_len, device=device, dtype=torch.float32)
        freqs = torch.einsum('i,j->ij', positions, inv_freq)  # [T, dim/2]

        return freqs.to(target_dtype)

    def resize_cache(self, max_seq_len: int) -> None:
        """Dynamically resize RoPE cache for longer sequences."""
        if hasattr(self, 'rope_fallback') and self.rope_fallback is not None:
            # Resize fallback RoPE
            self.rope_fallback = RotaryPositionEmbedding(
                self.rope_fallback.dim, max_seq_len, self.rope_fallback.base
            )
        else:
            # For YaRN, dynamically extend the target context
            if max_seq_len > self.target_ctx:
                old_target = self.target_ctx
                self.target_ctx = max_seq_len
                logger.info(f"YaRN: Extended target context from {old_target} to {max_seq_len}")
                # Note: Frequency calculations will adapt on-the-fly in _frequencies()
            else:
                logger.debug(f"YaRN: Requested resize to {max_seq_len} but current target_ctx is {self.target_ctx}")

    @staticmethod
    def apply(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        """
        x: [B, H, T, Dh], freqs: [T, Dh/2]
        Apply RoPE rotation to the last dimension of x.
        """
        d = x.shape[-1]  # Dh (head dimension)
        # x shape: [B, H, T, Dh]
        # freqs shape: [T, Dh/2]

        # Split x into even and odd dimensions
        x_even = x[..., ::2]  # [B, H, T, Dh/2]
        x_odd = x[..., 1::2]  # [B, H, T, Dh/2]

        # Get sin and cos with proper broadcasting
        # freqs: [T, Dh/2] -> sin/cos: [1, 1, T, Dh/2]
        sin = freqs.sin().unsqueeze(0).unsqueeze(0)  # [1, 1, T, Dh/2]
        cos = freqs.cos().unsqueeze(0).unsqueeze(0)  # [1, 1, T, Dh/2]

        # Apply rotation
        y_even = x_even * cos - x_odd * sin
        y_odd = x_odd * cos + x_even * sin

        # Interleave back
        y = torch.empty_like(x)
        y[..., ::2] = y_even
        y[..., 1::2] = y_odd

        # Numerical stability: Check for NaN/Inf in RoPE output
        if torch.isnan(y).any() or torch.isinf(y).any():
            y = torch.clamp(y, min=-1e4, max=1e4)

        return y

    @dynamic_rope_update
    def apply_rope(self, x: torch.Tensor, start: int = 0) -> torch.Tensor:
        """Apply RoPE rotation, using fallback if YaRN is not enabled."""
        if not self.enabled and self.rope_fallback is not None:
            return self.rope_fallback(x, seq_start=start)
        else:
            # Use YaRN RoPE
            seq_len = x.shape[-2]  # seq_len is second-to-last dim
            freqs = self._frequencies(seq_len, start)
            return RopeWithYaRN.apply(x, freqs)


def _build_default_rope(config: "ModelSettings") -> RopeWithYaRN:
    """Builder for standard rotary embeddings."""
    head_dim = config.n_embd // config.n_head
    return RopeWithYaRN(
        dim=head_dim,
        base=config.rope_base,
        orig_ctx=config.block_size,
        target_ctx=config.block_size,
        alpha=1.0,
        beta=1.0,
        enabled=False,
    )


def _build_yarn_rope(config: "ModelSettings") -> RopeWithYaRN:
    """Builder for YaRN-scaled rotary embeddings."""
    head_dim = config.n_embd // config.n_head
    return RopeWithYaRN(
        dim=head_dim,
        base=config.rope_base,
        orig_ctx=config.yarn_orig_ctx,
        target_ctx=config.yarn_target_ctx,
        alpha=config.yarn_alpha,
        beta=config.yarn_beta,
        enabled=True,
    )


ROPE_BUILDERS = {
    "default": _build_default_rope,
    "yarn": _build_yarn_rope,
}


def build_rope(config: "ModelSettings") -> RopeWithYaRN:
    """Return the rope module requested by configuration."""
    rope_key = (config.rope_type or "auto").lower()
    if rope_key == "auto":
        rope_key = "yarn" if config.use_yarn else "default"

    builder = ROPE_BUILDERS.get(rope_key)
    if builder is None:
        raise ValueError(f"Unknown rope_type '{config.rope_type}'. Available: {', '.join(ROPE_BUILDERS)}")
    return builder(config)


class LCRBlock(nn.Module):
    """
    Local Convolutional Reasoning:

    RMSNorm -> depthwise conv (captures n-gram locality) -> SwiGLU -> proj -> residual
    """

    def __init__(self, d_model: int, kernel_size: int = 7, expand: int = 2, dropout: float = 0.0, bias: bool = True):
        super().__init__()

        self.norm = RMSNorm(d_model)

        self.dw = nn.Conv1d(d_model, d_model, kernel_size, groups=d_model, padding=kernel_size // 2)

        self.up = nn.Linear(d_model, expand * d_model * 2, bias=bias)  # *2 for SwiGLU split

        self.down = nn.Linear(expand * d_model, d_model, bias=bias)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]

        h = self.norm(x)

        h = self.dw(h.transpose(1, 2)).transpose(1, 2)  # depthwise conv along T

        u, v = self.up(h).chunk(2, dim=-1)

        h = F.silu(u) * v                                # SwiGLU

        h = self.down(h)

        return x + self.dropout(h)


class GTRBlock(nn.Module):
    """
    Global Token Reasoning (single-head, full-width attention):

      1) Global tokens attend to sequence (gathers a summary)

      2) Sequence attends to globals (redistributes summary)

    """

    def __init__(self, d_model: int, num_tokens: int = 8, dropout: float = 0.0, bias: bool = True):
        super().__init__()

        self.num_tokens = num_tokens
        self.global_tokens = nn.Parameter(torch.zeros(1, num_tokens, d_model))
        nn.init.normal_(self.global_tokens, std=0.02)

        # Norms
        self.norm_seq1 = RMSNorm(d_model)
        self.norm_g1   = RMSNorm(d_model)
        self.norm_gmlp = RMSNorm(d_model)
        self.norm_seq2 = RMSNorm(d_model)
        self.norm_g2   = RMSNorm(d_model)

        # Cross attentions (simple cross-attention without projections)
        self.q_proj1 = nn.Linear(d_model, d_model, bias=bias)  # For global -> seq
        self.k_proj1 = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj1 = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj1 = nn.Linear(d_model, d_model, bias=bias)

        self.q_proj2 = nn.Linear(d_model, d_model, bias=bias)  # For seq -> global
        self.k_proj2 = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj2 = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj2 = nn.Linear(d_model, d_model, bias=bias)

        # Tiny MLP for globals (SwiGLU: gate * SiLU(up) -> down)
        self.g_gate = nn.Linear(d_model, 4 * d_model, bias=bias)
        self.g_up   = nn.Linear(d_model, 4 * d_model, bias=bias)
        self.g_down = nn.Linear(4 * d_model, d_model, bias=bias)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, C = x.shape
        g = self.global_tokens.expand(B, -1, -1)  # [B, G, C]

        # 1) Global tokens <- Sequence (Q=g, K/V=seq)
        s1 = self.norm_seq1(x)  # [B, T, C]
        g1 = self.norm_g1(g)    # [B, G, C]

        # Project to queries, keys, values
        q1 = self.q_proj1(g1)  # [B, G, C]
        k1 = self.k_proj1(s1)  # [B, T, C]
        v1 = self.v_proj1(s1)  # [B, T, C]

        # Simple cross-attention: Q @ K^T @ V
        # q1: [B, G, C], k1: [B, T, C] -> attn: [B, G, T]
        attn1 = (q1 @ k1.transpose(-2, -1)) / (C ** 0.5)
        if attn_mask is not None:
            # attn_mask is (B, T) where 1 = real token, 0 = padding
            # Convert to additive mask: 0 for keep, -inf for mask
            attn_mask_1 = (1.0 - attn_mask.float()) * float('-inf')
            attn_mask_1 = attn_mask_1.unsqueeze(1)  # (B, 1, T)
            attn1 = attn1 + attn_mask_1
        attn1 = F.softmax(attn1, dim=-1)
        g_upd = attn1 @ v1  # [B, G, C]
        g_upd = self.out_proj1(g_upd)
        g = g + self.dropout(g_upd)

        # MLP on globals (SwiGLU)
        h = self.norm_gmlp(g)
        u = self.g_up(h)
        v = self.g_gate(h)
        h = F.silu(u) * v
        h = self.g_down(h)
        g = g + self.dropout(h)

        # 2) Sequence <- Globals (Q=seq, K/V=g)
        s2 = self.norm_seq2(x)  # [B, T, C]
        g2 = self.norm_g2(g)    # [B, G, C]

        # Project to queries, keys, values
        q2 = self.q_proj2(s2)  # [B, T, C]
        k2 = self.k_proj2(g2)  # [B, G, C]
        v2 = self.v_proj2(g2)  # [B, G, C]

        # Simple cross-attention: Q @ K^T @ V
        # q2: [B, T, C], k2: [B, G, C] -> attn: [B, T, G]
        attn2 = (q2 @ k2.transpose(-2, -1)) / (C ** 0.5)
        if attn_mask is not None:
            # attn_mask is (B, T) where 1 = real token, 0 = padding
            # For sequence -> global, mask out queries from padded positions
            attn_mask_2 = (1.0 - attn_mask.float()) * float('-inf')
            attn_mask_2 = attn_mask_2.unsqueeze(-1)  # (B, T, 1)
            attn2 = attn2 + attn_mask_2
        attn2 = F.softmax(attn2, dim=-1)
        x_upd = attn2 @ v2  # [B, T, C]
        x_upd = self.out_proj2(x_upd)
        x = x + self.dropout(x_upd)

        return x


class FocusedAttentionMechanism(nn.Module):
    """Multi-head self-attention mechanism with causal masking.

    Supports Grouped Query Attention (GQA) and sliding window attention.
    """

    def __init__(self, config: "ModelSettings") -> None:
        """Initialize the attention mechanism.

        Args:
            config: Model configuration containing attention parameters

        Raises:
            ValueError: If embedding dimension is not divisible by number of heads or GQA config is invalid
        """
        super().__init__()

        self.n_head = config.n_head
        self.n_kv_groups = config.n_kv_groups if config.n_kv_groups is not None else config.n_head
        self.n_kv_heads = self.n_kv_groups

        if config.n_embd % self.n_head != 0:
            raise ValueError(
                f"Embedding dimension ({config.n_embd}) must be divisible by "
                f"number of query heads ({self.n_head})"
            )

        if self.n_head % self.n_kv_groups != 0:
            raise ValueError(
                f"Number of query heads ({self.n_head}) must be divisible by "
                f"number of KV groups ({self.n_kv_groups})"
            )

        self.head_size = config.n_embd // self.n_head
        self.n_embd = config.n_embd
        self.use_fused_attention = config.use_fused_attention
        self.attn_logit_softcapping = config.attn_logit_softcapping
        self.use_fp32_softmax = config.use_fp32_softmax

        # Gemma-2 style: Fused QKV projection (faster, better memory access)
        if self.use_fused_attention:
            # Single matrix multiplication: Q + K + V
            total_head_dim = (self.n_head + 2 * self.n_kv_heads) * self.head_size
            self.qkv_projection = nn.Linear(config.n_embd, total_head_dim, bias=config.bias)
            logger.info("Using fused QKV attention projection (Gemma-2 style)")
            
            # Register hook to convert old state dict format (separate Q/K/V) to fused format
            def load_state_dict_hook(state_dict, prefix, *args):
                """Convert separate Q/K/V projections to fused QKV projection."""
                # Handle both pre-hook signatures (PyTorch version differences)
                if isinstance(prefix, dict):
                    # If prefix is actually state_dict (older PyTorch), adjust
                    state_dict = prefix
                    prefix = args[0] if args else ""
                
                q_weight_key = prefix + "query_projection.weight"
                k_weight_key = prefix + "key_projection.weight"
                v_weight_key = prefix + "value_projection.weight"
                
                if q_weight_key in state_dict:
                    # Convert separate to fused
                    q_weight = state_dict.pop(q_weight_key)
                    k_weight = state_dict.pop(k_weight_key)
                    v_weight = state_dict.pop(v_weight_key)
                    state_dict[prefix + "qkv_projection.weight"] = torch.cat([q_weight, k_weight, v_weight], dim=0)
                    
                    if config.bias:
                        q_bias = state_dict.pop(prefix + "query_projection.bias")
                        k_bias = state_dict.pop(prefix + "key_projection.bias")
                        v_bias = state_dict.pop(prefix + "value_projection.bias")
                        state_dict[prefix + "qkv_projection.bias"] = torch.cat([q_bias, k_bias, v_bias], dim=0)
            
            self.register_load_state_dict_pre_hook(load_state_dict_hook)
        else:
            # Separate projections (backward compatibility)
            self.query_projection = nn.Linear(config.n_embd, self.n_head * self.head_size, bias=config.bias)
            self.key_projection = nn.Linear(config.n_embd, self.n_kv_heads * self.head_size, bias=config.bias)
            self.value_projection = nn.Linear(config.n_embd, self.n_kv_heads * self.head_size, bias=config.bias)
        
        self.output_projection = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.attention_dropout = nn.Dropout(config.dropout)
        self.residual_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout

        # Keep a hard cap for KV cache length to bound memory/compute during generation
        self.max_cache_len = config.max_cache_len if config.max_cache_len > 0 else config.block_size

        context_limit = config.yarn_target_ctx if config.use_yarn else config.block_size
        context_limit = max(context_limit, config.max_cache_len if config.max_cache_len > 0 else config.block_size)

        # Determine SDPA/Flash path availability and log accurate reason when disabled
        sdpa_available = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        self.flash_available = sdpa_available
        if config.attn_logit_softcapping is not None:
            # Softcapping is incompatible with SDPA path in our implementation
            self.flash_available = False
            logger.info("Manual attention selected: attn_logit_softcapping is enabled.")
        if not self.flash_available:
            if not sdpa_available:
                logger.info("Manual attention selected: scaled_dot_product_attention not available in this PyTorch.")
            else:
                logger.info("Manual attention selected: feature settings require non-SDPA path.")
            self.register_buffer(
                "causal_mask",
                torch.tril(torch.ones(context_limit, context_limit))
                .view(1, 1, context_limit, context_limit)
            )

        self.rope = RotaryPositionEmbedding(self.head_size, context_limit, base=config.rope_base)


    def _concat_past_kv(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        cache_len = 0
        if past_key_value is not None:
            past_key, past_value = past_key_value
            cache_len = past_key.shape[2]
            k = torch.cat([past_key, k], dim=2)
            v = torch.cat([past_value, v], dim=2)
        return k, v, cache_len

    def _apply_rope_to_qk(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        *,
        rope: RopeWithYaRN | None,
        freqs: torch.Tensor | None,
        apply_rope: bool,
        position_ids: Optional[torch.Tensor],
        cache_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not apply_rope or rope is None:
            return q, k

        if freqs is not None:
            q = RopeWithYaRN.apply(q, freqs)
            k_rot = RopeWithYaRN.apply(k, freqs)
        else:
            if position_ids is not None and position_ids.numel() > 0:
                seq_start = position_ids[0, 0].item()
            else:
                seq_start = cache_len
            q = self.rope(q, seq_start=seq_start)
            k_rot = self.rope(k, seq_start=seq_start)

        return q, k_rot

    def forward(
        self,
        x: torch.Tensor | None = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        *,
        query: torch.Tensor | None = None,
        key: torch.Tensor | None = None,
        value: torch.Tensor | None = None,
        rope: RopeWithYaRN | None = None,
        freqs: torch.Tensor | None = None,
        apply_rope: bool = True,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[torch.Tensor]]:
        """Apply multi-head attention to input tensor with optional GQA and KV caching.

        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim) for self-attention
            attention_mask: Optional attention mask of shape (batch_size, seq_len) where 1 = real token, 0 = padding.
                           This mask is converted internally to additive mask format (0 for keep, -inf for mask)
                           and broadcast to (B, 1, 1, T) for attention computation.
            position_ids: Optional position IDs tensor for KV caching
            past_key_value: Optional cached key and value states from previous forward pass
            use_cache: Whether to return cached key/value states
            query: Optional explicit query tensor for cross-attention
            key: Optional explicit key tensor for cross-attention
            value: Optional explicit value tensor for cross-attention
            rope: Optional RopeWithYaRN instance for position encoding
            freqs: Optional precomputed frequencies for YaRN RoPE
            apply_rope: Whether to apply RoPE to Q and K (skip for global tokens)

        Returns:
            Tuple of (output, past_key_value) where:
            - output: Output tensor of shape (batch_size, seq_len, embed_dim)
            - past_key_value: Cached key/value states if use_cache=True, else None
        """
        # Handle explicit Q/K/V inputs for cross-attention
        if query is not None and key is not None and value is not None:
            # Cross-attention mode: use provided tensors directly
            q, k, v = query, key, value
            B, n_heads_q, T_q, head_dim = q.shape
            _, n_heads_kv, T_kv, _ = k.shape
            original_seq_len = T_q

            # For cross-attention, we assume no cache support for now
            cache_len = 0
            use_cache = False
            past_key_value = None

            logger.debug(f"Cross-attention input shapes: Q(B={B}, H_q={n_heads_q}, T_q={T_q}, D={head_dim}), "
                        f"K(B={B}, H_kv={n_heads_kv}, T_kv={T_kv}, D={head_dim})")
        elif x is not None:
            # Self-attention mode: project from input
            if x.dim() != 3:
                raise ValueError(f"Input must be 3D tensor, got shape {x.shape}")

            B, T, C = x.size()
            original_seq_len = T  # Store original sequence length before cache concatenation
            logger.debug(f"Self-attention input shape: (B={B}, T={T}, C={C})")

            # Gemma-2 style: Fused QKV projection (single matrix multiplication)
            if self.use_fused_attention:
                qkv = self.qkv_projection(x)  # (B, T, total_head_dim)

                # Split into Q, K, V
                query_size = self.n_head * self.head_size
                kv_size = self.n_kv_heads * self.head_size
                q, k, v = qkv.split([query_size, kv_size, kv_size], dim=-1)
            else:
                # Separate projections (backward compatibility)
                q = self.query_projection(x)
                k = self.key_projection(x)
                v = self.value_projection(x)
        else:
            raise ValueError("Either 'x' for self-attention or 'query', 'key', 'value' for cross-attention must be provided")

        # Reshape and transpose (only for self-attention projections)
        if x is not None:  # Self-attention: reshape from (B, T, C) projections
            q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
            k = k.view(B, T, self.n_kv_heads, self.head_size).transpose(1, 2)
            v = v.view(B, T, self.n_kv_heads, self.head_size).transpose(1, 2)
            T_q, T_kv = T, T
        else:  # Cross-attention: tensors already in (B, H, T, D) format
            T_q, T_kv = T_q, T_kv

        cache_len = past_key_value[0].shape[2] if past_key_value is not None else 0

        q, k = self._apply_rope_to_qk(
            q,
            k,
            rope=rope,
            freqs=freqs,
            apply_rope=apply_rope,
            position_ids=position_ids,
            cache_len=cache_len,
        )

        new_k_tokens = k
        new_v_tokens = v

        concat_source = past_key_value if (past_key_value is not None and x is not None) else None
        k, v, concatenated_cache_len = self._concat_past_kv(k, v, concat_source)
        if concat_source is not None:
            cache_len = concatenated_cache_len

        # Enforce KV cache length limit by trimming oldest tokens (sliding window)
        T_kv = k.shape[2]
        if T_kv > self.max_cache_len:
            k = k[:, :, -self.max_cache_len :, :]
            v = v[:, :, -self.max_cache_len :, :]
            # Adjust cache_len to reflect trimmed historical tokens
            cache_len = max(0, min(cache_len, self.max_cache_len - T_q))
            # Update T_kv to reflect trimming
            T_kv = k.shape[2]

        is_self_attention = x is not None
        target_heads = self.n_head if is_self_attention else q.shape[1]
        if k.shape[1] != target_heads:
            if target_heads % k.shape[1] != 0:
                raise ValueError(
                    f"Cannot align KV heads: target {target_heads}, current {k.shape[1]}"
                )
            repeats = target_heads // k.shape[1]
            k = repeat_kv(k, repeats)
            v = repeat_kv(v, repeats)

        if q.shape[1] != k.shape[1] or v.shape[1] != k.shape[1]:
            raise ValueError(
                f"Head alignment failure: q={q.shape[1]}, k={k.shape[1]}, v={v.shape[1]}"
            )

        cache_tokens_for_mask = 0
        if is_self_attention and (past_key_value is not None or concat_source is not None):
            # After trimming, cached tokens are those preceding current queries
            cache_tokens_for_mask = max(T_kv - original_seq_len, 0)

        if is_self_attention:
            attn_mask_padding = build_attention_mask(
                batch_size=q.shape[0],
                total_kv_len=T_kv,
                attention_mask=attention_mask,
                cache_len=cache_tokens_for_mask,
                device=q.device,
            )
        else:
            attn_mask_padding = attention_mask

        attn_weights: Optional[torch.Tensor] = None

        if self.flash_available and q.device.type == 'cuda' and not output_attentions:
            # Use Flash Attention for 20-30% speedup on RTX 30 series
            # Convert attention_mask to additive mask for flash attention
            # Flash attention expects additive mask: 0 for keep, -inf for mask
            flash_attn_mask = attn_mask_padding

            # Use causal masking only for self-attention; additive masks handle padding
            is_causal = x is not None

            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=flash_attn_mask,
                dropout_p=self.dropout if self.training else 0,
                is_causal=is_causal
            )
        else:
            head_dim = q.shape[-1]
            if head_dim == 0:
                raise ValueError("Head dimension cannot be zero")
            scale_factor = math.sqrt(head_dim)

            att = (q @ k.transpose(-2, -1)) / scale_factor

            if att.shape[0] != v.shape[0] or att.shape[1] != v.shape[1]:
                raise ValueError(
                    f"Batch/head mismatch: att {att.shape[:2]} vs value {v.shape[:2]}"
                )
            if att.shape[-1] != v.shape[-2]:
                raise ValueError(
                    f"Matmul mismatch: att last dim {att.shape[-1]} vs value seq {v.shape[-2]}"
                )

            # Gemma-2 feature: Attention logit soft-capping (prevents overflow)
            if self.attn_logit_softcapping is not None:
                att = att / self.attn_logit_softcapping
                att = torch.tanh(att)
                att = att * self.attn_logit_softcapping

            # Apply padding mask
            if attn_mask_padding is not None:
                att = att + attn_mask_padding

            # Apply causal mask for self-attention
            if x is not None:
                if hasattr(self, "causal_mask"):
                    att = att.masked_fill(
                        self.causal_mask[:, :, :T_q, :T_kv] == 0,
                        float('-inf')
                    )

            # Numerical stability: Check for NaN/Inf in attention logits
            if torch.isnan(att).any() or torch.isinf(att).any():
                att = torch.clamp(att, min=-1e4, max=1e4)

            # Gemma-2 feature: Upcast softmax to FP32 for numerical stability
            if self.use_fp32_softmax:
                att = F.softmax(att, dim=-1, dtype=torch.float32).to(q.dtype)
            else:
                att = F.softmax(att, dim=-1)
            
            att = self.attention_dropout(att)
            if output_attentions:
                attn_weights = att.detach()
            y = att @ v

        y = y.transpose(1, 2).contiguous()

        # If using cache, we only care about outputs for new tokens (self-attention only)
        if past_key_value is not None and x is not None:
            # y shape is (B, T_q, H, D) where T_q may include cached context
            # We only want the new tokens (last original_seq_len tokens)
            y = y[:, -original_seq_len:, :, :]  # Only new tokens

        if x is not None:  # Self-attention: apply output projection
            y = y.contiguous().view(B, y.shape[1], C)
            y = self.residual_dropout(self.output_projection(y))
        else:  # Cross-attention: output is already in the right shape
            # y is (B, T_q, H, D) -> (B, T_q, H*D) where H*D should equal C
            y = y.contiguous().view(B, y.shape[1], -1)
        
        # Return cached key/value states if requested (self-attention only)
        past_key_value_out = None
        if use_cache and x is not None:
            # Return cumulative (trimmed) K/V for efficient caching across steps
            # in shape (B, n_kv_heads, seq_len, head_dim)
            past_key_value_out = (k.detach(), v.detach())
        
        if output_attentions and attn_weights is not None:
            elems = attn_weights.shape[-1] * attn_weights.shape[-2]
            if elems > 1_000_000:
                logger.warning(
                    "Collecting attention tensors for shape %dx%d may consume significant memory.",
                    attn_weights.shape[-2],
                    attn_weights.shape[-1],
                )

        return y, past_key_value_out, attn_weights


class FeedForwardNetwork(nn.Module):
    """Feed-forward network with SwiGLU activation."""

    def __init__(self, config: "ModelSettings") -> None:
        """Initialize the feed-forward network with SwiGLU activation.

        Args:
            config: Model configuration containing embedding and dropout parameters
        """
        super().__init__()
        hidden_dim = int(8 * config.n_embd / 3)

        self.gate_projection = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.up_projection = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.activation = nn.SiLU()
        self.down_projection = nn.Linear(hidden_dim, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SwiGLU feed-forward transformation to input tensor.

        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)

        Returns:
            Output tensor of shape (batch_size, seq_len, embed_dim)
        """
        gate = self.gate_projection(x)
        up = self.up_projection(x)
        x = gate * self.activation(up)
        x = self.down_projection(x)
        x = self.dropout(x)
        return x


class TransformerLayer(nn.Module):
    """A single transformer layer combining attention and feed-forward networks."""

    def __init__(self, config: "ModelSettings") -> None:
        """Initialize the transformer layer.

        Args:
            config: Model configuration containing layer parameters
        """
        super().__init__()
        self.pre_attention_norm = RMSNorm(config.n_embd)
        self.attention = FocusedAttentionMechanism(config)
        self.pre_mlp_norm = RMSNorm(config.n_embd)
        self.feed_forward = FeedForwardNetwork(config)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        rope: RopeWithYaRN | None = None,
        freqs: torch.Tensor | None = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[torch.Tensor]]:
        """Apply transformer layer to input tensor with optional KV caching.

        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            attention_mask: Optional attention mask of shape (batch_size, seq_len) where 1 = real token, 0 = padding.
                           Passed through to attention mechanism for masking padding tokens.
            position_ids: Optional position IDs for KV caching
            past_key_value: Optional cached key/value states
            use_cache: Whether to return cached states
            rope: Optional RopeWithYaRN instance for position encoding
            freqs: Optional precomputed frequencies for YaRN RoPE

        Returns:
            Tuple of (output, past_key_value)
        """
        attn_output, past_key_value, attn_probs = self.attention(
            self.pre_attention_norm(x),
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
            rope=rope,
            freqs=freqs,
            apply_rope=True,  # Always apply RoPE for transformer layers
            output_attentions=output_attentions,
        )
        x = x + attn_output
        x = x + self.feed_forward(self.pre_mlp_norm(x))
        return x, past_key_value, attn_probs


@dataclass
class ModelSettings:
    """Configuration class for GPT model hyperparameters.

    Attributes:
        block_size: Maximum sequence length the model can process
        vocab_size: Number of unique tokens in the vocabulary
        n_layer: Number of transformer layers to stack
        n_head: Number of attention heads per layer (query heads)
        n_embd: Dimensionality of token embeddings
        dropout: Dropout probability for regularization
        bias: Whether to use bias in linear layers
        n_kv_groups: Number of KV groups for GQA (None = standard MHA)
        dtype: Data type for mixed precision ('float16', 'bfloat16', 'float32', or None for auto-detect)
        use_fused_attention: Whether to use fused QKV projection (Gemma-2 style, faster)
        attn_logit_softcapping: Optional soft-capping value for attention logits (Gemma-2 feature, prevents overflow)
        use_fp32_softmax: Whether to upcast softmax to FP32 for numerical stability (recommended)
        # --- New/extended config fields ---
        # YaRN / RoPE scaling
        use_yarn: bool = True
        yarn_orig_ctx: int = 2048
        yarn_target_ctx: int = 8192   # e.g., 8k context; set higher if you want 16k/32k later
        yarn_alpha: float = 1.0       # frequency exponent scaling
        yarn_beta: float = 1.0        # magnitude adjustment; keep 1.0 to start
        rope_base: float = 10000.0    # keep your existing base (or whatever you currently use)
        # Reasoning layers
        use_lcr: bool = True
        use_gtr: bool = True
        lcr_kernel_size: int = 7      # odd >=3
        lcr_expand: int = 2           # channel expand factor for pointwise conv
        gtr_num_tokens: int = 8       # number of global tokens
        # gtr_num_heads deprecated - now uses n_head
    # KV cache management
    max_cache_len: int = 1024     # Maximum KV cache length to prevent unlimited memory growth
    allow_hybrid_cache: bool = False  # Set True to force cache usage even when LCR/GTR blocks exist
    """
    config_version: int = 3  # Increment when config serialization changes
    block_size: int = DEFAULT_BLOCK_SIZE
    vocab_size: int = DEFAULT_VOCAB_SIZE
    n_layer: int = 12            # bump default to 12
    n_head: int = DEFAULT_N_HEAD
    n_embd: int = DEFAULT_N_EMBD
    dropout: float = DEFAULT_DROPOUT
    bias: bool = DEFAULT_BIAS
    n_kv_groups: Optional[int] = None
    dtype: Optional[Union[str, torch.dtype]] = None  # accept str or torch.dtype; normalized in __post_init__
    use_fused_attention: bool = True  # Gemma-2 style fused QKV projection
    attn_logit_softcapping: Optional[float] = None  # Gemma-2 feature: set None to prefer SDPA; set >0 to enable soft-capping
    use_fp32_softmax: bool = True  # Upcast softmax to FP32 for numerical stability
    # --- New/extended config fields ---
    # YaRN / RoPE scaling
    use_yarn: bool = True
    yarn_orig_ctx: int = 2048
    yarn_target_ctx: int = 8192   # e.g., 8k context; set higher if you want 16k/32k later
    yarn_alpha: float = 1.0       # frequency exponent scaling
    yarn_beta: float = 1.0        # magnitude adjustment; keep 1.0 to start
    rope_base: float = 10000.0    # keep your existing base (or whatever you currently use)
    rope_type: str = "auto"
    long_context_mode: str = "yarn"
    # Reasoning layers
    use_lcr: bool = True
    use_gtr: bool = True
    lcr_kernel_size: int = 7      # odd >=3
    lcr_expand: int = 2           # channel expand factor for pointwise conv
    gtr_num_tokens: int = 8       # number of global tokens
    lcr_block_indices: Optional[List[int]] = None
    gtr_block_indices: Optional[List[int]] = None
    # gtr_num_heads removed - deprecated
    # KV cache management
    max_cache_len: int = 0        # Auto-set to block_size unless overridden
    allow_hybrid_cache: bool = False  # Force cache usage even with hybrid layers (experimental)
    use_activation_checkpointing: bool = False  # Enable torch.utils.checkpoint during training

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.block_size <= 0:
            raise ValueError(f"block_size must be positive, got {self.block_size}")
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {self.vocab_size}")
        if self.n_layer <= 0:
            raise ValueError(f"n_layer must be positive, got {self.n_layer}")
        if self.n_head <= 0:
            raise ValueError(f"n_head must be positive, got {self.n_head}")
        if self.n_embd <= 0:
            raise ValueError(f"n_embd must be positive, got {self.n_embd}")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"
            )
        # Normalize and validate dtype
        if isinstance(self.dtype, torch.dtype):
            if self.dtype is torch.float16:
                self.dtype = 'float16'
            elif self.dtype is torch.bfloat16:
                self.dtype = 'bfloat16'
            elif self.dtype is torch.float32:
                self.dtype = 'float32'
            else:
                self.dtype = 'float32'
        if self.dtype is not None:
            valid_dtypes = ['float16', 'bfloat16', 'float32']
            if str(self.dtype).lower() not in valid_dtypes:
                raise ValueError(f"dtype must be one of {valid_dtypes}, got {self.dtype}")
        # Auto-detect dtype based on device capability if not specified
        if (self.dtype is None or str(self.dtype).lower() not in ['float16','bfloat16','float32']) and torch.cuda.is_available():
            # BF16 supported on Ampere+ (compute capability >= 8.0)
            props = torch.cuda.get_device_properties(0)
            compute_capability = props.major + props.minor / 10.0
            if compute_capability >= 8.0:
                self.dtype = 'bfloat16'  # Better for training, less precision loss
            else:
                self.dtype = 'float16'  # Works on older GPUs
        elif self.dtype is None or str(self.dtype).lower() not in ['float16','bfloat16','float32']:
            self.dtype = 'float32'  # CPU default
        valid_rope_types = {"auto", *ROPE_BUILDERS.keys()}
        self.rope_type = (self.rope_type or "auto").lower()
        if self.rope_type not in valid_rope_types:
            raise ValueError(f"rope_type must be one of {sorted(valid_rope_types)}, got {self.rope_type}")
        if self.rope_type == "auto":
            self.rope_type = "yarn" if self.use_yarn else "default"
        valid_long_modes = {"none", "yarn"}
        self.long_context_mode = (self.long_context_mode or "yarn").lower()
        if self.long_context_mode not in valid_long_modes:
            raise ValueError(f"long_context_mode must be one of {sorted(valid_long_modes)}, got {self.long_context_mode}")
        if self.long_context_mode == "yarn":
            if not self.use_yarn:
                logger.info("long_context_mode='yarn' enabling YaRN scaling.")
            self.use_yarn = True
        else:
            self.use_yarn = False
        if self.n_kv_groups is None:
            self.n_kv_groups = self.n_head
        if self.n_kv_groups is not None:
            if self.n_kv_groups <= 0:
                raise ValueError(f"n_kv_groups must be positive, got {self.n_kv_groups}")
            if self.n_kv_groups > self.n_head:
                raise ValueError(
                    f"n_kv_groups ({self.n_kv_groups}) cannot exceed n_head ({self.n_head})"
                )
            if self.n_head % self.n_kv_groups != 0:
                raise ValueError(
                    f"n_head ({self.n_head}) must be divisible by n_kv_groups ({self.n_kv_groups})"
                )
        if self.attn_logit_softcapping is not None:
            if self.attn_logit_softcapping <= 0:
                raise ValueError(f"attn_logit_softcapping must be positive, got {self.attn_logit_softcapping}")
        # Validate YaRN parameters
        if self.yarn_orig_ctx <= 0:
            raise ValueError(f"yarn_orig_ctx must be positive, got {self.yarn_orig_ctx}")
        if self.yarn_target_ctx <= 0:
            raise ValueError(f"yarn_target_ctx must be positive, got {self.yarn_target_ctx}")
        # If target_ctx <= orig_ctx, disable YaRN scaling
        if self.use_yarn and self.yarn_target_ctx <= self.yarn_orig_ctx:
            logger.info(f"YaRN target_ctx ({self.yarn_target_ctx}) <= orig_ctx ({self.yarn_orig_ctx}), disabling YaRN")
            self.use_yarn = False
        if self.yarn_alpha <= 0:
            raise ValueError(f"yarn_alpha must be positive, got {self.yarn_alpha}")
        if self.yarn_beta <= 0:
            raise ValueError(f"yarn_beta must be positive, got {self.yarn_beta}")
        # Validate reasoning parameters
        if self.lcr_kernel_size < 3 or self.lcr_kernel_size % 2 == 0:
            raise ValueError(f"lcr_kernel_size must be odd and >= 3, got {self.lcr_kernel_size}")
        if self.lcr_expand <= 0:
            raise ValueError(f"lcr_expand must be positive, got {self.lcr_expand}")
        if self.gtr_num_tokens <= 0:
            raise ValueError(f"gtr_num_tokens must be positive, got {self.gtr_num_tokens}")
        def _normalize_positions(name: str, values: Optional[List[int]]) -> List[int]:
            if not values:
                return []
            unique = sorted(set(values))
            for idx in unique:
                if idx < 0 or idx >= self.n_layer:
                    raise ValueError(f"{name} index {idx} is outside valid layer range [0, {self.n_layer - 1}]")
            return unique
        if self.use_lcr:
            self.lcr_block_indices = _normalize_positions("lcr_block_indices", self.lcr_block_indices)
            if not self.lcr_block_indices:
                self.lcr_block_indices = [max(1, self.n_layer // 2)]
        else:
            self.lcr_block_indices = []
        if self.use_gtr:
            self.gtr_block_indices = _normalize_positions("gtr_block_indices", self.gtr_block_indices)
            if not self.gtr_block_indices:
                self.gtr_block_indices = [max(self.n_layer - 3, 0)]
        else:
            self.gtr_block_indices = []
        # gtr_num_heads validation removed - deprecated parameter
        # Validate KV cache parameters
        if self.max_cache_len <= 0:
            self.max_cache_len = self.block_size

        # Allow max_cache_len > block_size when using YaRN with extended context
        effective_context_limit = self.block_size
        if self.use_yarn and self.yarn_target_ctx > self.block_size:
            effective_context_limit = self.yarn_target_ctx

        if self.max_cache_len > effective_context_limit:
            raise ValueError(
                f"max_cache_len ({self.max_cache_len}) cannot exceed effective context limit ({effective_context_limit})"
            )
        # Validate attention sink parameters
        self.use_activation_checkpointing = bool(self.use_activation_checkpointing)


class ModelArchitecture(nn.Module):
    """Core GPT model architecture."""

    def __init__(self, config: "ModelSettings") -> None:
        """Initialize the model architecture.

        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config
        self.gradient_checkpointing = bool(config.use_activation_checkpointing)

        # Global numerical stability: disable reduced precision reductions to prevent overflow
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

        # Initialize RoPE helper from registry
        self.rope = build_rope(config)
        logger.info("Initialized RoPE type: %s", config.rope_type)

        depth = config.n_layer

        lcr_positions = set(config.lcr_block_indices if config.use_lcr else [])
        gtr_positions = set(config.gtr_block_indices if config.use_gtr else [])
        overlap = lcr_positions & gtr_positions
        if overlap:
            raise ValueError(f"LCR and GTR blocks cannot share the same indices: {sorted(overlap)}")

        layers: List[nn.Module] = []
        for i in range(depth):
            if i in lcr_positions:
                layers.append(
                    LCRBlock(
                        config.n_embd,
                        kernel_size=config.lcr_kernel_size,
                        expand=config.lcr_expand,
                        dropout=config.dropout,
                        bias=config.bias,
                    )
                )
            elif i in gtr_positions:
                layers.append(
                    GTRBlock(
                        config.n_embd,
                        num_tokens=config.gtr_num_tokens,
                        dropout=config.dropout,
                        bias=config.bias,
                    )
                )
            else:
                layers.append(TransformerLayer(config))

        self.transformer = nn.ModuleDict(dict(
            token_embeddings = nn.Embedding(config.vocab_size, config.n_embd),
            dropout = nn.Dropout(config.dropout),
            layers = nn.ModuleList(layers),
            final_norm = RMSNorm(config.n_embd),
        ))

        logger.info(
            "Depth=%d; LCR positions=%s; GTR positions=%s; long_context_mode=%s",
            depth,
            config.lcr_block_indices if config.use_lcr else "disabled",
            config.gtr_block_indices if config.use_gtr else "disabled",
            config.long_context_mode,
        )

        self.language_model_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.transformer.token_embeddings.weight = self.language_model_head.weight

        self.apply(self._init_weights)

        for param_name, param in self.named_parameters():
            if param_name.endswith('output_projection.weight') or param_name.endswith('language_model_head.weight'):
                # Special initialization for output projections and language model head
                torch.nn.init.normal_(param, mean=0.0, std=WEIGHT_INIT_STD/math.sqrt(2 * config.n_layer))

        param_count = self.get_parameter_count()
        logger.info(f"Model initialized with {param_count/1e6:.2f}M parameters")
        logger.debug(f"Model config: n_layer={config.n_layer}, n_head={config.n_head}, "
                    f"n_embd={config.n_embd}, vocab_size={config.vocab_size}, block_size={config.block_size}")

    def enable_lora(self, r: int = 8, lora_alpha: int = 16, lora_dropout: float = 0.05,
                   target_modules: Optional[List[str]] = None):
        """Enable LoRA (Low-Rank Adaptation) and return a PEFT-wrapped model.

        Args:
            r: LoRA rank
            lora_alpha: LoRA scaling factor
            lora_dropout: LoRA dropout rate
            target_modules: Module names to apply LoRA to. If None, inferred from the model.

        Returns:
            A PEFT-wrapped model (PeftModel) ready for training.
        """
        try:
            from peft import LoraConfig, get_peft_model  # type: ignore
        except ImportError:
            raise ImportError("PEFT library required for LoRA. Install with: pip install peft")

        if target_modules is None:
            # Infer sensible defaults based on available projection names
            target_modules = self._infer_lora_target_modules()

        lora_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )

        logger.info(
            "Enabling LoRA with r=%s, alpha=%s, dropout=%s on modules=%s",
            r, lora_alpha, lora_dropout, target_modules,
        )

        wrapped = get_peft_model(self, lora_config)
        # Expose for downstream tooling
        setattr(wrapped, "lora_config", lora_config)
        setattr(wrapped, "lora_enabled", True)
        return wrapped

    def _infer_lora_target_modules(self) -> List[str]:
        """Detect likely attention projection modules for LoRA.

        Returns a list including fused and/or separate QKV projections plus output projection
        depending on what exists in the current model.
        """
        candidates = {"qkv_projection", "query_projection", "key_projection", "value_projection", "output_projection"}
        found: set[str] = set()
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                leaf = name.rsplit(".", 1)[-1]
                if leaf in candidates:
                    found.add(leaf)
        if not found:
            # Fall back to broad default across known names
            return sorted(candidates)
        return sorted(found)

    @classmethod
    def create_with_lora(cls, config: "ModelSettings", r: int = 8, lora_alpha: int = 16,
                        lora_dropout: float = 0.05, target_modules: Optional[List[str]] = None):
        """Create a model with LoRA enabled from the start.

        Args:
            config: Model configuration
            r: LoRA rank
            lora_alpha: LoRA scaling factor
            lora_dropout: LoRA dropout rate
            target_modules: List of module names to apply LoRA to

        Returns:
            LoRA-wrapped model
        """
        try:
            from peft import LoraConfig, get_peft_model  # type: ignore
        except ImportError:
            raise ImportError("PEFT library required for LoRA. Install with: pip install peft")

        # Create base model
        model = cls(config)

        if target_modules is None:
            target_modules = model._infer_lora_target_modules()

        lora_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )

        # Wrap with LoRA
        lora_model = get_peft_model(model, lora_config)
        setattr(lora_model, "lora_config", lora_config)
        setattr(lora_model, "lora_enabled", True)

        try:
            trainable = lora_model.get_nb_trainable_parameters()  # type: ignore[attr-defined]
            logger.info(f"Created model with LoRA: r={r}, alpha={lora_alpha}, target_modules={target_modules}")
            logger.info(f"Trainable parameters: {trainable}")
        except Exception:
            logger.info(f"Created model with LoRA: r={r}, alpha={lora_alpha}, target_modules={target_modules}")

        return lora_model

    def get_parameter_count(self, include_embeddings: bool = True) -> int:
        """Get the number of parameters in the model.

        Args:
            include_embeddings: Whether to include token embeddings in count

        Returns:
            Number of parameters
        """
        if include_embeddings:
            # Count all parameters
            with torch.no_grad():
                param_count = sum(p.numel() for p in self.parameters())
        else:
            # Exclude embeddings - only count non-embedding parameters
            param_count = 0
            for name, module in self.named_modules():
                if isinstance(module, nn.Embedding):
                    continue  # Skip all embedding layers
                with torch.no_grad():
                    param_count += sum(p.numel() for p in module.parameters())
        return param_count

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize model weights using LLaMA-style initialization.

        Args:
            module: Module to initialize
        """
        if isinstance(module, nn.Linear):
            in_features, out_features = module.weight.shape
            std = math.sqrt(2.0 / (in_features + out_features))
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=WEIGHT_INIT_STD)

    def forward(
        self,
        token_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        position_ids: Optional[torch.Tensor] = None,
        return_full_logits: bool = False,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        skip_lm_head: bool = False,
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[List[Tuple[torch.Tensor, torch.Tensor]]],
        Optional[torch.Tensor],
        Optional[Tuple[Optional[torch.Tensor], ...]],
    ]:
        """Forward pass through the model with optional KV caching.

        Args:
            token_ids: Input token IDs of shape (batch_size, seq_len)
            targets: Optional target token IDs for training
            attention_mask: Optional attention mask of shape (batch_size, seq_len) where 1 = real token, 0 = padding.
                           Used to mask padding tokens in attention computation and loss calculation.
            past_key_values: Optional list of cached key/value states for each layer
            use_cache: Whether to return cached key/value states
            position_ids: Optional position IDs for KV caching
            return_full_logits: When True and targets is None, return logits for all tokens instead of
                just the last position. Defaults to False for generation use-cases.
            output_hidden_states: When True, also return the final hidden states before the LM head.
            skip_lm_head: When True, skip applying the LM head entirely. Requires output_hidden_states=True and
                is incompatible with providing targets/return_full_logits.

        Returns:
            Tuple of (logits, loss, past_key_values) where:
            - logits: Model output logits
            - loss: Loss value (None during inference). Padding tokens are automatically ignored via attention_mask.
            - past_key_values: Cached key/value states if use_cache=True
        """
        if token_ids.dim() != 2:
            raise ValueError(f"token_ids must be 2D tensor, got shape {token_ids.shape}")
        if token_ids.dtype != torch.long:
            raise ValueError(f"token_ids must be long tensor, got {token_ids.dtype}")

        if skip_lm_head:
            if not output_hidden_states:
                raise ValueError("skip_lm_head=True requires output_hidden_states=True to return meaningful data.")
            if targets is not None or return_full_logits:
                raise ValueError(
                    "skip_lm_head=True cannot be combined with targets or return_full_logits since logits are skipped."
                )

        device = token_ids.device
        batch_size, seq_len = token_ids.size()

        # Fix: Handle edge cases for seq_len=0 or seq_len=1
        if seq_len == 0:
            raise ValueError("Sequence length must be positive, got 0")
        
        if seq_len > self.config.block_size:
            raise ValueError(f"Sequence length {seq_len} exceeds block size {self.config.block_size}")

        # Token ID policy: validate in training, clamp in inference for robustness
        if self.training:
            if token_ids.min().item() < 0 or token_ids.max().item() >= self.config.vocab_size:
                raise ValueError(
                    f"token_ids out of range [0, {self.config.vocab_size - 1}] during training"
                )
        else:
            token_ids = torch.clamp(token_ids, 0, self.config.vocab_size - 1)

        auto_targets = targets is None
        hybrid_blocks_enabled = bool(self.config.use_lcr or self.config.use_gtr)
        hybrid_cache_blocked = hybrid_blocks_enabled and not getattr(self.config, "allow_hybrid_cache", False)

        if hybrid_cache_blocked:
            # Be permissive: ignore provided caches and disable returning new caches
            if past_key_values:
                if not getattr(self, "_hybrid_cache_warning_emitted", False):
                    logger.warning(
                        "Ignoring provided past_key_values because hybrid LCR/GTR blocks are active and "
                        "allow_hybrid_cache=False. Full prefix will be recomputed each step."
                    )
                past_key_values = None
            if use_cache and not getattr(self, "_hybrid_cache_warning_emitted", False):
                logger.warning(
                    "Disabling use_cache because LCR/GTR blocks require full sequence context. "
                    "Set allow_hybrid_cache=True to enable transformer-layer KV caching."
                )
                self._hybrid_cache_warning_emitted = True
            if use_cache:
                use_cache = False

        # Forward pass with centralized mixed precision
        dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16}
        use_autocast = device.type == "cuda" and self.config.dtype in dtype_map
        if use_autocast:
            autocast_ctx = torch.autocast(device_type=device.type, dtype=dtype_map[self.config.dtype], enabled=True)
        else:
            autocast_ctx = contextlib.nullcontext()

        with autocast_ctx:
            token_embeds = self.transformer.token_embeddings(token_ids)
            hidden_states = self.transformer.dropout(token_embeds)

            freq_device = hidden_states.device
            freq_dtype = hidden_states.dtype if hidden_states.dtype in (torch.float16, torch.bfloat16, torch.float32) else torch.float32
            freq_cache: Dict[int, torch.Tensor] = {}
            attentions_output: List[Optional[torch.Tensor]] = [] if output_attentions else []

            def get_freqs(start: int) -> torch.Tensor:
                if start not in freq_cache:
                    freq_cache[start] = self.rope._frequencies(
                        seq_len=seq_len,
                        start=start,
                        device=freq_device,
                        dtype=freq_dtype,
                    )
                return freq_cache[start]

            # Initialize past_key_values list if using cache
            present_key_values = [] if use_cache else None

            # Dynamic KV cache extension for long sequences
            if past_key_values and seq_len > 0:
                # Check if we need to extend RoPE cache beyond block_size
                max_cache_len = max(layer_past[0].shape[2] for layer_past in past_key_values if layer_past[0] is not None) if past_key_values else 0
                if seq_len + max_cache_len > self.config.block_size:
                    # Extend rope freqs dynamically with YaRN extrapolation
                    self.rope.resize_cache(max_seq_len=seq_len + max_cache_len)

            # Process through layers with KV caching and different block types
            kv_index = 0
            for layer in self.transformer.layers:
                layer_past = None

                if isinstance(layer, TransformerLayer):
                    if past_key_values is not None and kv_index < len(past_key_values):
                        layer_past = past_key_values[kv_index]
                    layer_cache_len = 0
                    if layer_past is not None and layer_past[0] is not None:
                        layer_cache_len = int(layer_past[0].shape[2])
                    freqs = get_freqs(layer_cache_len)
                    # Standard transformer layer with YaRN RoPE
                    if (
                        self.config.use_activation_checkpointing
                        and self.training
                        and not use_cache
                    ):
                        def layer_forward(hs: torch.Tensor) -> torch.Tensor:
                            output, _ = layer(
                                hs,
                                attention_mask=attention_mask,
                                position_ids=position_ids,
                                past_key_value=None,
                                use_cache=False,
                                rope=self.rope,
                                freqs=freqs,
                            )
                            return output

                        hidden_states = checkpoint(layer_forward, hidden_states)
                        layer_present = None
                    else:
                        hidden_states, layer_present, layer_attention = layer(
                            hidden_states,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            past_key_value=layer_past,
                            use_cache=use_cache,
                            rope=self.rope,
                            freqs=freqs,
                            output_attentions=output_attentions,
                        )
                        if output_attentions:
                            attentions_output.append(layer_attention)
                    kv_index += 1
                elif isinstance(layer, LCRBlock):
                    # Local Convolutional Reasoning block
                    if (
                        self.config.use_activation_checkpointing
                        and self.training
                        and not use_cache
                    ):
                        def lcr_forward(hs: torch.Tensor) -> torch.Tensor:
                            return layer(hs)
                        hidden_states = checkpoint(lcr_forward, hidden_states)
                    else:
                        hidden_states = layer(hidden_states)
                    layer_present = None
                    if output_attentions:
                        attentions_output.append(None)
                elif isinstance(layer, GTRBlock):
                    # Global Token Reasoning block
                    if (
                        self.config.use_activation_checkpointing
                        and self.training
                        and not use_cache
                    ):
                        def gtr_forward(hs: torch.Tensor) -> torch.Tensor:
                            return layer(hs, attn_mask=attention_mask)
                        hidden_states = checkpoint(gtr_forward, hidden_states)
                    else:
                        hidden_states = layer(hidden_states, attn_mask=attention_mask)
                    layer_present = None
                    if output_attentions:
                        attentions_output.append(None)
                else:
                    # Fallback for future block types
                    hidden_states = layer(hidden_states)
                    layer_present = None
                    if output_attentions:
                        attentions_output.append(None)

                if isinstance(layer, TransformerLayer) and use_cache and layer_present is not None:
                    present_key_values.append(layer_present)

            hidden_states = self.transformer.final_norm(hidden_states)
            hidden_output = hidden_states if output_hidden_states else None

            logits = None
            logits_full = None
            if not skip_lm_head:
                logits_full = self.language_model_head(hidden_states)
                if targets is not None or return_full_logits:
                    logits = logits_full
                else:
                    logits = logits_full[:, [-1], :]

            loss = None
            targets_for_loss: Optional[torch.Tensor] = None
            logits_for_loss: Optional[torch.Tensor] = None
            mask_for_loss: Optional[torch.Tensor] = None

            if logits_full is not None:
                if auto_targets and seq_len > 1:
                    logits_for_loss = logits_full[:, :-1, :].contiguous()
                    targets_for_loss = token_ids[:, 1:].contiguous()
                    if attention_mask is not None:
                        mask_for_loss = attention_mask[:, 1:].contiguous()
                elif not auto_targets and targets is not None:
                    logits_for_loss = logits_full
                    targets_for_loss = targets
                    mask_for_loss = attention_mask

            if logits_for_loss is not None and targets_for_loss is not None:
                targets_flat = targets_for_loss.view(-1).clone()
                if mask_for_loss is not None:
                    mask_flat = mask_for_loss.view(-1)
                    targets_flat[mask_flat == 0] = -1
                loss = F.cross_entropy(
                    logits_for_loss.view(-1, logits_for_loss.size(-1)),
                    targets_flat,
                    ignore_index=-1,
                )

        attentions_tuple: Optional[Tuple[Optional[torch.Tensor], ...]] = None
        if output_attentions:
            attentions_tuple = tuple(attentions_output)

        present = present_key_values if use_cache else None
        return logits, loss, present, hidden_output if output_hidden_states else None, attentions_tuple

    def crop_context_window(self, new_block_size: int) -> None:
        """Reduce the model's maximum context window.

        Args:
            new_block_size: New maximum sequence length

        Raises:
            ValueError: If new_block_size is larger than current block_size
        """
        if new_block_size > self.config.block_size:
            raise ValueError(f"Cannot increase block size from {self.config.block_size} to {new_block_size}")

        self.config.block_size = new_block_size

        for layer in self.transformer.layers:
            if not hasattr(layer, "attention"):
                continue

            attn = layer.attention

            if hasattr(attn, 'causal_mask') and attn.causal_mask is not None:
                device = attn.causal_mask.device
                dtype = attn.causal_mask.dtype
                causal = torch.tril(torch.ones(new_block_size, new_block_size, device=device, dtype=dtype))
                attn.causal_mask = causal.view(1, 1, new_block_size, new_block_size)

            if hasattr(attn, 'rope'):
                if isinstance(attn.rope, RotaryPositionEmbedding):
                    attn.rope = RotaryPositionEmbedding(attn.head_size, new_block_size)
                elif hasattr(attn.rope, 'apply_rope'):  # RopeWithYaRN
                    # Rebuild YaRN rope with new target context
                    config = self.config
                    attn.rope = RopeWithYaRN(
                        dim=attn.head_size,
                        base=config.rope_base,
                        orig_ctx=config.yarn_orig_ctx,
                        target_ctx=new_block_size,  # Use new block size as target
                        alpha=config.yarn_alpha,
                        beta=config.yarn_beta,
                        enabled=config.use_yarn,
                        device=None,
                        dtype=None
                    )


    def get_input_embeddings(self) -> nn.Embedding:
        return self.transformer.token_embeddings

    def set_input_embeddings(self, new_embeddings: nn.Embedding) -> None:
        self.transformer.token_embeddings = new_embeddings
        self.language_model_head.weight = new_embeddings.weight


# Exported classes are now in separate modules:
# - OptimizerFactory, PerformanceMonitor: training.py
# - TextGenerator: generator.py

# Alias for backward compatibility
GPTConfig = ModelSettings
