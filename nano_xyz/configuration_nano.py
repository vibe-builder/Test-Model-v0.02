"""Hugging Face configuration wrapper for the Nano XYZ model."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from transformers import PretrainedConfig
import torch

from .model import DEFAULT_BLOCK_SIZE, DEFAULT_N_EMBD, DEFAULT_N_HEAD, DEFAULT_N_LAYER, ModelSettings

logger = logging.getLogger(__name__)


class NanoConfig(PretrainedConfig):
    """Configuration compatible with the Hugging Face ecosystem."""

    model_type = "nano_xyz"
    # Map HF-standard fields to our internal names for better interoperability
    attribute_map = {
        "num_hidden_layers": "n_layer",
        "num_attention_heads": "n_head",
        "hidden_size": "n_embd",
    }

    def __init__(
        self,
        block_size: int = DEFAULT_BLOCK_SIZE,
        vocab_size: int = 50257,
        n_layer: int = DEFAULT_N_LAYER,
        n_head: int = DEFAULT_N_HEAD,
        n_embd: int = DEFAULT_N_EMBD,
        dropout: float = 0.0,
        bias: bool = True,
        n_kv_groups: Optional[int] = None,
        dtype: Optional[str] = None,
        use_fused_attention: bool = True,
        attn_logit_softcapping: Optional[float] = None,
        use_fp32_softmax: bool = True,
        use_yarn: bool = True,
        yarn_orig_ctx: int = 2048,
        yarn_target_ctx: int = 8192,
        yarn_alpha: float = 1.0,
        yarn_beta: float = 1.0,
        rope_base: float = 10000.0,
        rope_type: str = "auto",
        long_context_mode: str = "yarn",
        use_lcr: bool = True,
        use_gtr: bool = True,
        lcr_kernel_size: int = 7,
        lcr_expand: int = 2,
        gtr_num_tokens: int = 8,
        lcr_block_indices: Optional[list] = None,
        gtr_block_indices: Optional[list] = None,
        max_cache_len: int = 0,
        use_activation_checkpointing: bool = False,
        max_position_embeddings: Optional[int] = None,
        allow_hybrid_cache: bool = False,
        torch_dtype: Optional[str] = None,
        tie_word_embeddings: bool = True,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        bos_token_id: Optional[int] = None,
        use_return_dict: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            bos_token_id=bos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            torch_dtype=torch_dtype,
            return_dict=use_return_dict,
            **kwargs,
        )
        self.block_size = block_size
        self.max_position_embeddings = max_position_embeddings or block_size
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.bias = bias
        self.n_kv_groups = n_kv_groups
        self.dtype = dtype
        self.use_fused_attention = use_fused_attention
        self.attn_logit_softcapping = attn_logit_softcapping
        self.use_fp32_softmax = use_fp32_softmax
        self.use_yarn = use_yarn
        self.yarn_orig_ctx = yarn_orig_ctx
        self.yarn_target_ctx = yarn_target_ctx
        self.yarn_alpha = yarn_alpha
        self.yarn_beta = yarn_beta
        self.rope_base = rope_base
        self.rope_type = rope_type
        self.long_context_mode = long_context_mode
        self.use_lcr = use_lcr
        self.use_gtr = use_gtr
        self.lcr_kernel_size = lcr_kernel_size
        self.lcr_expand = lcr_expand
        self.gtr_num_tokens = gtr_num_tokens
        self.lcr_block_indices = lcr_block_indices
        self.gtr_block_indices = gtr_block_indices
        self.max_cache_len = max_cache_len
        self.use_activation_checkpointing = use_activation_checkpointing
        self.allow_hybrid_cache = allow_hybrid_cache
        self.torch_dtype = torch_dtype if torch_dtype is not None else getattr(self, "torch_dtype", None)
        self.tie_word_embeddings = tie_word_embeddings
        self.return_dict = use_return_dict
        self._validate()

    def to_model_settings(self) -> ModelSettings:
        """Convert HF config into the internal ModelSettings dataclass."""
        def _normalize_dtype(val):
            if val is None:
                return None
            if isinstance(val, str):
                v = val.lower()
                if v in {"float16", "bfloat16", "float32"}:
                    return v
                return v
            if isinstance(val, torch.dtype):
                if val is torch.float16:
                    return "float16"
                if val is torch.bfloat16:
                    return "bfloat16"
                if val is torch.float32:
                    return "float32"
            return str(val)
        return ModelSettings(
            block_size=self.block_size,
            vocab_size=self.vocab_size,
            n_layer=self.n_layer,
            n_head=self.n_head,
            n_embd=self.n_embd,
            dropout=self.dropout,
            bias=self.bias,
            n_kv_groups=self.n_kv_groups,
            dtype=_normalize_dtype(self.dtype),
            use_fused_attention=self.use_fused_attention,
            attn_logit_softcapping=self.attn_logit_softcapping,
            use_fp32_softmax=self.use_fp32_softmax,
            use_yarn=self.use_yarn,
            yarn_orig_ctx=self.yarn_orig_ctx,
            yarn_target_ctx=self.yarn_target_ctx,
            yarn_alpha=self.yarn_alpha,
            yarn_beta=self.yarn_beta,
            rope_base=self.rope_base,
            rope_type=self.rope_type,
            long_context_mode=self.long_context_mode,
            use_lcr=self.use_lcr,
            use_gtr=self.use_gtr,
            lcr_kernel_size=self.lcr_kernel_size,
            lcr_expand=self.lcr_expand,
            gtr_num_tokens=self.gtr_num_tokens,
            lcr_block_indices=self.lcr_block_indices,
            gtr_block_indices=self.gtr_block_indices,
            max_cache_len=self.max_cache_len,
            use_activation_checkpointing=self.use_activation_checkpointing,
            allow_hybrid_cache=self.allow_hybrid_cache,
        )

    @classmethod
    def from_model_settings(cls, settings: ModelSettings, **kwargs: Any) -> "NanoConfig":
        """Helper to build a config from an existing ModelSettings object."""
        data: Dict[str, Any] = settings.__dict__.copy()
        data.update(kwargs)
        return cls(**data)

    def _validate(self) -> None:
        """Validate configuration coherence."""
        if self.max_position_embeddings and self.max_position_embeddings < self.block_size:
            raise ValueError(
                f"max_position_embeddings ({self.max_position_embeddings}) cannot be less than block_size "
                f"({self.block_size})"
            )
        if self.pad_token_id is None and self.eos_token_id is not None:
            logger.info("pad_token_id missing; defaulting to eos_token_id=%s", self.eos_token_id)
            self.pad_token_id = self.eos_token_id

        hybrid_blocks_enabled = bool(self.use_lcr or self.use_gtr)
        if hybrid_blocks_enabled and self.allow_hybrid_cache:
            logger.warning(
                "allow_hybrid_cache=True: Nano will attempt KV caching even though LCR/GTR blocks are active. "
                "This path is experimental and may recompute hybrid layers each step."
            )
        if hybrid_blocks_enabled and not self.allow_hybrid_cache and self.max_cache_len > 0:
            logger.info(
                "Hybrid blocks detected; KV caching will be disabled unless allow_hybrid_cache=True."
            )

        if self.max_cache_len > 0 and self.max_cache_len < self.block_size:
            logger.warning(
                "max_cache_len (%s) is smaller than block_size (%s); long sequences may evict cached tokens early.",
                self.max_cache_len,
                self.block_size,
            )
