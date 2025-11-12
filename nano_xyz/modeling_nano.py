"""Hugging Face compatible Nano XYZ modeling classes."""

from __future__ import annotations

import importlib.metadata
import logging
from typing import Optional, Tuple, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from transformers.cache_utils import Cache

import torch
from torch import nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    CausalLMOutputWithCrossAttentions,
    Seq2SeqLMOutput,
)

# Check transformers version for proper imports
_transformers_version = tuple(map(int, importlib.metadata.version("transformers").split(".")))

# GenerationMixin import - available in transformers >= 4.35.0 in generation.utils
if _transformers_version >= (4, 35, 0):
    from transformers.generation.utils import GenerationMixin
else:
    # Fallback for older versions
    try:
        from transformers.generation import GenerationMixin  # type: ignore
    except ImportError:
        # Minimal fallback class for basic compatibility
        class GenerationMixin:
            """Minimal GenerationMixin fallback for older transformers versions."""
            pass

# Auto registration functions - API changed in different versions
if _transformers_version >= (4, 33, 0):
    try:
        # Try the old API first (transformers < 4.48.0)
        from transformers.utils import register_for_auto_class
    except ImportError:
        # New API (transformers >= 4.48.0) - create wrapper functions
        from transformers.models.auto.configuration_auto import AutoConfig
        from transformers.models.auto.modeling_auto import AutoModel, AutoModelForCausalLM

        def register_for_auto_class(auto_class: str):
            def decorator(cls):
                if auto_class == "auto-config":
                    AutoConfig.register(cls.model_type, cls)
                elif auto_class == "auto-model":
                    AutoModel.register(cls.model_type, cls)
                elif auto_class == "auto-model-for-causal-lm":
                    AutoModelForCausalLM.register(cls.model_type, cls)
                return cls
            return decorator
else:
    # Fallback for very old versions
    def register_for_auto_class(_auto_class: str):  # pragma: no cover
        def decorator(cls):
            return cls
        return decorator

from .configuration_nano import NanoConfig
from .base import NanoPreTrainedModel
from .model import NanoModel, NanoEncoder, NanoDecoder
from .cache_utils import get_past_key_values_length

logger = logging.getLogger(__name__)


# Register the configuration class
if _transformers_version >= (4, 33, 0):
    from .configuration_nano import NanoConfig
    AutoConfig.register("nano", NanoConfig)


@register_for_auto_class("AutoModelForCausalLM")
class NanoForCausalLM(NanoPreTrainedModel, GenerationMixin):
    """Nano XYZ causal LM head compatible with transformers.generate."""

    _keys_to_ignore_on_save = ["lm_head.weight", "model.lm_head.weight"]

    def __init__(self, config: NanoConfig) -> None:
        super().__init__(config)
        self.model = NanoModel(config)
        self.lm_head = self.model.lm_head
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[list] = None,
        use_cache: bool = True,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        position_ids = kwargs.pop("position_ids", None)
        output_hidden_states = kwargs.pop("output_hidden_states", False)
        output_attentions = kwargs.pop("output_attentions", False)
        return_dict = kwargs.pop("return_dict", self.config.use_return_dict)

        effective_use_cache = use_cache

        # Apply dynamic quantization for inference if configured
        # Quantization is applied lazily on first inference call
        if (self.config.quantization_config and
            hasattr(self.model, 'quantize_model') and
            not self.training):
            # Check if model has already been quantized (avoid re-quantizing)
            if not hasattr(self, '_quantized_applied'):
                try:
                    self.model.quantize_model()
                    self._quantized_applied = True
                except Exception as e:
                    logger.warning(f"Failed to apply quantization during inference: {e}")

        model_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=effective_use_cache,
            position_ids=position_ids,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=True,
            **kwargs,
        )
        hidden_states = getattr(model_outputs, 'hidden_states', None)
        if hidden_states is None:
            # Fallback for different ModelOutput formats
            hidden_states = getattr(model_outputs, 'last_hidden_state', None)
        if hidden_states is None:
            raise AttributeError("ModelOutput missing hidden states")
        last_hidden_state = model_outputs.last_hidden_state
        presents = model_outputs.past_key_values
        logits = self.lm_head(last_hidden_state)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            if attention_mask is not None:
                shift_mask = attention_mask[..., 1:].contiguous()
                shift_labels = shift_labels.masked_fill(shift_mask == 0, -100)
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        causal_outputs = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=presents,
            hidden_states=hidden_states,
            attentions=model_outputs.attentions,
        )

        if return_dict:
            return causal_outputs

        output = (logits, presents, hidden_states, model_outputs.attentions)
        if loss is not None:
            output = (loss,) + output
        return output

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[list] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Prepare inputs for generation using HF's built-in helpers.

        This replaces manual tensor manipulation with validated HF GenerationMixin methods
        to ensure consistency and correctness across different generation scenarios.
        """
        # Use HF's standard input preparation for basic input handling
        inputs = self._prepare_model_inputs(
            inputs=input_ids,
            model_kwargs={
                "past_key_values": past_key_values,
                "attention_mask": attention_mask,
                **kwargs
            }
        )

        # Extract the prepared inputs
        prepared_input_ids, model_input_name, model_kwargs = inputs

        # Handle attention mask preparation with bounds checking
        if attention_mask is not None and past_key_values is not None:
            # Use HF's attention mask preparation with proper validation
            generation_config = getattr(self, "generation_config", None)
            if generation_config is not None:
                try:
                    prepared_attention_mask = self._prepare_attention_mask_for_generation(
                        inputs_tensor=prepared_input_ids,
                        generation_config=generation_config,
                        model_kwargs=model_kwargs,
                    )
                    model_kwargs["attention_mask"] = prepared_attention_mask
                except (AttributeError, TypeError):
                    # Fallback to validated manual preparation if HF method fails
                    model_kwargs["attention_mask"] = self._prepare_attention_mask_safely(
                        attention_mask, prepared_input_ids, past_key_values
                    )
            else:
                # Fallback when no generation config available
                model_kwargs["attention_mask"] = self._prepare_attention_mask_safely(
                    attention_mask, prepared_input_ids, past_key_values
                )

        # Update with prepared input_ids
        model_kwargs["input_ids"] = prepared_input_ids

        return model_kwargs

    def _prepare_attention_mask_safely(
        self,
        attention_mask: torch.Tensor,
        prepared_input_ids: torch.Tensor,
        past_key_values: list,
    ) -> torch.Tensor:
        """
        Safely prepare attention mask with bounds checking.

        This handles the case where attention_mask needs to be properly sized for generation,
        accounting for past context and current inputs.
        """
        past_length = get_past_key_values_length(past_key_values)
        batch_size, seq_len = prepared_input_ids.shape

        # Validate input dimensions
        if attention_mask.dim() != 2:
            raise ValueError(f"attention_mask must be 2D, got shape {attention_mask.shape}")

        # The attention_mask should cover the full context (past + current)
        total_expected_length = past_length + seq_len

        # If attention_mask is shorter than expected, we need to extend it
        if attention_mask.size(1) < total_expected_length:
            # Create padding with ones for the missing past context
            padding_length = total_expected_length - attention_mask.size(1)
            padding = attention_mask.new_ones(batch_size, padding_length)
            attention_mask = torch.cat([padding, attention_mask], dim=1)
        elif attention_mask.size(1) > total_expected_length:
            # Truncate if too long (shouldn't happen in normal usage)
            attention_mask = attention_mask[:, :total_expected_length]

        return attention_mask

    def _reorder_cache(self, past: list, beam_idx: torch.Tensor) -> list:
        return [tuple(p.index_select(0, beam_idx) for p in layer) for layer in past]

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def set_input_embeddings(self, new_embeddings: nn.Embedding) -> None:
        self.model.embed_tokens = new_embeddings

    def get_position_embeddings(self) -> Optional[nn.Embedding]:
        """Nano XYZ doesn't use position embeddings (uses RoPE instead)."""
        return None

    def set_position_embeddings(self, new_position_embeddings: nn.Embedding) -> None:
        """Nano XYZ doesn't use position embeddings (uses RoPE instead)."""
        pass

    def get_output_embeddings(self) -> nn.Linear:
        """Get the language modeling head."""
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Linear) -> None:
        """Set the language modeling head."""
        self.lm_head = new_embeddings

    def tie_weights(self) -> None:
        """Ensure embedding and LM head weights stay shared."""
        embedding = self.get_input_embeddings()
        self._tie_or_clone_weights(self.lm_head, embedding)


@register_for_auto_class("AutoModel")
class NanoEncoderModel(NanoPreTrainedModel):
    """Nano XYZ encoder model for sequence-to-sequence tasks.

    Follows Hugging Face's encoder-decoder model pattern:
    https://huggingface.co/docs/transformers/en/model_doc/encoder-decoder

    Separate encoder/decoder models allow reuse with HF's EncoderDecoderModel wrapper,
    providing access to proven generation utilities, training infrastructure, and
    ecosystem compatibility.

    Based on T5/BART architecture: "BART: Denoising Sequence-to-Sequence Pre-training"
    (Lewis et al., 2019) https://arxiv.org/abs/1910.13461 - bidirectional encoder
    """

    config_class = NanoConfig
    _keys_to_ignore_on_save = ["lm_head.weight"]

    def __init__(self, config: NanoConfig) -> None:
        super().__init__(config)
        self.encoder = NanoEncoder(config)
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states or False,
            return_dict=True,
        )

        if not return_dict:
            outputs: Tuple[torch.Tensor, ...] = (encoder_outputs.last_hidden_state,)
            if output_hidden_states and encoder_outputs.hidden_states is not None:
                outputs += (encoder_outputs.hidden_states,)
            return outputs

        return BaseModelOutputWithPast(
            last_hidden_state=encoder_outputs.last_hidden_state,
            hidden_states=encoder_outputs.hidden_states if output_hidden_states else None,
            attentions=None,
        )

    def get_input_embeddings(self) -> nn.Embedding:
        return self.encoder.embed_tokens

    def set_input_embeddings(self, new_embeddings: nn.Embedding) -> None:
        self.encoder.embed_tokens = new_embeddings


@register_for_auto_class("AutoModel")
class NanoDecoderModel(NanoPreTrainedModel):
    """Nano XYZ decoder model for sequence-to-sequence tasks.

    Implements causal self-attention + cross-attention to encoder outputs.
    Follows T5/BART decoder architecture with three sub-layers per decoder block:

    1. Causal self-attention (autoregressive generation)
    2. Cross-attention to encoder outputs (source context)
    3. Feed-forward network

    Proven by: "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer"
    (Raffel et al., 2019) https://arxiv.org/abs/1910.10683 - Section 2.1.2 "Decoder"

    And: "BART: Denoising Sequence-to-Sequence Pre-training"
    (Lewis et al., 2019) https://arxiv.org/abs/1910.13461 - left-to-right decoder with cross-attention
    """

    config_class = NanoConfig
    _keys_to_ignore_on_save = ["lm_head.weight"]

    def __init__(self, config: NanoConfig) -> None:
        super().__init__(config)
        self.decoder = NanoDecoder(config)
        self.lm_head = self.decoder.lm_head
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[list] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        return_hidden_states = output_hidden_states if output_hidden_states is not None else False

        decoder_outputs = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_outputs=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_hidden_states=return_hidden_states,
        )

        hidden_states = decoder_outputs.get("hidden_states")
        logits = decoder_outputs.get("logits")
        if logits is None and hidden_states is not None:
            logits = self.lm_head(hidden_states)

        past = decoder_outputs.get("past_key_values")

        if not return_dict:
            outputs: Tuple[torch.Tensor, ...] = (logits, past)
            if return_hidden_states and hidden_states is not None:
                outputs += (hidden_states,)
            else:
                outputs += (None,)
            outputs += (None, None)  # attentions, cross_attentions
            return outputs

        return CausalLMOutputWithCrossAttentions(
            loss=None,
            logits=logits,
            past_key_values=past,
            hidden_states=hidden_states if return_hidden_states else None,
            attentions=None,
            cross_attentions=None,
        )

    def get_input_embeddings(self) -> nn.Embedding:
        return self.decoder.embed_tokens

    def set_input_embeddings(self, new_embeddings: nn.Embedding) -> None:
        self.decoder.embed_tokens = new_embeddings

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Linear) -> None:
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[list] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict:
        """Prepare inputs for generation."""
        # Decoder expects decoder_input_ids, not input_ids
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_attention_mask": encoder_attention_mask,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
        }


# Encoder-Decoder Model Imports
from transformers import EncoderDecoderModel as HFEncoderDecoderModel
from .configuration_nano import NanoConfig


class NanoEncoderDecoderModel(nn.Module):
    """
    Nano XYZ Encoder-Decoder model for sequence-to-sequence tasks.

    Combines NanoEncoder and NanoDecoder with cross-attention for tasks like:
    - Machine translation
    - Text summarization
    - Question answering
    - Any seq2seq task

    Supports HuggingFace generate() methods with proper attention mask handling.
    """

    def __init__(self, config: NanoConfig):
        """
        Initialize encoder-decoder model.

        Args:
            config: NanoConfig with encoder/decoder specifications
        """
        super().__init__()

        # Create separate configs for encoder and decoder
        encoder_config = NanoConfig(**config.__dict__)
        encoder_config.is_decoder = False
        encoder_config.add_cross_attention = False

        decoder_config = NanoConfig(**config.__dict__)
        decoder_config.is_decoder = True
        decoder_config.add_cross_attention = True

        # Create encoder and decoder with appropriate configs
        self.encoder = NanoEncoder(encoder_config)
        self.decoder = NanoDecoder(decoder_config)

        # Tie embeddings if specified (saves parameters, improves performance)
        if config.tie_word_embeddings:
            self.decoder.embed_tokens = self.encoder.embed_tokens

    def get_encoder(self):
        """Get the encoder model."""
        return self.encoder

    def get_decoder(self):
        """Get the decoder model."""
        return self.decoder

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        labels=None,
        **kwargs
    ):
        """
        Forward pass through encoder-decoder model.

        Args:
            input_ids: Encoder input token ids [batch, seq_len]
            attention_mask: Encoder attention mask [batch, seq_len]
            decoder_input_ids: Decoder input token ids [batch, seq_len]
            labels: Target labels for loss computation [batch, seq_len]

        Returns:
            ModelOutput with last_hidden_state, loss, etc.
        """
        # Encode input sequence
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )

        # Decode with cross-attention to encoder outputs
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            encoder_attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )

        return decoder_outputs

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[Tuple] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        encoder_outputs: Optional[Tuple] = None,
        **kwargs,
    ):
        """Prepare inputs for generation."""
        # Get decoder input IDs (usually just the last token for generation)
        decoder_input_ids = input_ids

        return {
            "input_ids": None,  # Encoder inputs already processed
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "use_cache": use_cache,
        }

    def get_input_embeddings(self):
        """Get encoder input embeddings."""
        return self.encoder.embed_tokens

    def get_output_embeddings(self):
        """Get decoder output embeddings (LM head)."""
        return self.decoder.lm_head

    def set_input_embeddings(self, new_embeddings):
        """Set encoder input embeddings."""
        self.encoder.embed_tokens = new_embeddings

    def set_output_embeddings(self, new_embeddings):
        """Set decoder output embeddings (LM head)."""
        self.decoder.lm_head = new_embeddings

    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None, pad_to_multiple_of=None) -> nn.Embedding:
        """Resize token embeddings for both encoder and decoder."""
        # Resize encoder embeddings
        encoder_embeds = self.encoder.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)

        # Resize decoder embeddings (shared with LM head)
        decoder_embeds = self.decoder.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)

        # Ensure they are the same object (weight sharing)
        if encoder_embeds is not decoder_embeds:
            self.decoder.embed_tokens = encoder_embeds
            self.decoder.lm_head = nn.Linear(encoder_embeds.embedding_dim, encoder_embeds.num_embeddings, bias=False)
            self.decoder.lm_head.weight = encoder_embeds.weight

        return encoder_embeds

