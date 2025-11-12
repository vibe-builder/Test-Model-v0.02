"""Base classes for Nano XYZ models."""

from __future__ import annotations

from typing import Optional

import torch
from transformers import PreTrainedModel

from .configuration_nano import NanoConfig


class NanoPreTrainedModel(PreTrainedModel):
    """Base class to integrate with the HF checkpoint ecosystem."""

    config_class = NanoConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True

    def _init_weights(self, module) -> None:
        # ModelArchitecture already initializes all modules.
        return

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs: Optional[dict] = None) -> None:  # type: ignore[override]
        try:
            super().gradient_checkpointing_enable(gradient_checkpointing_kwargs)
        except TypeError:
            super().gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self) -> None:  # type: ignore[override]
        super().gradient_checkpointing_disable()
