# Checkpoint Management
# =====================
#
# Utilities for saving and loading model checkpoints during training.
#
# Components:
# - CheckpointManager: Handles checkpoint save/load operations

import torch
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any

from .model import ModelArchitecture, ModelSettings

logger = logging.getLogger(__name__)

# Optional safetensors support
try:
    from safetensors.torch import save_file, load_file
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False

class CheckpointManager:
    """Manages model checkpoint saving and loading."""
    
    @staticmethod
    def save_checkpoint(
        model: ModelArchitecture,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        epoch: int = 0,
        step: int = 0,
        loss: float = 0.0,
        checkpoint_dir: str = "./checkpoints",
        filename: Optional[str] = None,
        extra_state: Optional[Dict[str, Any]] = None,
        use_safetensors: bool = True
    ) -> str:
        """Save a training checkpoint.

        Args:
            model: Model to save
            optimizer: Optimizer state to save
            scheduler: Optional learning rate scheduler to save state
            epoch: Current epoch number
            step: Current step number
            loss: Current loss value
            checkpoint_dir: Directory to save checkpoint
            filename: Optional filename (auto-generated if None)
            extra_state: Optional additional state to save
            use_safetensors: Whether to use safetensors format for model weights (default: True)

        Returns:
            Path to saved checkpoint
        """
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        if filename is None:
            filename = f"checkpoint_epoch_{epoch}_step_{step}.pt"
        
        # Initialize checkpoint data
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
            'loss': loss,
            'config': {
                'config_version': model.config.config_version,
                'block_size': model.config.block_size,
                'vocab_size': model.config.vocab_size,
                'n_layer': model.config.n_layer,
                'n_head': model.config.n_head,
                'n_embd': model.config.n_embd,
                'dropout': model.config.dropout,
                'bias': model.config.bias,
                'n_kv_groups': model.config.n_kv_groups,
                'sliding_window': model.config.sliding_window,
                'dtype': model.config.dtype,
                'use_fused_attention': model.config.use_fused_attention,
                'attn_logit_softcapping': model.config.attn_logit_softcapping,
                'use_fp32_softmax': model.config.use_fp32_softmax,
                'use_yarn': model.config.use_yarn,
                'yarn_orig_ctx': model.config.yarn_orig_ctx,
                'yarn_target_ctx': model.config.yarn_target_ctx,
                'yarn_alpha': model.config.yarn_alpha,
                'yarn_beta': model.config.yarn_beta,
                'rope_base': model.config.rope_base,
                'use_lcr': model.config.use_lcr,
                'use_gtr': model.config.use_gtr,
                'lcr_kernel_size': model.config.lcr_kernel_size,
                'lcr_expand': model.config.lcr_expand,
                'gtr_num_tokens': model.config.gtr_num_tokens,
            },
        }

        # Determine file extension and save model weights separately if using safetensors
        if use_safetensors and SAFETENSORS_AVAILABLE:
            model_filename = filename.replace('.pt', '.safetensors')
            checkpoint_filename = filename.replace('.pt', '_meta.pt')
            model_path = checkpoint_dir / model_filename
            checkpoint_path = checkpoint_dir / checkpoint_filename

            # Save model weights with safetensors
            # Handle tied weights by creating a copy of the state dict
            state_dict = model.state_dict()
            # Safetensors doesn't like shared tensors, so we need to make copies
            # For tied embeddings, we save them separately
            if 'language_model_head.weight' in state_dict and 'transformer.token_embeddings.weight' in state_dict:
                # Check if they are the same tensor (tied)
                if id(state_dict['language_model_head.weight']) == id(state_dict['transformer.token_embeddings.weight']):
                    # Clone the tied weight to avoid shared tensor issues in safetensors
                    state_dict = dict(state_dict)  # Make a shallow copy
                    state_dict['language_model_head.weight'] = state_dict['language_model_head.weight'].clone()

            try:
                save_file(state_dict, model_path)
                checkpoint['model_weights_file'] = model_filename
            except RuntimeError as e:
                if "share memory" in str(e):
                    # Fall back to regular PyTorch saving for models with tied weights
                    logger.warning("Safetensors doesn't support tied weights, falling back to PyTorch format")
                    checkpoint['model_state_dict'] = state_dict
                    checkpoint_filename = filename
                    checkpoint_path = checkpoint_dir / checkpoint_filename
                else:
                    raise
        else:
            checkpoint['model_state_dict'] = model.state_dict()
            checkpoint_filename = filename
            checkpoint_path = checkpoint_dir / checkpoint_filename

        if extra_state:
            checkpoint['extra_state'] = extra_state

        torch.save(checkpoint, checkpoint_path)
        
        # Also save config as JSON for easy inspection
        # Use checkpoint_filename (without extension) to match actual checkpoint file
        config_path = checkpoint_dir / f"{Path(checkpoint_filename).stem}.config.json"
        with open(config_path, 'w') as f:
            json.dump(checkpoint['config'], f, indent=2)
        
        return str(checkpoint_path)
    
    @staticmethod
    def load_checkpoint(
        checkpoint_path: str,
        model: Optional[ModelArchitecture] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: Optional[torch.device] = None,
        map_location: Optional[str] = None
    ) -> Dict[str, Any]:
        """Load a training checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            model: Optional model to load state into
            optimizer: Optional optimizer to load state into
            scheduler: Optional scheduler to load state into
            device: Device to load checkpoint on
            map_location: Optional map_location for torch.load (overrides device)

        Returns:
            Dictionary containing checkpoint data
        """
        # Determine map_location
        if map_location is None and device is not None:
            map_location = device
        elif map_location is None:
            map_location = 'cpu'

        checkpoint = torch.load(checkpoint_path, map_location=map_location)

        if model is not None:
            if checkpoint.get('model_weights_file') and SAFETENSORS_AVAILABLE:
                # Load from safetensors file
                checkpoint_dir = Path(checkpoint_path).parent
                weights_path = checkpoint_dir / checkpoint['model_weights_file']
                # Handle device/map_location conversion safely
                device_str = str(map_location) if isinstance(map_location, torch.device) else map_location
                state_dict = load_file(weights_path, device=device_str)
                model.load_state_dict(state_dict)
            elif 'model_state_dict' in checkpoint:
                # Load from regular PyTorch checkpoint
                model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if scheduler is not None and checkpoint.get('scheduler_state_dict') is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        return checkpoint
    
    @staticmethod
    def load_config_from_checkpoint(checkpoint_path: str, map_location: str = 'cpu') -> ModelSettings:
        """Load model configuration from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            map_location: Device to load checkpoint on

        Returns:
            ModelSettings object
        """
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        config_dict = checkpoint['config']

        # Validate required config fields
        required_fields = {
            'block_size', 'vocab_size', 'n_layer', 'n_head', 'n_embd'
        }
        missing = sorted(field for field in required_fields if field not in config_dict)
        if missing:
            raise ValueError(f"Checkpoint config missing required fields: {', '.join(missing)}")

        # Validate config values make sense
        if config_dict.get('block_size', 0) <= 0:
            raise ValueError(f"Invalid block_size in checkpoint: {config_dict.get('block_size')}")
        if config_dict.get('vocab_size', 0) <= 0:
            raise ValueError(f"Invalid vocab_size in checkpoint: {config_dict.get('vocab_size')}")
        if config_dict.get('n_layer', 0) <= 0:
            raise ValueError(f"Invalid n_layer in checkpoint: {config_dict.get('n_layer')}")
        if config_dict.get('n_head', 0) <= 0:
            raise ValueError(f"Invalid n_head in checkpoint: {config_dict.get('n_head')}")
        if config_dict.get('n_embd', 0) <= 0:
            raise ValueError(f"Invalid n_embd in checkpoint: {config_dict.get('n_embd')}")

        # Warn about missing optional fields but don't fail
        optional_fields = {
            'config_version', 'use_yarn', 'yarn_orig_ctx', 'yarn_target_ctx', 'yarn_alpha', 'yarn_beta',
            'rope_base', 'use_lcr', 'use_gtr', 'lcr_kernel_size', 'lcr_expand', 'gtr_num_tokens'
        }
        missing_optional = sorted(field for field in optional_fields if field not in config_dict)
        if missing_optional:
            logger.warning(
                "Checkpoint config missing optional fields %s; using ModelSettings defaults",
                ", ".join(missing_optional)
            )

        return ModelSettings(**config_dict)
