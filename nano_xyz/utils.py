# Training Utilities
# ===================
#
# Training-related utilities including:
# - OptimizerFactory: AdamW optimizer configuration
# - PerformanceMonitor: Model FLOPs utilization calculation
# - Training helpers for FP16/BF16 mixed precision

import contextlib
import inspect
import logging
from typing import Optional, Dict, List, Tuple, Any

import torch
import torch.nn as nn

from .model import ModelArchitecture, ModelSettings

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def amp_context(config: "ModelSettings", device: torch.device):
    """Centralized context manager for automatic mixed precision.

    Args:
        config: Model configuration containing dtype settings
        device: Device to run on (determines if AMP is enabled)

    Yields:
        AMP context manager
    """
    enabled = (config.dtype in ['float16', 'bfloat16'] and device.type == 'cuda')
    dtype = None

    if enabled:
        if config.dtype == 'float16':
            dtype = torch.float16
        elif config.dtype == 'bfloat16':
            dtype = torch.bfloat16

    with torch.amp.autocast(device_type=device.type, enabled=enabled, dtype=dtype):
        yield


def create_amp_scaler(config: "ModelSettings", device: torch.device) -> Optional[torch.cuda.amp.GradScaler]:
    """Create GradScaler for mixed precision training.

    Only needed for float16 (not bfloat16 which has better numerical stability).

    Args:
        config: Model configuration
        device: Training device

    Returns:
        GradScaler if needed, None otherwise
    """
    if config.dtype == 'float16' and device.type == 'cuda':
        return torch.cuda.amp.GradScaler()
    return None


class OptimizerFactory:
    """Factory for creating optimizers and training configurations."""

    @staticmethod
    def create_optimizer(model: nn.Module,
                        weight_decay: float,
                        learning_rate: float,
                        betas: Tuple[float, float],
                        device_type: str) -> torch.optim.Optimizer:
        """Create AdamW optimizer with proper weight decay grouping.

        Args:
            model: Model to optimize
            weight_decay: Weight decay factor
            learning_rate: Learning rate
            betas: Adam beta parameters
            device_type: Device type ('cuda' or 'cpu')

        Returns:
            Configured AdamW optimizer
        """
        param_groups = OptimizerFactory._create_parameter_groups(model, weight_decay)

        # Check for fused AdamW support (available in PyTorch >= 1.12)
        try:
            # Parse torch version
            torch_version = tuple(map(int, torch.__version__.split('.')[:2]))
            fused_supported = torch_version >= (1, 12)
        except (ValueError, AttributeError):
            fused_supported = False

        # Additional check for fused parameter in signature
        fused_in_signature = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_supported and fused_in_signature and device_type == 'cuda'

        optimizer_args = {
            'lr': learning_rate,
            'betas': betas,
        }
        
        # Fix: Conditionally add fused parameter only if available
        if use_fused:
            optimizer_args['fused'] = True

        optimizer = torch.optim.AdamW(param_groups, **optimizer_args)
        logger.info(f"Created AdamW optimizer (fused={use_fused}) with {len(param_groups)} parameter groups")

        return optimizer

    @staticmethod
    def _create_parameter_groups(model: nn.Module, weight_decay: float) -> List[Dict[str, Any]]:
        """Create parameter groups for optimizer with proper weight decay.

        Args:
            model: Model to create groups for
            weight_decay: Weight decay factor

        Returns:
            List of parameter groups
        """
        param_dict = {name: param for name, param in model.named_parameters() if param.requires_grad}

        decay_params = [param for name, param in param_dict.items() if param.dim() >= 2]
        no_decay_params = [param for name, param in param_dict.items() if param.dim() < 2]

        param_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]

        decay_count = sum(p.numel() for p in decay_params)
        no_decay_count = sum(p.numel() for p in no_decay_params)

        logger.info(f"Parameter groups: {len(decay_params)} decay params ({decay_count:,}), "
                   f"{len(no_decay_params)} no-decay params ({no_decay_count:,})")

        return param_groups


class PerformanceMonitor:
    """Monitors and calculates model performance metrics."""

    @staticmethod
    def _detect_device_peak_flops(device_type: str = "auto") -> float:
        """Detect peak FLOPs for the current device.
        
        Fix: Device-aware FLOPs detection instead of hardcoded A100 value.
        
        Args:
            device_type: Device type ('auto', 'cuda', 'cpu')
            
        Returns:
            Peak FLOPs per second (defaults to A100 if cannot determine)
        """
        if device_type == "auto":
            if torch.cuda.is_available():
                device_type = "cuda"
            else:
                device_type = "cpu"
        
        if device_type == "cuda":
            try:
                props = torch.cuda.get_device_properties(0)
                device_name = props.name.lower()
                
                # Fix: Detect common GPU types and estimate peak FLOPs
                # T4: ~8.1 TFLOPS FP32, ~65 TFLOPS FP16 (Tensor Core)
                # A100: ~312 TFLOPS bfloat16 (Tensor Core)
                # V100: ~15.7 TFLOPS FP32, ~125 TFLOPS FP16
                # RTX 3090: ~35.6 TFLOPS FP32, ~285 TFLOPS FP16
                
                if "t4" in device_name or "tesla t4" in device_name:
                    # T4 peak FP16 with Tensor Cores
                    peak_flops = 65e12
                    logger.debug(f"Detected T4 GPU, using peak FLOPs: {peak_flops/1e12:.1f} TFLOPS (FP16)")
                elif "a100" in device_name:
                    peak_flops = 312e12
                    logger.debug(f"Detected A100 GPU, using peak FLOPs: {peak_flops/1e12:.1f} TFLOPS (bf16)")
                elif "v100" in device_name:
                    peak_flops = 125e12
                    logger.debug(f"Detected V100 GPU, using peak FLOPs: {peak_flops/1e12:.1f} TFLOPS (FP16)")
                elif "rtx 3090" in device_name or "3090" in device_name:
                    peak_flops = 285e12
                    logger.debug(f"Detected RTX 3090 GPU, using peak FLOPs: {peak_flops/1e12:.1f} TFLOPS (FP16)")
                else:
                    # Default: estimate based on compute capability and multiprocessors
                    compute_capability = props.major + props.minor / 10.0
                    if compute_capability >= 7.0:  # Volta/Turing/Ampere/Ada
                        # Use improved estimation: multiprocessors * 2048 * clock_rate / 1000 * 2
                        # This accounts for Tensor Core performance more accurately
                        clock_ghz = props.clock_rate / 1000.0  # Convert to GHz
                        peak_flops = props.multi_processor_count * 2048 * clock_ghz * 2
                        logger.debug(f"Detected GPU {device_name} (CC {compute_capability:.1f}), "
                                   f"estimated peak FLOPs: {peak_flops/1e12:.1f} TFLOPS (FP16)")
                    else:
                        # Older GPU, default to conservative estimate
                        peak_flops = 20e12
                        logger.debug(f"Detected older GPU {device_name}, using conservative estimate: "
                                   f"{peak_flops/1e12:.1f} TFLOPS")

                    # Try to use nvidia-smi for more accurate detection if available
                    try:
                        import subprocess
                        result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader,nounits'],
                                              capture_output=True, text=True, timeout=5)
                        if result.returncode == 0:
                            detected_name = result.stdout.strip().lower()
                            # Update peak FLOPs based on nvidia-smi detection if it differs
                            if "rtx 3060" in detected_name and peak_flops < 100e12:
                                peak_flops = 125e12  # More accurate RTX 3060 estimate
                                logger.debug(f"Updated RTX 3060 peak FLOPs to: {peak_flops/1e12:.1f} TFLOPS")
                    except (subprocess.SubprocessError, FileNotFoundError, subprocess.TimeoutExpired):
                        pass  # nvidia-smi not available, use existing estimate
                
                return peak_flops
            except Exception as e:
                logger.warning(f"Could not detect GPU properties: {e}. Using A100 default.")
                return 312e12  # Default to A100
        else:
            # CPU: very low FLOPs compared to GPU
            logger.debug("Using CPU, MFU will be very low")
            return 1e12  # 1 TFLOPS as rough CPU estimate
    
    @staticmethod
    def estimate_model_flops_utilization(model: ModelArchitecture,
                                       forward_backward_per_iter: int,
                                       iteration_time: float,
                                       peak_flops: Optional[float] = None,
                                       device_type: str = "auto") -> float:
        """Estimate model FLOPs utilization (MFU).

        Fix: Device-aware peak FLOPs detection instead of hardcoded A100 value.

        Args:
            model: Model to analyze
            forward_backward_per_iter: Number of forward+backward passes per iteration
            iteration_time: Time per iteration in seconds
            peak_flops: Optional peak FLOPs per second (auto-detected if None)
            device_type: Device type for auto-detection ('auto', 'cuda', 'cpu')

        Returns:
            MFU as a fraction of device peak performance
        """
        # Ensure model is on the correct device for accurate parameter counting
        device = torch.device(device_type if device_type != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
        model = model.to(device)

        # Create a test tensor on the same device to ensure device consistency
        t = torch.tensor([1.0], device=device)

        param_count = model.get_parameter_count(include_embeddings=False)
        config = model.config

        layer_factor = config.n_layer
        head_factor = config.n_head
        embed_factor = config.n_embd // config.n_head
        seq_factor = config.block_size

        # Base transformer FLOPs
        flops_per_token = 6 * param_count + 12 * layer_factor * head_factor * embed_factor * seq_factor

        # Add FLOPs for reasoning layers if enabled
        if config.use_lcr:
            # LCR: Local Convolutional Reasoning
            # Approximate: 2 * n_embd^2 * kernel_size per LCR layer, placed every 6 layers
            lcr_layers = max(1, layer_factor // 6)  # At least 1 LCR layer if enabled
            lcr_flops = lcr_layers * (2 * config.n_embd ** 2 * config.lcr_kernel_size)
            flops_per_token += lcr_flops

        if config.use_gtr:
            # GTR: Global Token Reasoning
            # Approximate: 12 * gtr_num_tokens * n_embd * block_size per GTR layer, placed every 3 layers
            gtr_layers = max(1, layer_factor // 3)  # At least 1 GTR layer if enabled
            gtr_flops = gtr_layers * (12 * config.gtr_num_tokens * config.n_embd * seq_factor)
            flops_per_token += gtr_flops
        flops_per_forward_backward = flops_per_token * seq_factor
        flops_per_iter = flops_per_forward_backward * forward_backward_per_iter

        flops_achieved = flops_per_iter / iteration_time

        if peak_flops is None:
            peak_flops = PerformanceMonitor._detect_device_peak_flops(device_type)

        mfu = flops_achieved / peak_flops
        return mfu


def train_step(model: ModelArchitecture, token_ids: torch.Tensor, targets: torch.Tensor,
               optimizer: torch.optim.Optimizer, scaler: Optional[torch.cuda.amp.GradScaler] = None) -> float:
    """Perform a single training step with optional mixed precision.
    
    Args:
        model: Model to train
        token_ids: Input token IDs
        targets: Target token IDs
        optimizer: Optimizer instance
        scaler: Optional GradScaler for FP16 (not needed for BF16)
    
    Returns:
        Loss value
    """
    model.train()
    optimizer.zero_grad()
    
    # Determine if we should use AMP
    use_amp = model.config.dtype in ['float16', 'bfloat16'] and token_ids.device.type == 'cuda'
    dtype = None
    if use_amp:
        if model.config.dtype == 'float16':
            dtype = torch.float16
        elif model.config.dtype == 'bfloat16':
            dtype = torch.bfloat16
    
    # Forward pass with mixed precision
    device_type = token_ids.device.type
    with torch.amp.autocast(device_type=device_type, enabled=use_amp, dtype=dtype):
        logits, loss = model(token_ids, targets=targets)
    
    # Backward pass with gradient scaling for FP16
    if use_amp and model.config.dtype == 'float16' and scaler is not None:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()
    
    return loss.item()


def inference_step(model: ModelArchitecture, token_ids: torch.Tensor) -> torch.Tensor:
    """Perform inference with optional mixed precision.
    
    Args:
        model: Model for inference
        token_ids: Input token IDs
    
    Returns:
        Logits tensor
    """
    model.eval()
    
    # Determine if we should use AMP
    use_amp = model.config.dtype in ['float16', 'bfloat16'] and token_ids.device.type == 'cuda'
    dtype = None
    if use_amp:
        if model.config.dtype == 'float16':
            dtype = torch.float16
        elif model.config.dtype == 'bfloat16':
            dtype = torch.bfloat16
    
    with torch.no_grad():
        device_type = token_ids.device.type
        with torch.amp.autocast(device_type=device_type, enabled=use_amp, dtype=dtype):
            logits, _ = model(token_ids)
    
    return logits














