# Training Script
# ===============
#
# Complete training script for the nano model with checkpointing, 
# learning rate scheduling, and gradient accumulation support.

import argparse
import logging
import time
from pathlib import Path
from typing import Optional
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR, ConstantLR

from .model import ModelArchitecture, ModelSettings
from .utils import OptimizerFactory
from .processor import TextProcessor
from .dataset import create_dataloader
from .checkpoint import CheckpointManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('nano.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def _get_memory_usage(device: torch.device) -> str:
    """Get memory usage information for logging."""
    try:
        if device.type == 'cuda' and torch.cuda.is_available():
            # GPU memory
            allocated = torch.cuda.memory_allocated(device) / 1024**3  # GB
            reserved = torch.cuda.memory_reserved(device) / 1024**3   # GB

            # Check if usage is >80% of total
            total_memory = torch.cuda.get_device_properties(device).total_memory / 1024**3
            usage_percent = (allocated / total_memory) * 100

            warning = " ⚠️ HIGH MEMORY" if usage_percent > 80 else ""
            return f", GPU Mem={allocated:.2f}/{reserved:.2f}GB ({usage_percent:.1f}%){warning}"
        else:
            # CPU memory (using psutil if available)
            try:
                import psutil
                memory = psutil.virtual_memory()
                usage_percent = memory.percent
                used_gb = memory.used / 1024**3
                total_gb = memory.total / 1024**3

                warning = " ⚠️ HIGH MEMORY" if usage_percent > 80 else ""
                return f", CPU Mem={used_gb:.2f}/{total_gb:.2f}GB ({usage_percent:.1f}%){warning}"
            except ImportError:
                return ""
    except Exception:
        return ""


def create_lr_scheduler(optimizer, num_training_steps: int, warmup_steps: int = 1000):
    """Create learning rate scheduler with warmup and cosine decay."""

    warmup_steps = max(0, min(warmup_steps, num_training_steps))
    schedulers = []
    milestones = []

    if warmup_steps > 0:
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_steps
        )
        schedulers.append(warmup_scheduler)
        milestones.append(warmup_steps)

    remaining_steps = max(0, num_training_steps - warmup_steps)

    if remaining_steps <= 1:
        logger.info(
            "Using ConstantLR scheduler (steps=%d, warmup=%d)",
            num_training_steps,
            warmup_steps
        )
        schedulers.append(ConstantLR(optimizer, factor=1.0))
    else:
        schedulers.append(
            CosineAnnealingLR(
                optimizer,
                T_max=remaining_steps,
                eta_min=1e-6
            )
        )

    if not schedulers:
        return ConstantLR(optimizer, factor=1.0)

    if len(schedulers) == 1:
        return schedulers[0]

    return SequentialLR(
        optimizer,
        schedulers=schedulers,
        milestones=milestones
    )


def accumulate_gradients(
    model: ModelArchitecture,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None
) -> float:
    """Perform forward and backward pass for gradient accumulation.
    
    This function does NOT call optimizer.zero_grad() or optimizer.step(),
    allowing gradients to accumulate across multiple batches.
    
    Args:
        model: Model to train
        input_ids: Input token IDs
        targets: Target token IDs
        scaler: Optional GradScaler for FP16
        
    Returns:
        Loss value
    """
    model.train()

    # Forward pass with centralized mixed precision
    from .utils import amp_context
    device = input_ids.device
    with amp_context(model.config, device):
        # Explicitly disable KV cache during training for memory savings
        use_cache = False  # model.training is already set by model.train() call above
        logits, loss = model(input_ids, targets=targets, attention_mask=attention_mask, use_cache=use_cache)

    # Backward pass with gradient scaling for FP16
    if model.config.dtype == 'float16' and device.type == 'cuda' and scaler is not None:
        scaler.scale(loss).backward()
    else:
        loss.backward()
    
    return loss.item()


def train(
    model: ModelArchitecture,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
    epochs: int,
    save_dir: str,
    save_every: int = 1000,
    gradient_accumulation_steps: int = 1,
    max_grad_norm: float = 1.0,
    resume_from: Optional[str] = None,
    val_loader: Optional[torch.utils.data.DataLoader] = None,
    eval_every: int = 1000,
    use_ddp: bool = False
):
    """Complete training loop.

    Args:
        model: Model to train
        train_loader: DataLoader for training data
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Training device
        epochs: Number of epochs
        save_dir: Directory to save checkpoints
        save_every: Save checkpoint every N steps
        gradient_accumulation_steps: Accumulate gradients over N steps
        max_grad_norm: Maximum gradient norm for clipping
        resume_from: Optional checkpoint path to resume from
        val_loader: Optional DataLoader for validation data
        eval_every: Run validation every N training steps
        use_ddp: Enable Distributed Data Parallel training
    """
    # Model device placement will be handled by DDP setup if enabled
    if not use_ddp:
        model = model.to(device)
    model.train()

    # Create centralized AMP scaler
    from .utils import create_amp_scaler
    scaler = create_amp_scaler(model.config, device)
    
    # Resume from checkpoint if provided
    start_epoch = 0
    global_step = 0
    if resume_from:
        logger.info(f"Resuming from checkpoint: {resume_from}")
        checkpoint = CheckpointManager.load_checkpoint(
            resume_from, model=model, optimizer=optimizer, scheduler=scheduler, device=device
        )
        start_epoch = checkpoint['epoch']
        global_step = checkpoint['step']
        logger.info(f"Resumed at epoch {start_epoch}, step {global_step}")
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Setup Distributed Data Parallel if requested
    if use_ddp:
        try:
            import torch.distributed as dist
            from torch.nn.parallel import DistributedDataParallel as DDP

            # Initialize the process group
            if not dist.is_initialized():
                # Use environment variables for distributed setup
                # RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT should be set by launcher
                dist.init_process_group(backend='nccl', init_method='env://')
                logger.info("Distributed training initialized")

            # Get local rank for this process
            local_rank = dist.get_rank()
            world_size = dist.get_world_size()

            # Set device based on local rank
            device = torch.device(f'cuda:{local_rank}')
            model = model.to(device)
            model = DDP(model, device_ids=[local_rank])

            # Create distributed sampler if train_loader uses a sampler
            if hasattr(train_loader, 'sampler') and hasattr(train_loader.sampler, '__iter__'):
                train_sampler = torch.utils.data.distributed.DistributedSampler(
                    train_loader.dataset,
                    num_replicas=world_size,
                    rank=local_rank,
                    shuffle=True
                )
                train_loader = torch.utils.data.DataLoader(
                    train_loader.dataset,
                    batch_size=train_loader.batch_size,
                    sampler=train_sampler,
                    num_workers=train_loader.num_workers,
                    pin_memory=train_loader.pin_memory,
                    drop_last=train_loader.drop_last
                )

            logger.info(f"DDP setup complete: rank {local_rank}/{world_size}, device {device}")

        except ImportError:
            logger.warning("torch.distributed not available, falling back to single GPU training")
            use_ddp = False
        except Exception as e:
            logger.warning(f"DDP initialization failed: {e}, falling back to single GPU training")
            use_ddp = False

    # Training loop
    for epoch in range(start_epoch, epochs):
        logger.info(f"Starting epoch {epoch + 1}/{epochs}")
        epoch_start_time = time.time()
        epoch_loss = 0.0
        num_batches = 0

        # Set epoch for distributed sampler
        if use_ddp and hasattr(train_loader, 'sampler') and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)

        optimizer.zero_grad()
        
        for batch_idx, (input_ids, targets, attention_mask) in enumerate(train_loader):
            input_ids = input_ids.to(device)
            targets = targets.to(device)
            attention_mask = attention_mask.to(device)

            # Forward and backward pass (gradients accumulate)
            loss = accumulate_gradients(
                model, input_ids, targets, attention_mask, scaler
            )

            epoch_loss += loss
            num_batches += 1
            
            # Gradient accumulation: only step optimizer every N steps
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                # Gradient clipping per parameter group for sensitivity handling
                if max_grad_norm > 0:
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                    # Clip gradients per parameter group instead of globally
                    for group in optimizer.param_groups:
                        torch.nn.utils.clip_grad_norm_(
                            group['params'], max_grad_norm
                        )
                
                # Optimizer step
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                
                # Logging
                if global_step % 100 == 0:
                    current_lr = scheduler.get_last_lr()[0]
                    memory_info = _get_memory_usage(device)
                    logger.info(
                        f"Step {global_step}: Loss={loss:.4f}, "
                        f"LR={current_lr:.2e}, Epoch={epoch+1}/{epochs}{memory_info}"
                    )
                
                # Validation
                if val_loader is not None and global_step % eval_every == 0:
                    model.eval()
                    val_loss = 0.0
                    val_steps = 0
                    with torch.no_grad():
                        for val_batch in val_loader:
                            input_ids, targets, attention_mask = val_batch
                            input_ids = input_ids.to(device)
                            targets = targets.to(device)
                            attention_mask = attention_mask.to(device)

                            logits, loss = model(input_ids, targets=targets, attention_mask=attention_mask)
                            val_loss += loss.item()
                            val_steps += 1
                            if val_steps >= 10:  # Limit validation steps for speed
                                break
                    avg_val_loss = val_loss / val_steps
                    logger.info(f"Step {global_step}: Validation loss={avg_val_loss:.4f}")
                    model.train()

                # Save checkpoint
                if global_step % save_every == 0:
                    checkpoint_path = CheckpointManager.save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch,
                        step=global_step,
                        loss=loss,
                        checkpoint_dir=save_dir,
                        filename=f"checkpoint_step_{global_step}.pt"
                    )
                    logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        # Handle remaining gradients if batch doesn't divide evenly
        if num_batches % gradient_accumulation_steps != 0:
            if max_grad_norm > 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                # Clip gradients per parameter group instead of globally
                for group in optimizer.param_groups:
                    torch.nn.utils.clip_grad_norm_(
                        group['params'], max_grad_norm
                    )
            
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1
        
        avg_loss = epoch_loss / num_batches
        epoch_time = time.time() - epoch_start_time
        logger.info(
            f"Epoch {epoch + 1} completed. Average loss: {avg_loss:.4f}, "
            f"Time: {epoch_time:.2f}s"
        )
        
        # Save epoch checkpoint
        CheckpointManager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch + 1,
            step=global_step,
            loss=avg_loss,
            checkpoint_dir=save_dir,
            filename=f"checkpoint_epoch_{epoch + 1}.pt"
        )


def main():
    # Set seeds for reproducibility
    import torch, random, numpy as np
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Enable TF32 for faster training on Ampere+ GPUs
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    parser = argparse.ArgumentParser(description="Train nano model")
    parser.add_argument("--data_file", type=str, required=True,
                       help="Path to training data file")
    parser.add_argument("--save_dir", type=str, default="./checkpoints",
                       help="Directory to save checkpoints")
    parser.add_argument("--block_size", type=int, default=1024,
                       help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size")
    parser.add_argument("--epochs", type=int, default=10,
                       help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=3e-4,
                       help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.1,
                       help="Weight decay")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                       help="Gradient accumulation steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                       help="Maximum gradient norm")
    parser.add_argument("--save_every", type=int, default=1000,
                       help="Save checkpoint every N steps")
    parser.add_argument("--warmup_steps", type=int, default=1000,
                       help="Number of warmup steps for learning rate")
    parser.add_argument("--resume_from", type=str, default=None,
                       help="Resume training from checkpoint")
    parser.add_argument("--device", type=str, default=None,
                       help="Device (cuda/cpu, auto-detected if None)")
    parser.add_argument("--tokenizer_name", type=str, default="gpt2",
                       help="HuggingFace tokenizer name (e.g., 'gpt2', 'google/gemma-2-2b')")
    parser.add_argument("--vocab_size", type=int, default=None,
                       help="Vocabulary size (auto-detected from tokenizer if None, deprecated)")
    parser.add_argument("--n_layer", type=int, default=12,
                       help="Number of transformer layers")
    parser.add_argument("--n_head", type=int, default=12,
                       help="Number of attention heads")
    parser.add_argument("--n_embd", type=int, default=768,
                       help="Embedding dimension")
    parser.add_argument("--dropout", type=float, default=0.0,
                       help="Dropout probability")
    parser.add_argument("--max_cache_len", type=int, default=4096,
                       help="Maximum KV cache length to retain during generation (limits memory)")

    # YaRN / RoPE scaling
    parser.add_argument("--use_yarn", dest="use_yarn", action="store_true",
                       help="Enable YaRN for extended context windows (default)")
    parser.add_argument("--no_yarn", dest="use_yarn", action="store_false",
                       help="Disable YaRN frequency scaling")
    parser.set_defaults(use_yarn=True)
    parser.add_argument("--yarn_orig_ctx", type=int, default=2048,
                       help="Original context length for YaRN scaling")
    parser.add_argument("--yarn_target_ctx", type=int, default=8192,
                       help="Target context length for YaRN scaling")
    parser.add_argument("--yarn_alpha", type=float, default=1.0,
                       help="Frequency exponent scaling for YaRN")
    parser.add_argument("--yarn_beta", type=float, default=1.0,
                       help="Magnitude adjustment for YaRN")
    parser.add_argument("--rope_base", type=float, default=10000.0,
                       help="Base for RoPE frequency computation")

    parser.add_argument("--use_activation_checkpointing", dest="use_activation_checkpointing", action="store_true",
                       help="Enable activation checkpointing for transformer layers during training")
    parser.add_argument("--no_activation_checkpointing", dest="use_activation_checkpointing", action="store_false",
                       help="Disable activation checkpointing (default)")
    parser.set_defaults(use_activation_checkpointing=False)

    # Reasoning layers
    parser.add_argument("--use_lcr", dest="use_lcr", action="store_true",
                       help="Enable Local Convolutional Reasoning block (default)")
    parser.add_argument("--no_lcr", dest="use_lcr", action="store_false",
                       help="Disable the LCR reasoning block")
    parser.set_defaults(use_lcr=True)

    parser.add_argument("--use_gtr", dest="use_gtr", action="store_true",
                       help="Enable Global Token Reasoning block (default)")
    parser.add_argument("--no_gtr", dest="use_gtr", action="store_false",
                       help="Disable the GTR reasoning block")
    parser.set_defaults(use_gtr=True)

    parser.add_argument("--lcr_kernel_size", type=int, default=7,
                       help="Kernel size for LCR depthwise convolution")
    parser.add_argument("--lcr_expand", type=int, default=2,
                       help="Channel expansion factor for LCR")
    parser.add_argument("--gtr_num_tokens", type=int, default=8,
                       help="Number of global tokens in GTR block")
    parser.add_argument("--gtr_num_heads", type=int, default=4,
                       help="(Deprecated) number of heads for GTR cross-attention")

    args = parser.parse_args()
    
    # Device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")

    if args.use_lcr and args.n_layer < 6:
        logger.info("LCR block will be auto-disabled because n_layer=%d < 6", args.n_layer)
    if args.use_gtr and args.n_layer < 6:
        logger.info("GTR block will be auto-disabled because n_layer=%d < 6", args.n_layer)
    
    # Initialize processor first to get vocab_size from tokenizer
    # Note: pad token handling is already done in TextProcessor.__init__
    processor = TextProcessor(tokenizer_name=args.tokenizer_name, vocab_size=args.vocab_size)
    logger.info(f"Tokenizer vocab_size: {processor.vocab_size}")
    
    # Create config dict from args, override vocab_size from tokenizer
    config_dict = vars(args).copy()
    config_dict['vocab_size'] = processor.vocab_size
    config_dict['dtype'] = None  # Auto-detect based on device capabilities

    # Create centralized config
    config = ModelSettings(**config_dict)

    # Additional validation for reasoning layers
    if (config.use_lcr or config.use_gtr) and config.n_layer < 6:
        logger.warning(f"LCR/GTR reasoning layers enabled but n_layer={config.n_layer} < 6. "
                      f"Reasoning layers will be automatically disabled.")
        config.use_lcr = False
        config.use_gtr = False
    
    # Model
    model = ModelArchitecture(config)
    logger.info(f"Model has {model.get_parameter_count()/1e6:.2f}M parameters")
    
    # Data
    logger.info("Creating data loader...")
    train_loader = create_dataloader(
        args.data_file,
        processor,
        args.block_size,
        args.batch_size
    )
    logger.info(f"Data loader created with {len(train_loader)} batches")

    # Guard against empty dataset
    if len(train_loader) == 0:
        raise ValueError("Empty dataset from data_file; verify file contents and ensure sufficient data for training")
    
    # Optimizer
    optimizer = OptimizerFactory.create_optimizer(
        model=model,
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,
        betas=(0.9, 0.95),
        device_type=device.type
    )
    
    # Scheduler
    num_training_steps = len(train_loader) * args.epochs
    scheduler = create_lr_scheduler(optimizer, num_training_steps, args.warmup_steps)
    logger.info(f"Training will run for {num_training_steps} steps across {args.epochs} epochs")
    
    # Train
    try:
        train(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epochs=args.epochs,
            save_dir=args.save_dir,
            save_every=args.save_every,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_grad_norm=args.max_grad_norm,
            resume_from=args.resume_from
        )
        logger.info("Training completed successfully!")
    except KeyboardInterrupt:
        logger.info("Training interrupted by user. Saving checkpoint...")
        CheckpointManager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=None,  # No scheduler available in interrupt handler
            epoch=args.epochs,  # Will be updated by train() if resuming
            step=0,  # Will be updated by train() if resuming
            loss=0.0,
            checkpoint_dir=args.save_dir,
            filename="checkpoint_interrupted.pt"
        )
        logger.info("Checkpoint saved. You can resume with --resume_from")
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()

