# Text Generation
# ===============
#
# High-level text generation interface.

import logging
from typing import Optional, Tuple, Union, List

import torch
from torch.nn import functional as F

from .model import ModelArchitecture, ModelSettings, MIN_TEMPERATURE, MAX_TEMPERATURE
from .utils import OptimizerFactory, PerformanceMonitor, amp_context

logger = logging.getLogger(__name__)


class TextGenerator:
    """Text generation system coordinating all components."""

    def __init__(
        self, 
        config: Optional[ModelSettings] = None, 
        model: Optional[ModelArchitecture] = None,
        device: Optional[Union[str, torch.device]] = None,
        processor: Optional['TextProcessor'] = None  # type: ignore[name-defined]
    ) -> None:
        """Initialize the text generator.

        Args:
            config: Model configuration (ignored if model provided)
            model: Pre-initialized model (optional)
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
                   Fix: Adds device fallback for Colab compatibility
            processor: Optional TextProcessor instance for EOS token handling
        """
        if model is not None:
            self.model = model
            self.config = model.config
        else:
            self.config = config or ModelSettings()
            self.model = ModelArchitecture(self.config)
        
        # Fix: Device detection and fallback for Colab
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self.model = self.model.to(self.device)
        self.processor = processor

        if self.device.type == "cuda" and torch.cuda.is_available():
            current_dtype = getattr(self.config, "dtype", None)
            if current_dtype in (None, 'float32'):
                try:
                    if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
                        logger.info("Enabling bfloat16 inference for TextGenerator to reduce memory usage")
                        self.config.dtype = 'bfloat16'
                    else:
                        logger.info("Using float16 inference for TextGenerator on CUDA device")
                        self.config.dtype = 'float16' if current_dtype is None else current_dtype
                except RuntimeError:
                    logger.warning("Could not query CUDA bf16 support; falling back to float16 inference")
                    self.config.dtype = 'float16' if current_dtype is None else current_dtype

        # Set default stop token from processor if available
        self.default_stop_token = None
        if processor is not None and hasattr(processor, 'eos_token_id') and processor.eos_token_id is not None:
            self.default_stop_token = processor.eos_token_id

        logger.info(f"Model initialized on device: {self.device}")

    def generate(self, token_ids: torch.Tensor,
                max_new_tokens: int,
                temperature: float = 1.0,
                top_k: Optional[int] = None,
                top_p: Optional[float] = None,
                stop_token: Optional[int] = None) -> torch.Tensor:
        """Generate text from input token sequence.

        Args:
            token_ids: Input token sequence of shape (batch_size, seq_len)
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature (controls randomness, clipped to [0.1, 5.0])
            top_k: Limit sampling to top-k most likely tokens (None = no limit)
            top_p: Nucleus sampling probability mass (None = no nucleus sampling)
            stop_token: Optional token ID to stop generation early (e.g., EOS token)

        Returns:
            Generated token sequence of shape (batch_size, seq_len + actual_new_tokens)

        Raises:
            ValueError: If temperature, top_k, or top_p are invalid
        """
        if token_ids.dim() != 2:
            raise ValueError(f"token_ids must be 2D tensor, got shape {token_ids.shape}")

        # Handle empty input
        if token_ids.size(1) == 0:
            logger.warning("Empty token_ids provided, returning empty tensor")
            return torch.empty_like(token_ids)

        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")

        if top_k is not None and top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")

        if top_p is not None and (top_p <= 0 or top_p > 1):
            raise ValueError(f"top_p must be in (0, 1], got {top_p}")

        # Fix: Explicitly move input to correct device
        token_ids = token_ids.to(self.device)

        # Determine maximum context length (respect YaRN extension and cache limits)
        max_context = self.config.yarn_target_ctx if self.config.use_yarn else self.config.block_size
        max_cache_len = getattr(self.config, "max_cache_len", None)
        if max_cache_len:
            max_context = max(max_context, max_cache_len)

        full_input = token_ids
        if full_input.size(1) > max_context:
            logger.warning(
                "Input sequence length %d exceeds max context %d, truncating to recent tokens",
                full_input.size(1),
                max_context,
            )
            full_input = full_input[:, -max_context:]

        prefill_limit = min(self.config.block_size, full_input.size(1))
        prefill_context = full_input[:, -prefill_limit:]
        overflow_tokens = full_input[:, :-prefill_limit]
        token_ids = prefill_context.clone()

        # Use processor's EOS token if stop_token is None and processor is available
        if stop_token is None:
            if self.processor is not None and self.processor.eos_token_id is not None:
                stop_token = self.processor.eos_token_id
            elif self.default_stop_token is not None:
                stop_token = self.default_stop_token
            else:
                stop_token = 0  # Fallback to token 0

        # Fix: Clip temperature to valid range
        temperature = max(MIN_TEMPERATURE, min(temperature, MAX_TEMPERATURE))

        self.model.eval()

        cuda_enabled = self.device.type == "cuda" and torch.cuda.is_available()
        if cuda_enabled:
            torch.cuda.reset_peak_memory_stats(self.device)

        # Early return if no tokens to generate
        if max_new_tokens <= 0:
            return token_ids

        # Initialize KV cache for efficient generation
        past_key_values = None
        use_cache = True

        with torch.no_grad():
            attention_mask = torch.ones(
                token_ids.shape[0],
                token_ids.size(1),
                dtype=torch.long,
                device=self.device,
            )

            from .utils import amp_context
            with amp_context(self.config, self.device):
                position_ids = torch.arange(0, token_ids.size(1), device=self.device).unsqueeze(0)
                result = self.model(
                    token_ids,
                    attention_mask=attention_mask,
                    use_cache=use_cache,
                    position_ids=position_ids,
                    return_full_logits=False,
                )
                if len(result) == 3:
                    current_logits, _, past_key_values = result
                else:
                    current_logits, _ = result
                    past_key_values = None
                    use_cache = False

            forced_queue = list(torch.split(overflow_tokens, 1, dim=1)) if overflow_tokens.numel() > 0 else []
            current_pos = token_ids.size(1)
            step_index = 0
            generated = 0

            def advance(next_token: torch.Tensor, step_idx: int, forced: bool) -> Optional[torch.Tensor]:
                nonlocal past_key_values, token_ids, current_pos, use_cache
                try:
                    with amp_context(self.config, self.device):
                        position_ids_inner = torch.tensor([[current_pos]], device=self.device)
                        result_inner = self.model(
                            next_token,
                            use_cache=use_cache,
                            position_ids=position_ids_inner,
                            past_key_values=past_key_values,
                            return_full_logits=False,
                        )
                        if len(result_inner) == 3:
                            logits_step, _, past_key_values_inner = result_inner
                        else:
                            logits_step, _ = result_inner
                            past_key_values_inner = None
                    past_key_values = past_key_values_inner
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    logger.warning("OOM detected during generation; cleared cache and skipping this step")
                    return None

                token_ids = torch.cat((token_ids, next_token), dim=1)
                current_pos += next_token.size(1)
                if token_ids.size(1) > max_context:
                    token_ids = token_ids[:, -max_context:]
                if past_key_values is not None and len(past_key_values) > 0 and max_cache_len:
                    cache_len = past_key_values[0][0].shape[2]
                    if cache_len > max_cache_len:
                        slice_len = cache_len - max_cache_len
                        past_key_values = [
                            (k[..., -max_cache_len:, :], v[..., -max_cache_len:, :])
                            for k, v in past_key_values
                        ]
                        current_pos = max(0, current_pos - slice_len)

                if past_key_values is not None:
                    past_key_values = [
                        (k[..., -max_context:, :], v[..., -max_context:, :])
                        for k, v in past_key_values
                    ]

                current_pos = min(current_pos, token_ids.size(1))

                if logger.isEnabledFor(logging.DEBUG):
                    cache_lengths = [kv[0].shape[2] for kv in past_key_values] if past_key_values else []
                    logger.debug(
                        "Step %d (%s): tokens=%d cache=%s",
                        step_idx,
                        "forced" if forced else "sampled",
                        token_ids.size(1),
                        cache_lengths,
                    )

                return logits_step

            while forced_queue:
                next_forced = forced_queue.pop(0)
                current_logits = advance(next_forced, step_index, forced=True)
                if current_logits is None:
                    break
                step_index += 1
            else:
                while generated < max_new_tokens:
                    logits_step = current_logits[:, -1, :] / temperature
                    logits_step = torch.clamp(logits_step, min=-1e4, max=1e4)

                    if top_k is not None:
                        top_k_logits, _ = torch.topk(logits_step, min(top_k, logits_step.size(-1)))
                        logits_step = logits_step.masked_fill(logits_step < top_k_logits[:, -1:], float('-inf'))

                    if top_p is not None:
                        logits_step = self._apply_top_p_sampling(logits_step, top_p)

                    probs = F.softmax(logits_step, dim=-1)
                    if torch.any(torch.isnan(probs)) or torch.any(torch.isinf(probs)) or torch.any(probs < 0):
                        logger.warning("Invalid probabilities detected during decoding; using uniform sampling")
                        probs = torch.ones_like(probs) / probs.size(-1)

                    next_token = torch.multinomial(probs, num_samples=1)
                    current_logits = advance(next_token, step_index, forced=False)
                    if current_logits is None:
                        break

                    generated += 1
                    step_index += 1

                    if stop_token is not None and (next_token == stop_token).any():
                        logger.debug(
                            "Stop token %d generated at step %d, stopping early",
                            stop_token,
                            step_index,
                        )
                        break

        # Log memory usage after generation
        try:
            if self.device.type == 'cuda' and torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(self.device) / 1024**3  # GB
                total_memory = torch.cuda.get_device_properties(self.device).total_memory / 1024**3
                usage_percent = (allocated / total_memory) * 100

                if usage_percent > 80:
                    logger.warning(f"High GPU memory usage during generation: {allocated:.2f}/{total_memory:.2f}GB ({usage_percent:.1f}%)")
                else:
                    logger.debug(f"GPU memory usage: {allocated:.2f}/{total_memory:.2f}GB ({usage_percent:.1f}%)")
        except Exception:
            pass  # Silently ignore memory monitoring errors

        if cuda_enabled:
            peak_mem = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
            logger.debug("Generation peak memory: %.2f MB", peak_mem)

        return token_ids

    def generate_batch(self, token_ids_list: List[torch.Tensor],
                      max_new_tokens: int,
                      temperature: float = 1.0,
                      top_k: Optional[int] = None,
                      top_p: Optional[float] = None,
                      stop_tokens: Optional[List[Optional[int]]] = None,
                      pad_token_id: int = 0) -> List[torch.Tensor]:
        """Generate text for multiple prompts with potentially different stop tokens.

        Args:
            token_ids_list: List of input token sequences, each of shape (seq_len,)
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p sampling parameter
            stop_tokens: List of stop tokens, one per sequence (None for no stop)
            pad_token_id: Token ID to use for padding

        Returns:
            List of generated token sequences
        """
        if not token_ids_list:
            return []

        batch_size = len(token_ids_list)

        # Prepare stop tokens
        if stop_tokens is None:
            stop_tokens = [None] * batch_size
        elif len(stop_tokens) != batch_size:
            raise ValueError(f"stop_tokens length ({len(stop_tokens)}) must match batch size ({batch_size})")

        # Find maximum sequence length for padding
        max_input_len = max(seq.size(0) for seq in token_ids_list)

        # Pad sequences to same length
        padded_inputs = []
        attention_masks = []
        for seq in token_ids_list:
            pad_len = max_input_len - seq.size(0)
            if pad_len > 0:
                padded_seq = torch.cat([seq, torch.full((pad_len,), pad_token_id, dtype=seq.dtype, device=seq.device)], dim=0)
                attention_mask = torch.cat([torch.ones(seq.size(0)), torch.zeros(pad_len)], dim=0)
            else:
                padded_seq = seq
                attention_mask = torch.ones(seq.size(0))

            padded_inputs.append(padded_seq)
            attention_masks.append(attention_mask)

        # Stack into batch
        batch_input_ids = torch.stack(padded_inputs, dim=0).to(self.device)
        batch_attention_mask = torch.stack(attention_masks, dim=0).to(self.device)

        # Generate using the main generate method
        # Note: This still uses the early-stopping behavior of the main method
        # For truly independent generation, we'd need more complex batch handling
        generated_batch = self.generate(
            batch_input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            stop_token=None,  # Disable global stop token
            attention_mask=batch_attention_mask
        )

        # Split back into individual sequences
        results = []
        for i in range(batch_size):
            seq = generated_batch[i]
            # Remove padding tokens that were added
            if i < len(token_ids_list):
                orig_len = token_ids_list[i].size(0)
                seq = seq[orig_len:]  # Remove original input
                # Remove padding from generated part if present
                if pad_token_id in seq:
                    pad_idx = (seq == pad_token_id).nonzero(as_tuple=True)[0]
                    if len(pad_idx) > 0:
                        seq = seq[:pad_idx[0]]

            results.append(seq)

        return results

    def _apply_top_p_sampling(self, logits: torch.Tensor, top_p: float) -> torch.Tensor:
        """Apply nucleus (top-p) sampling to logits.
        
        Fixed: Proper implementation with numerical stability using scatter-based approach.

        Args:
            logits: Logits tensor of shape (batch_size, vocab_size)
            top_p: Probability mass to keep (nucleus)

        Returns:
            Modified logits with low-probability tokens masked to -inf
        """
        # Sort logits in descending order
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        
        # Compute softmax probabilities on sorted logits (with small epsilon for stability)
        sorted_probs = F.softmax(sorted_logits + 1e-10, dim=-1)

        # Compute cumulative probabilities with epsilon for numerical stability
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1) + 1e-6
        
        # Remove tokens with cumulative probability above the threshold
        # Keep at least one token (the first one)
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift right to keep first token above threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        
        # Create a mask for the original indices using scatter
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        
        # Set logits to -inf for removed tokens
        logits = logits.masked_fill(indices_to_remove, float('-inf'))

        # Ensure at least one token per batch has finite probability (min_keep=1)
        # This handles edge cases where all tokens might be masked
        has_finite = torch.isfinite(logits).any(dim=-1, keepdim=True)
        if not has_finite.all():
            # For batches with no finite tokens, keep the highest probability token
            max_logits = torch.max(logits, dim=-1, keepdim=True)[0]
            logits = torch.where(has_finite, logits, max_logits)

        return logits

    def configure_optimizer(self, weight_decay: float,
                           learning_rate: float,
                           betas: Tuple[float, float],
                           device_type: str) -> torch.optim.Optimizer:
        """Configure optimizer for training.

        Args:
            weight_decay: Weight decay factor
            learning_rate: Learning rate
            betas: Adam beta parameters
            device_type: Device type

        Returns:
            Configured optimizer
        """
        return OptimizerFactory.create_optimizer(
            self.model, weight_decay, learning_rate, betas, device_type
        )

    def estimate_performance(self, forward_backward_per_iter: int, iteration_time: float,
                             peak_flops: Optional[float] = None, device_type: str = "auto") -> float:
        """Estimate model performance (MFU).

        Args:
            forward_backward_per_iter: Forward+backward passes per iteration
            iteration_time: Time per iteration in seconds
            peak_flops: Optional peak FLOPs per second (auto-detected if None)
            device_type: Device type for auto-detection ('auto', 'cuda', 'cpu')

        Returns:
            Model FLOPs utilization as fraction of device peak performance
        """
        return PerformanceMonitor.estimate_model_flops_utilization(
            self.model, forward_backward_per_iter, iteration_time, peak_flops, device_type
        )

    def crop_context_window(self, new_block_size: int) -> None:
        """Reduce the model's maximum context window.

        Args:
            new_block_size: New maximum sequence length
        """
        self.model.crop_context_window(new_block_size)
        self.config.block_size = new_block_size

    def get_parameter_count(self, include_embeddings: bool = True) -> int:
        """Get the number of parameters in the model.

        Args:
            include_embeddings: Whether to include token embeddings

        Returns:
            Number of parameters
        """
        return self.model.get_parameter_count(include_embeddings)


# Alias for backward compatibility
GPT = TextGenerator














