# Text Processing Utilities
# ==========================
#
# Text processing utilities for tokenization.

import logging
import warnings
from typing import Optional, List, Dict

import torch

from .model import DEFAULT_VOCAB_SIZE

logger = logging.getLogger(__name__)


class TextProcessor:
    """Text processing utilities for tokenization using HuggingFace tokenizers."""

    def __init__(
        self,
        vocab_size: Optional[int] = None,
        tokenizer_name: str = "gpt2"
    ) -> None:
        """Initialize text processor with HuggingFace tokenizer.

        Args:
            vocab_size: Optional vocabulary size for validation (auto-detected from tokenizer if None)
            tokenizer_name: HuggingFace tokenizer name (e.g., "gpt2", "google/gemma-2-2b")

        Raises:
            ImportError: If transformers library is not installed
            ValueError: If vocab_size is provided and doesn't match tokenizer vocab_size
        """
        # Load HuggingFace tokenizer
        try:
            from transformers import AutoTokenizer  # type: ignore[import-untyped]
        except ImportError:
            raise ImportError(
                "transformers library is required for tokenization. "
                "Install with: pip install transformers"
            )

        # Load tokenizer with error handling for connection issues
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        except Exception as e:
            if "ConnectionError" in str(type(e)) or "timeout" in str(e).lower():
                logger.warning(
                    f"Failed to load tokenizer '{tokenizer_name}' due to network issues. "
                    f"Using offline fallback if available, otherwise this will fail. Error: {e}"
                )
                # Re-raise to let user know about the connection issue
                raise RuntimeError(
                    f"Cannot load tokenizer '{tokenizer_name}' due to network connectivity issues. "
                    f"Please check your internet connection or download the tokenizer manually. "
                    f"Original error: {e}"
                ) from e
            else:
                # Re-raise other tokenizer loading errors
                raise

        self.tokenizer_name = tokenizer_name

        # Auto-detect vocab_size from tokenizer
        self.vocab_size = len(self.tokenizer)

        # Validate vocab_size if provided
        if vocab_size is not None:
            if vocab_size <= 0:
                raise ValueError(f"vocab_size must be positive, got {vocab_size}")
            if vocab_size != self.vocab_size:
                warnings.warn(
                    (
                        f"Provided vocab_size ({vocab_size}) doesn't match tokenizer vocab_size "
                        f"({self.vocab_size}). Using tokenizer value instead."
                    ),
                    UserWarning,
                )
                logger.warning(
                    "Overriding provided vocab_size=%d with tokenizer vocab_size=%d",
                    vocab_size,
                    self.vocab_size,
                )

        # Add pad token handling for tokenizers like GPT-2 that don't have one
        if self.tokenizer.pad_token is None:
            pad_token_id = 0
            self.tokenizer.pad_token_id = pad_token_id
            try:
                self.tokenizer.pad_token = self.tokenizer.convert_ids_to_tokens(pad_token_id)
            except Exception:
                self.tokenizer.pad_token = "<pad>"

        # Store special token IDs with safe defaults
        self.pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0

        # Default eos_token_id to vocab_size - 1 (following HF GPT2Tokenizer convention)
        if self.tokenizer.eos_token_id is not None:
            self.eos_token_id = self.tokenizer.eos_token_id
        else:
            self.eos_token_id = self.vocab_size - 1
            logger.info(f"No EOS token found in tokenizer, defaulting to {self.eos_token_id}")

        # Other special tokens (can remain None if not present)
        self.bos_token_id = self.tokenizer.bos_token_id
        self.unk_token_id = self.tokenizer.unk_token_id
        
        logger.info(
            f"Using HuggingFace tokenizer '{tokenizer_name}' "
            f"(vocab_size={self.vocab_size}, "
            f"pad_token_id={self.pad_token_id}, "
            f"eos_token_id={self.eos_token_id})"
        )

    def encode(self, text: str) -> List[int]:
        """Encode text into token IDs.

        Args:
            text: Input text to tokenize

        Returns:
            List of token IDs
        """
        tokens = self.tokenizer.encode(text, return_tensors=None, add_special_tokens=False)

        # Clamp tokens to valid range and count violations (defensive programming)
        clamped_tokens = []
        clamped_count = 0

        for t in tokens:
            original_t = t
            clamped_t = min(max(0, t), self.vocab_size - 1)
            clamped_tokens.append(clamped_t)
            if clamped_t != original_t:
                clamped_count += 1

        # Warn if significant token clamping occurs
        if clamped_count > 0:
            clamped_percentage = (clamped_count / len(tokens)) * 100
            message = (
                f"Clamped {clamped_count} tokens ({clamped_percentage:.1f}%) to valid range "
                f"[0, {self.vocab_size-1}]"
            )
            warnings.warn(message, UserWarning)
            logger.warning(message)

            # Raise error if excessive clamping (>5%)
            if clamped_percentage > 5.0:
                raise ValueError(
                    f"Excessive token clamping detected: {clamped_count}/{len(tokens)} tokens "
                    f"({clamped_percentage:.1f}%) were outside valid range. "
                    f"This suggests a tokenizer mismatch or corrupted input data."
                )

        return clamped_tokens

    def decode(self, token_ids: List[int], skip_special_tokens: bool = False) -> str:
        """Decode token IDs back to text.

        Args:
            token_ids: List of token IDs (will be clamped to valid range)
            skip_special_tokens: Whether to skip special tokens during decoding

        Returns:
            Decoded text
        """
        # Clamp tokens to valid range and track adjustments
        valid_tokens: List[int] = []
        clamped_count = 0
        for t in token_ids:
            clamped_t = min(max(0, t), self.vocab_size - 1)
            if clamped_t != t:
                clamped_count += 1
            valid_tokens.append(clamped_t)

        if clamped_count > 0 and token_ids:
            clamped_percentage = (clamped_count / len(token_ids)) * 100
            message = (
                f"Clamped {clamped_count} tokens ({clamped_percentage:.1f}%) to valid range "
                f"[0, {self.vocab_size-1}]"
            )
            warnings.warn(message, UserWarning)
            logger.warning(message)

        return self.tokenizer.decode(valid_tokens, skip_special_tokens=skip_special_tokens)
    
    def get_special_token_ids(self) -> Dict[str, Optional[int]]:
        """Get special token IDs.

        Returns:
            Dictionary with keys: pad_token_id, eos_token_id, bos_token_id, unk_token_id
        """
        return {
            "pad_token_id": self.pad_token_id,
            "eos_token_id": self.eos_token_id,
            "bos_token_id": self.bos_token_id,
            "unk_token_id": self.unk_token_id,
        }

    def preprocess_text(self, text: str, lowercase: bool = False, clean_text: bool = False) -> str:
        """Preprocess text with optional cleaning.

        Args:
            text: Raw input text
            lowercase: Whether to convert to lowercase
            clean_text: Whether to remove non-ASCII characters and keep only alphanumeric + spaces

        Returns:
            Preprocessed text
        """
        processed = text.strip()

        if lowercase:
            processed = processed.lower()

        if clean_text:
            import re
            processed = re.sub(r'[^a-zA-Z0-9\s]', '', processed)

        return processed

    def create_batches(self, texts: List[str], batch_size: int,
                      max_length: Optional[int] = None, truncate: bool = True,
                      lowercase: bool = False, clean_text: bool = False,
                      shuffle: bool = False) -> List[torch.Tensor]:
        """Create batched token tensors from texts.

        Args:
            texts: List of input texts
            batch_size: Size of each batch
            max_length: Maximum sequence length (pad/truncate to this)
            truncate: Whether to truncate sequences longer than max_length
            lowercase: Whether to convert texts to lowercase
            clean_text: Whether to clean non-ASCII characters from texts
            shuffle: Whether to shuffle the input texts before batching

        Returns:
            List of batched token tensors
        """
        all_tokens = []

        for text in texts:
            processed = self.preprocess_text(text, lowercase=lowercase, clean_text=clean_text)
            tokens = self.encode(processed)

            if max_length:
                if truncate:
                    tokens = tokens[:max_length]
                else:
                    # For training, we typically want to truncate, but keep this option
                    tokens = tokens[:max_length]

                # Always pad to max_length for consistent batching
                while len(tokens) < max_length:
                    tokens.append(self.pad_token_id)

            all_tokens.append(tokens)

        # Shuffle if requested
        if shuffle:
            import random
            random.shuffle(all_tokens)

        batches = []
        for i in range(0, len(all_tokens), batch_size):
            batch_tokens = all_tokens[i:i + batch_size]
            max_batch_len = max(len(tokens) for tokens in batch_tokens)

            # For training, limit batch length to max_length (typically block_size)
            if max_length:
                max_batch_len = min(max_batch_len, max_length)

            padded_batch = []
            for tokens in batch_tokens:
                if truncate and len(tokens) > max_batch_len:
                    tokens = tokens[:max_batch_len]
                padded = tokens + [self.pad_token_id] * (max_batch_len - len(tokens))
                padded_batch.append(padded)

            batch_tensor = torch.tensor(padded_batch, dtype=torch.long)
            batches.append(batch_tensor)

        return batches














