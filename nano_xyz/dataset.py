# Dataset Loading
# ===============
#
# Dataset and DataLoader utilities for loading and processing text files for training.
#
# Components:
# - TextFileDataset: Dataset class for text files
# - create_dataloader: Factory function for creating DataLoaders

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional, Tuple
import logging

from .processor import TextProcessor

logger = logging.getLogger(__name__)


class TextFileDataset(Dataset):
    """Dataset for loading and chunking large text files with streaming support."""

    def __init__(
        self,
        file_path: str,
        processor: TextProcessor,
        block_size: int,
        chunk_size: Optional[int] = None,
        overlap: int = 0,
        streaming: bool = False,
        stream_chunk_size: int = 100 * 1024 * 1024  # 100MB default
    ):
        """Initialize dataset from text file or directory.

        Args:
            file_path: Path to text file or directory containing text files
            processor: TextProcessor instance
            block_size: Maximum sequence length
            chunk_size: Size of chunks to read per file (None = read all)
            overlap: Number of tokens to overlap between chunks
            streaming: Whether to use streaming processing for large files
            stream_chunk_size: Size of text chunks to process at once (in bytes)
        """
        self.processor = processor
        self.block_size = block_size
        self.overlap = overlap
        self.streaming = streaming
        self.stream_chunk_size = stream_chunk_size

        import os
        import time

        if os.path.isdir(file_path):
            # Directory mode - still load all at once for simplicity
            print(f"Loading and tokenizing directory {file_path}...")
            start_time = time.time()

            text = self._load_text(file_path, chunk_size)
            load_time = time.time() - start_time
            print(".2f")

            tokenize_start = time.time()
            self.tokens = self.processor.encode(text)
            tokenize_time = time.time() - tokenize_start
            print(f"Tokenized {len(self.tokens)} tokens in {tokenize_time:.2f}s "
                  f"({len(self.tokens)/tokenize_time:.0f} tokens/sec)")

            chunk_start = time.time()
            self.chunks = self._create_chunks()
            chunk_time = time.time() - chunk_start
            print(f"Created {len(self.chunks)} chunks of size {block_size} in {chunk_time:.2f}s")

            total_time = time.time() - start_time
            print(".2f")
        else:
            # Single file mode
            file_size = os.path.getsize(file_path)

            # Auto-enable streaming for very large files
            if file_size > 500 * 1024 * 1024:  # > 500MB
                self.streaming = True
                print(".2f")
            elif streaming:
                print(f"Using streaming mode for {file_path}")

            if self.streaming:
                # Streaming mode: prepare file for chunked reading
                print(f"Preparing streaming dataset from {file_path}...")
                start_time = time.time()

                self.file_path = file_path
                self.file_size = file_size
                self.chunks = self._create_chunks_streaming()

                total_time = time.time() - start_time
                print(".2f")
            else:
                # Traditional mode: load entire file
                print(f"Loading and tokenizing {file_path}...")
                start_time = time.time()

                text = self._load_text(file_path, chunk_size)
                load_time = time.time() - start_time
                print(".2f")

                tokenize_start = time.time()
                self.tokens = self.processor.encode(text)
                tokenize_time = time.time() - tokenize_start
                print(f"Tokenized {len(self.tokens)} tokens in {tokenize_time:.2f}s "
                      f"({len(self.tokens)/tokenize_time:.0f} tokens/sec)")

                chunk_start = time.time()
                self.chunks = self._create_chunks()
                chunk_time = time.time() - chunk_start
                print(f"Created {len(self.chunks)} chunks of size {block_size} in {chunk_time:.2f}s")

                total_time = time.time() - start_time
                print(".2f")

    def _load_text(self, file_path: str, chunk_size: Optional[int] = None) -> str:
        """Load text from file or directory."""
        import os

        if os.path.isdir(file_path):
            # Load from directory - concatenate all text files
            texts = []
            for filename in os.listdir(file_path):
                filepath = os.path.join(file_path, filename)
                if os.path.isfile(filepath) and filename.endswith(('.txt', '.md')):
                    print(f"Loading file: {filename}")
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            file_text = f.read(chunk_size) if chunk_size else f.read()
                            texts.append(file_text)
                    except Exception as e:
                        print(f"Warning: Could not read {filename}: {e}")
            return ' '.join(texts)
        else:
            # Load from single file - show file size for large files
            file_size = os.path.getsize(file_path)
            if file_size > 500 * 1024 * 1024:  # > 500MB
                print(".2f")
            elif file_size > 100 * 1024 * 1024:  # > 100MB
                print(".2f")
            elif file_size > 10 * 1024 * 1024:  # > 10MB
                print(".2f")

            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read(chunk_size) if chunk_size else f.read()
    
    def _create_chunks_streaming(self) -> List[Tuple[int, int]]:
        """Create chunk metadata for streaming processing.

        For streaming, we pre-calculate which byte ranges of the file correspond
        to which training chunks. Each training chunk is identified by:
        (file_chunk_index, token_offset_within_chunk)

        Returns:
            List of tuples: (file_chunk_start_byte, file_chunk_end_byte)
        """
        print(f"Setting up streaming chunks for {self.file_path}...")

        # Divide file into byte chunks
        chunk_ranges = []
        file_pos = 0
        while file_pos < self.file_size:
            chunk_start = file_pos
            chunk_end = min(file_pos + self.stream_chunk_size, self.file_size)
            chunk_ranges.append((chunk_start, chunk_end))
            file_pos = chunk_end

        print(f"File divided into {len(chunk_ranges)} chunks of ~{self.stream_chunk_size // (1024*1024)}MB each")
        return chunk_ranges

    def _create_chunks(self) -> List[List[int]]:
        """Create overlapping chunks from tokens."""
        chunks = []

        # Guard against insufficient data
        if len(self.tokens) < self.block_size:
            print(f"Warning: Dataset has only {len(self.tokens)} tokens, but block_size is {self.block_size}. "
                  f"No chunks will be created. Consider using a smaller block_size or more data.")
            return chunks

        stride = self.block_size - self.overlap

        for i in range(0, len(self.tokens) - self.block_size + 1, stride):
            chunk = self.tokens[i:i + self.block_size]
            chunks.append(chunk)

        # Add last chunk if needed
        if len(self.tokens) % stride != 0:
            last_start = len(self.tokens) - self.block_size
            if last_start > 0:
                chunks.append(self.tokens[last_start:])

        return chunks
    
    def __len__(self) -> int:
        if self.streaming:
            # Estimate total chunks based on file size and tokenization ratio
            # This is approximate but good enough for DataLoader
            avg_tokens_per_byte = 0.1  # Conservative estimate: 1 token per 10 bytes
            estimated_total_tokens = int(self.file_size * avg_tokens_per_byte)
            if estimated_total_tokens < self.block_size:
                return 0
            stride = self.block_size - self.overlap
            return max(1, (estimated_total_tokens - self.block_size) // stride + 1)
        else:
            return len(self.chunks)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a chunk as tensor with input and targets.

        Args:
            idx: Index of the chunk

        Returns:
            Tuple of (input_ids, targets) where:
            - input_ids: All tokens except last (shape: [block_size - 1])
            - targets: All tokens except first (shape: [block_size - 1])
        """
        if self.streaming:
            chunk = self._get_streaming_chunk(idx)
        else:
            chunk = self.chunks[idx]

        chunk_tensor = torch.tensor(chunk, dtype=torch.long)

        if self.overlap > 0:
            # With overlap, adjust to avoid training on duplicate sequences
            # Remove overlap tokens from both ends to prevent data leakage
            input_ids = chunk_tensor[:-1-self.overlap]
            targets = chunk_tensor[1+self.overlap:]
        else:
            # Standard case: input is all tokens except last, targets are all tokens except first
            input_ids = chunk_tensor[:-1]
            targets = chunk_tensor[1:]

        return input_ids, targets

    def _get_streaming_chunk(self, idx: int) -> List[int]:
        """Get a chunk in streaming mode by loading data on demand."""
        # Calculate which file chunk this training sample belongs to
        stride = self.block_size - self.overlap
        tokens_per_file_chunk = int(self.stream_chunk_size * 0.1)  # Estimate tokens per file chunk
        chunks_per_file_chunk = max(1, (tokens_per_file_chunk - self.block_size) // stride + 1)

        file_chunk_idx = idx // chunks_per_file_chunk
        chunk_within_file = idx % chunks_per_file_chunk

        # Load the appropriate file chunk
        if file_chunk_idx >= len(self.chunks):
            # Handle out of bounds by using the last available chunk
            file_chunk_idx = len(self.chunks) - 1
            chunk_within_file = 0  # Just take the first chunk from the last file chunk

        chunk_start_byte, chunk_end_byte = self.chunks[file_chunk_idx]

        # Read and tokenize the file chunk
        with open(self.file_path, 'rb') as f:
            f.seek(chunk_start_byte)
            chunk_data = f.read(chunk_end_byte - chunk_start_byte)
            text = chunk_data.decode('utf-8', errors='ignore')

        tokens = self.processor.encode(text)

        # Extract the specific training chunk
        token_start = chunk_within_file * stride
        if token_start + self.block_size > len(tokens):
            # Last chunk of this file chunk - take the final block_size tokens
            if len(tokens) >= self.block_size:
                token_start = len(tokens) - self.block_size
            else:
                # File chunk too small, pad with pad tokens
                token_start = 0

        chunk = tokens[token_start:token_start + self.block_size]
        if len(chunk) < self.block_size:
            # Pad if necessary
            pad_needed = self.block_size - len(chunk)
            chunk.extend([self.processor.pad_token_id] * pad_needed)

        return chunk


def create_dataloader(
    file_path: str,
    processor: TextProcessor,
    block_size: int,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    streaming: bool = None
) -> DataLoader:
    """Create a DataLoader for training.

    Args:
        file_path: Path to text file
        processor: TextProcessor instance
        block_size: Maximum sequence length
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory for faster GPU transfer
        streaming: Whether to use streaming processing for large files.
                  If None, automatically enabled for files > 500MB.

    Returns:
        DataLoader instance
    """
    # Auto-enable streaming for large files if not specified
    if streaming is None:
        import os
        if os.path.isfile(file_path):
            file_size = os.path.getsize(file_path)
            streaming = file_size > 500 * 1024 * 1024  # > 500MB
        else:
            streaming = False

    dataset = TextFileDataset(file_path, processor, block_size, streaming=streaming)
    
    # Custom collate function to handle variable length sequences
    pad_token_id = processor.pad_token_id
    
    def collate_fn(batch):
        input_ids_list, targets_list = zip(*batch)

        # Check if padding is actually needed
        lengths = [len(seq) for seq in input_ids_list]
        max_len = max(lengths)
        all_same_length = all(length == max_len for length in lengths)

        if all_same_length:
            # No padding needed - just stack tensors
            input_ids_batch = torch.stack(input_ids_list)
            targets_batch = torch.stack(targets_list)
            attention_masks = torch.ones_like(input_ids_batch, dtype=torch.long)
        else:
            # Use efficient pad_sequence for variable lengths
            from torch.nn.utils.rnn import pad_sequence

            # Pad input_ids and targets
            input_ids_batch = pad_sequence(input_ids_list, batch_first=True, padding_value=pad_token_id)
            targets_batch = pad_sequence(targets_list, batch_first=True, padding_value=-1)  # -1 for ignore_index

            # Create attention masks: 1 for real tokens, 0 for padding
            attention_masks = torch.zeros_like(input_ids_batch, dtype=torch.long)
            for i, length in enumerate(lengths):
                attention_masks[i, :length] = 1

            # Ensure targets are -1 where attention_mask is 0 (padding positions)
            # This ensures cross_entropy with ignore_index=-1 works correctly
            targets_batch = torch.where(attention_masks == 1, targets_batch, -1)

        return input_ids_batch, targets_batch, attention_masks
    
    # For streaming datasets, disable shuffle and use fewer workers to avoid issues
    if streaming:
        shuffle = False
        num_workers = min(num_workers, 2)  # Limit workers for streaming
        print(f"Streaming mode: shuffle={shuffle}, num_workers={num_workers}")

    # Check and adjust num_workers for multiprocessing compatibility
    if num_workers > 0:
        try:
            import multiprocessing
            max_workers = min(num_workers, multiprocessing.cpu_count())
            if max_workers != num_workers:
                logger.warning(f"Capping num_workers from {num_workers} to {max_workers} (CPU count limit)")
                num_workers = max_workers
        except (ImportError, OSError):
            logger.warning("Multiprocessing not available, setting num_workers=0")
            num_workers = 0

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn
    )
