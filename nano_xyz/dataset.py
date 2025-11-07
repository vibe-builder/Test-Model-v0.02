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
    """Dataset for loading and chunking large text files."""
    
    def __init__(
        self,
        file_path: str,
        processor: TextProcessor,
        block_size: int,
        chunk_size: Optional[int] = None,
        overlap: int = 0
    ):
        """Initialize dataset from text file or directory.

        Args:
            file_path: Path to text file or directory containing text files
            processor: TextProcessor instance
            block_size: Maximum sequence length
            chunk_size: Size of chunks to read per file (None = read all)
            overlap: Number of tokens to overlap between chunks
        """
        self.processor = processor
        self.block_size = block_size
        self.overlap = overlap

        # Read and tokenize file(s)
        print(f"Loading and tokenizing {file_path}...")
        text = self._load_text(file_path, chunk_size)

        # Tokenize entire text
        self.tokens = self.processor.encode(text)
        print(f"Tokenized {len(self.tokens)} tokens")

        # Create chunks
        self.chunks = self._create_chunks()
        print(f"Created {len(self.chunks)} chunks of size {block_size}")

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
            # Load from single file
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read(chunk_size) if chunk_size else f.read()
    
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


def create_dataloader(
    file_path: str,
    processor: TextProcessor,
    block_size: int,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True
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
        
    Returns:
        DataLoader instance
    """
    dataset = TextFileDataset(file_path, processor, block_size)
    
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
