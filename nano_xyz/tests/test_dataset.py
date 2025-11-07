"""Tests for dataset components."""

import pytest
import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import nano_xyz
from nano_xyz.dataset import create_dataloader, TextFileDataset
from nano_xyz.processor import TextProcessor


class TestDatasetComponents:
    """Test dataset components."""

    def test_collate_ignore_index(self):
        """Test dataset collate correctly sets targets==-1 on padding."""
        # Create a temporary text file for testing
        import tempfile
        test_text = "This is a test sentence for dataset collation."

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(test_text)
            temp_file = f.name

        try:
            # Create processor
            processor = TextProcessor(vocab_size=1000)  # Small vocab for testing

            # Create dataset with small chunks
            dataset = TextFileDataset(temp_file, processor, block_size=8, overlap=2)

            # Get a batch with variable lengths (simulate padding scenario)
            # Manually create batch data to test collate function
            from torch.utils.data import DataLoader

            # Create dataloader to test collate
            dataloader = create_dataloader(temp_file, processor, block_size=8, batch_size=2)

            # Get one batch
            batch = next(iter(dataloader))
            input_ids, targets, attention_masks = batch

            # Verify shapes
            assert input_ids.shape[0] == 2  # batch_size
            assert targets.shape[0] == 2
            assert attention_masks.shape[0] == 2

            # Check that targets are -1 where attention_mask is 0 (padding positions)
            # This ensures cross_entropy with ignore_index=-1 works correctly
            padding_mask = attention_masks == 0
            assert torch.all(targets[padding_mask] == -1), "Targets should be -1 for padding positions"

            # Check that targets are not -1 where attention_mask is 1 (real positions)
            real_mask = attention_masks == 1
            if real_mask.any():
                assert torch.all(targets[real_mask] != -1), "Targets should not be -1 for real positions"

            print("Collate ignore_index test passed!")

        finally:
            # Clean up temp file
            os.unlink(temp_file)
