"""Tests for checkpoint components."""

import pytest
import torch
import sys
import os
import tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import nano_xyz
from nano_xyz.model import ModelSettings, ModelArchitecture
from nano_xyz.checkpoint import CheckpointManager


class TestCheckpointComponents:
    """Test checkpoint components."""

    def test_checkpoint_roundtrip_tied_weights(self):
        """Test checkpoint round-trip with tied embeddings using safetensors."""
        config = ModelSettings(n_layer=2, n_head=4, n_embd=64, block_size=32)
        model = ModelArchitecture(config)

        # Verify that embeddings are tied
        assert torch.equal(model.transformer.token_embeddings.weight, model.language_model_head.weight), \
            "Embeddings should be tied initially"

        # Create some dummy optimizer and scheduler
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save checkpoint with safetensors
            checkpoint_path = CheckpointManager.save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=1,
                step=100,
                loss=0.5,
                checkpoint_dir=tmpdir,
                filename='test_checkpoint.pt',
                use_safetensors=True  # Force safetensors to test tied weights handling
            )

            # Create new model instance
            new_model = ModelArchitecture(config)

            # Verify new model also has tied embeddings
            assert torch.equal(new_model.transformer.token_embeddings.weight, new_model.language_model_head.weight), \
                "New model embeddings should be tied"

            # Load checkpoint
            loaded_checkpoint = CheckpointManager.load_checkpoint(
                checkpoint_path, model=new_model, optimizer=optimizer, scheduler=scheduler
            )

            # Verify that embeddings are still tied after loading
            assert torch.equal(new_model.transformer.token_embeddings.weight, new_model.language_model_head.weight), \
                "Embeddings should remain tied after loading"

            # Verify model weights were loaded (compare with original)
            original_params = list(model.parameters())
            loaded_params = list(new_model.parameters())
            assert len(original_params) == len(loaded_params), "Parameter count should match"

            for orig, loaded in zip(original_params, loaded_params):
                torch.testing.assert_close(orig, loaded, msg="Parameters should match after loading")

            print("Checkpoint round-trip with tied weights test passed!")
