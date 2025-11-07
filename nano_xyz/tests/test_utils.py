"""Tests for utility functions."""

import pytest
import torch
import tempfile
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import nano_xyz
from nano_xyz.utils import PerformanceMonitor, amp_context, create_amp_scaler
from nano_xyz.checkpoint import CheckpointManager


class TestPerformanceMonitor:
    """Test performance monitoring utilities."""

    def test_flops_estimation(self, small_model):
        """Test FLOPs estimation."""
        # Simulate 1 iteration of forward+backward
        mfu = PerformanceMonitor.estimate_model_flops_utilization(
            small_model, 1, 1.0  # 1 iter, 1 second
        )

        assert isinstance(mfu, float)
        assert 0 <= mfu <= 1  # MFU should be between 0 and 1

    def test_device_detection(self):
        """Test device detection for FLOPs."""
        flops = PerformanceMonitor._detect_device_peak_flops("auto")
        assert isinstance(flops, float)
        assert flops > 0


class TestAMPUtilities:
    """Test mixed precision utilities."""

    def test_amp_context_cpu(self):
        """Test AMP context on CPU."""
        from nano_xyz.model import ModelSettings
        config = ModelSettings(dtype='float32')

        device = torch.device('cpu')

        # Should work without CUDA
        with amp_context(config, device):
            x = torch.randn(2, 2)
            assert x.device == device

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_amp_context_cuda(self):
        """Test AMP context on CUDA."""
        from nano_xyz.model import ModelSettings

        for dtype in ['float16', 'bfloat16']:
            config = ModelSettings(dtype=dtype)
            device = torch.device('cuda')

            with amp_context(config, device):
                x = torch.randn(2, 2, device=device)
                assert x.device == device
                assert x.dtype in [torch.float16, torch.bfloat16]

    def test_amp_scaler_creation(self):
        """Test GradScaler creation."""
        from nano_xyz.model import ModelSettings

        # Float16 on CUDA should create scaler
        if torch.cuda.is_available():
            config = ModelSettings(dtype='float16')
            device = torch.device('cuda')
            scaler = create_amp_scaler(config, device)
            assert scaler is not None

        # Float32 should not create scaler
        config = ModelSettings(dtype='float32')
        device = torch.device('cpu')
        scaler = create_amp_scaler(config, device)
        assert scaler is None


class TestCheckpointManager:
    """Test checkpoint functionality."""

    def test_checkpoint_save_load(self, small_model):
        """Test checkpoint save and load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save checkpoint
            checkpoint_path = CheckpointManager.save_checkpoint(
                model=small_model,
                optimizer=torch.optim.AdamW(small_model.parameters()),
                epoch=1,
                step=100,
                loss=1.5,
                checkpoint_dir=tmpdir
            )

            assert os.path.exists(checkpoint_path)

            # Load checkpoint
            new_model = type(small_model)(small_model.config)
            new_optimizer = torch.optim.AdamW(new_model.parameters())

            checkpoint = CheckpointManager.load_checkpoint(
                checkpoint_path, new_model, new_optimizer
            )

            assert checkpoint['epoch'] == 1
            assert checkpoint['step'] == 100
            assert abs(checkpoint['loss'] - 1.5) < 1e-6

            # Verify model parameters match
            for (name1, param1), (name2, param2) in zip(
                small_model.named_parameters(),
                new_model.named_parameters()
            ):
                assert name1 == name2
                torch.testing.assert_close(param1, param2)
