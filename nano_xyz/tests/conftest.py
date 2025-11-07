"""Pytest configuration and fixtures."""

import pytest
import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import nano_xyz
from nano_xyz.model import ModelSettings, ModelArchitecture


@pytest.fixture(scope="session")
def small_config():
    """A small model configuration for testing."""
    return ModelSettings(
        n_layer=2,
        n_head=4,
        n_embd=64,
        block_size=32,
        vocab_size=1000,
        max_cache_len=16
    )


@pytest.fixture(scope="session")
def small_model(small_config):
    """A small model for testing."""
    return ModelArchitecture(small_config)


@pytest.fixture
def sample_batch(small_config):
    """Sample batch of token IDs."""
    batch_size, seq_len = 2, 10
    return torch.randint(0, small_config.vocab_size, (batch_size, seq_len))


@pytest.fixture
def device():
    """Get appropriate device for testing."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
