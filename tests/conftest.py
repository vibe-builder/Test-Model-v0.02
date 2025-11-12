import sys
import torch
import pytest
from pathlib import Path

# Ensure project root is on sys.path for package imports during pytest runs.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nano_xyz.configuration_nano import NanoConfig
from nano_xyz.model import NanoModel
from nano_xyz.modeling_nano import NanoForCausalLM, NanoEncoderModel, NanoDecoderModel
from nano_xyz.quantization import QuantizationConfig


@pytest.fixture(scope="session", autouse=True)
def set_test_seeds():
    """Set seeds for reproducible tests across the session."""
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)


@pytest.fixture
def device():
    """Get available device for testing."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def tiny_config():
    """Create a tiny config for fast testing."""
    return NanoConfig.from_preset("decoder_tiny")


@pytest.fixture
def encoder_config():
    """Create encoder config."""
    return NanoConfig.from_preset("encoder_decoder_tiny")


@pytest.fixture
def decoder_config():
    """Create decoder config."""
    return NanoConfig.from_preset("decoder_tiny")


@pytest.fixture
def dca_config():
    """Create DCA-enabled config."""
    config = NanoConfig.from_preset("decoder_tiny")
    config.use_dca = True
    config.dca_attention_budget = 0.5
    return config


@pytest.fixture
def quant_config_int8():
    """Create 8-bit quantization config."""
    return QuantizationConfig(
        method="torchao",
        bits=8,
        quant_type="int8_dyn_act_int4_weight",
        group_size=32,
        calibration_samples=10
    )


@pytest.fixture
def quant_config_int4():
    """Create 4-bit quantization config."""
    return QuantizationConfig(
        method="torchao",
        bits=4,
        quant_type="int8_dyn_act_int4_weight",
        group_size=32,
        calibration_samples=10
    )


@pytest.fixture
def tiny_model(tiny_config):
    """Create a tiny NanoForCausalLM model."""
    tiny_config.use_torch_compile = False  # Disable for testing
    return NanoForCausalLM(tiny_config)


@pytest.fixture
def dca_model(dca_config):
    """Create a DCA-enabled model."""
    dca_config.use_torch_compile = False
    return NanoForCausalLM(dca_config)


@pytest.fixture
def encoder_model(encoder_config):
    """Create encoder model."""
    encoder_config.use_torch_compile = False
    return NanoEncoderModel(encoder_config)


@pytest.fixture
def decoder_model(decoder_config):
    """Create decoder model."""
    decoder_config.use_torch_compile = False
    return NanoDecoderModel(decoder_config)


@pytest.fixture
def sample_inputs(tiny_config):
    """Create various sample inputs for testing."""
    batch_size = 2
    seq_len = 32
    return {
        "input_ids": torch.randint(0, tiny_config.vocab_size, (batch_size, seq_len)),
        "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
        "batch_size": batch_size,
        "seq_len": seq_len,
    }


@pytest.fixture
def sample_inputs_short(tiny_config):
    """Create short sample inputs."""
    return {
        "input_ids": torch.randint(0, tiny_config.vocab_size, (1, 5)),
        "attention_mask": torch.ones(1, 5, dtype=torch.long),
    }


@pytest.fixture
def sample_inputs_batch(tiny_config):
    """Create batch sample inputs."""
    return {
        "input_ids": torch.randint(0, tiny_config.vocab_size, (3, 10)),
        "attention_mask": torch.ones(3, 10, dtype=torch.long),
    }


@pytest.fixture
def sample_attention_inputs(device):
    """Create sample attention inputs for testing."""
    batch_size, n_heads, seq_len, head_dim = 1, 8, 512, 64

    return {
        "q": torch.randn(batch_size, n_heads, seq_len, head_dim, device=device),
        "k": torch.randn(batch_size, n_heads, seq_len, head_dim, device=device),
        "v": torch.randn(batch_size, n_heads, seq_len, head_dim, device=device),
        "device": device,
        "batch_size": batch_size,
        "n_heads": n_heads,
        "seq_len": seq_len,
        "head_dim": head_dim,
    }


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )
    config.addinivalue_line(
        "markers", "quantization: marks tests related to quantization"
    )

