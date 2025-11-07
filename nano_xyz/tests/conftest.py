"""Pytest configuration and fixtures."""

import pytest
import torch
import sys
import os

TESTS_DIR = os.path.dirname(__file__)
PACKAGE_DIR = os.path.dirname(TESTS_DIR)
REPO_ROOT = os.path.dirname(PACKAGE_DIR)

for path in (REPO_ROOT,):
    if path not in sys.path:
        sys.path.insert(0, path)

import nano_xyz
from nano_xyz.model import ModelSettings, ModelArchitecture


@pytest.fixture(autouse=True)
def stub_auto_tokenizer(monkeypatch):
    """Provide a lightweight tokenizer stub to avoid HuggingFace downloads during tests."""
    try:
        import transformers  # type: ignore
    except ImportError:  # pragma: no cover - transformers is an optional dev dependency
        return

    class DummyTokenizer:
        def __init__(self) -> None:
            self._vocab_size = 256
            self.pad_token_id = 0
            self.pad_token = "<pad>"
            self.eos_token_id = 1
            self.bos_token_id = 2
            self.unk_token_id = 3

        def __len__(self) -> int:
            return self._vocab_size

        def encode(self, text: str, return_tensors=None, add_special_tokens: bool = False):
            # Simple byte-level encoding bounded by vocab size
            encoded = [3 + (byte % (self._vocab_size - 4)) for byte in text.encode("utf-8", "ignore")]
            return encoded

        def decode(self, token_ids, skip_special_tokens: bool = False) -> str:
            special_ids = {self.pad_token_id, self.eos_token_id} if skip_special_tokens else set()
            chars = [
                chr((int(tid) % 26) + 97)
                for tid in token_ids
                if int(tid) not in special_ids
            ]
            return "".join(chars)

        def convert_ids_to_tokens(self, idx: int) -> str:
            return f"token_{idx}"

    def _from_pretrained(_: str):
        return DummyTokenizer()

    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", _from_pretrained, raising=False)


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
