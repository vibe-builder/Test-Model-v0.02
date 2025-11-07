"""
Nano XYZ - A lightweight transformer language model.

This package contains:
- model.py: Core model architecture
- utils.py: Training utilities (optimizers, performance monitoring)
- processor.py: Text tokenization with HuggingFace tokenizers
- dataset.py: Data loading and batching
- checkpoint.py: Model checkpoint management
- generator.py: Text generation interface
- train.py: Main training script
"""

__version__ = "1.0.0"

from .model import ModelArchitecture, ModelSettings, RopeWithYaRN, LCRBlock, GTRBlock
from .utils import OptimizerFactory, PerformanceMonitor
from .processor import TextProcessor
from .dataset import TextFileDataset, create_dataloader
from .checkpoint import CheckpointManager
from .generator import TextGenerator

__all__ = [
    "ModelArchitecture",
    "ModelSettings",
    "RopeWithYaRN",
    "LCRBlock",
    "GTRBlock",
    "OptimizerFactory",
    "PerformanceMonitor",
    "TextProcessor",
    "TextFileDataset",
    "create_dataloader",
    "CheckpointManager",
    "TextGenerator",
]

