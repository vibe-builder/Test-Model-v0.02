# Nano XYZ Model Constants
# ================================

# Default model configuration values
DEFAULT_VOCAB_SIZE = 50257  # GPT-2 vocabulary size
DEFAULT_BLOCK_SIZE = 1024   # Maximum sequence length
DEFAULT_N_LAYER = 12        # Number of transformer layers
DEFAULT_N_HEAD = 12         # Number of attention heads
DEFAULT_N_EMBD = 768        # Embedding dimension
DEFAULT_DROPOUT = 0.0       # Default dropout rate
DEFAULT_BIAS = True         # Use bias in linear layers by default

# Weight initialization standard deviation
WEIGHT_INIT_STD = 0.02      # Standard initialization for transformer weights

# Dropout constraints for validation
DROPOUT_MIN = 0.0           # Minimum allowed dropout
DROPOUT_MAX = 0.5           # Maximum allowed dropout (to prevent overfitting)
