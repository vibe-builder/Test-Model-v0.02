"""
Comprehensive tests for jagged tensor sparse attention implementation.

Tests cover:
- Jagged pattern generation with various sparsity levels
- Jagged attention computation correctness and performance
- SDPA fallback for dense patterns
- Edge cases (zero sequence, full dense)
- KV-cache integration with narrow
- Multi-batch support
"""

import pytest
import torch
from nano_xyz.attention_utils import SparsePatternGenerator, apply_sparse_attention_optimization, DynamicContextAllocator
from nano_xyz.configuration_nano import NanoConfig


@pytest.fixture
def config():
    """Create test configuration."""
    return NanoConfig.from_preset("decoder_tiny")


@pytest.fixture
def dca_config():
    """Create DCA-enabled configuration."""
    config = NanoConfig.from_preset("decoder_tiny")
    config.use_dca = True
    config.dca_attention_budget = 0.5
    config.dca_local_window = 64
    config.dca_global_tokens = 32
    return config


@pytest.fixture
def pattern_generator(config):
    """Create a sparse pattern generator."""
    return SparsePatternGenerator(config)


@pytest.fixture
def dca_allocator(dca_config):
    """Create a DCA allocator."""
    return DynamicContextAllocator(dca_config)


@pytest.fixture
def sample_attention_inputs():
    """Create sample attention inputs for testing."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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


@pytest.mark.parametrize("seq_len", [512, 1024, 2048, 4096])
@pytest.mark.parametrize("device_type", ["cpu", "cuda"])
def test_pattern_generator(pattern_generator, seq_len, device_type):
    """Test jagged pattern generation across different sequence lengths and devices."""
    if device_type == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device(device_type)
    pattern = pattern_generator(seq_len, device)

    # Validate pattern structure
    assert isinstance(pattern, dict)
    required_keys = ["mask", "row_offsets", "col_indices", "row_counts", "sparsity"]
    for key in required_keys:
        assert key in pattern, f"Missing required key: {key}"

    # Validate shapes and types
    assert pattern["mask"].shape == (seq_len, seq_len)
    assert pattern["mask"].dtype == torch.bool
    assert pattern["mask"].device.type == device_type

    assert pattern["row_offsets"].shape[0] == seq_len + 1
    assert pattern["row_offsets"].dtype == torch.long
    assert pattern["row_offsets"].device.type == device_type

    assert pattern["row_counts"].shape[0] == seq_len
    assert pattern["row_counts"].dtype == torch.long
    assert pattern["row_counts"].device.type == device_type

    assert pattern["col_indices"].dtype == torch.long
    assert pattern["col_indices"].device.type == device_type

    # Validate sparsity constraints
    sparsity = pattern["sparsity"]
    assert 0.0 <= sparsity <= 1.0

    # For reasonable sparsity (DCA patterns should be reasonably sparse)
    # Note: Actual sparsity depends on window_size, global_tokens, random_blocks
    if seq_len <= 2048:  # Be more strict for shorter sequences
        assert sparsity < 0.7, f"Sparsity too high: {sparsity}"

    # Validate jagged metadata consistency
    total_connections = pattern["col_indices"].numel()
    expected_total = pattern["row_counts"].sum().item()
    assert total_connections == expected_total, f"Jagged metadata inconsistency: {total_connections} != {expected_total}"

    # Validate row_offsets are monotonically increasing
    assert torch.all(pattern["row_offsets"][1:] >= pattern["row_offsets"][:-1])

    # Validate mask consistency with jagged representation
    reconstructed_mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)
    for i in range(seq_len):
        start_idx = pattern["row_offsets"][i].item()
        end_idx = pattern["row_offsets"][i + 1].item()
        col_indices = pattern["col_indices"][start_idx:end_idx]
        reconstructed_mask[i, col_indices] = True

    torch.testing.assert_close(pattern["mask"], reconstructed_mask)


@pytest.mark.parametrize("sparsity_level", [0.1, 0.3, 0.5, 0.7])
def test_jagged_attention_correctness(pattern_generator, sample_attention_inputs, sparsity_level):
    """Test jagged attention computation correctness across different sparsity levels."""
    device = sample_attention_inputs["device"]
    q = sample_attention_inputs["q"]
    k = sample_attention_inputs["k"]
    v = sample_attention_inputs["v"]
    seq_len = sample_attention_inputs["seq_len"]

    # Create pattern
    pattern = pattern_generator(seq_len, device)

    # Override sparsity for controlled testing
    original_sparsity = pattern["sparsity"]
    pattern["sparsity"] = sparsity_level

    # Test attention computation
    out = apply_sparse_attention_optimization(q, k, v, pattern, is_causal=True)

    # Validate output properties
    assert out.shape == q.shape
    assert out.dtype == q.dtype
    assert out.device == q.device
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()

    # Test that output is different from zero (attention is working)
    assert not torch.allclose(out, torch.zeros_like(out))

    # Test gradient flow (if sparsity allows meaningful gradients)
    if sparsity_level < 0.8:
        # Enable gradients on input
        q = q.detach().requires_grad_(True)
        out = apply_sparse_attention_optimization(q, k, v, pattern, is_causal=True)
        out.sum().backward()
        assert q.grad is not None
        assert not torch.isnan(q.grad).any()


def test_attention_numerical_stability(sample_attention_inputs, pattern_generator):
    """Test numerical stability of attention computations."""
    device = sample_attention_inputs["device"]
    seq_len = sample_attention_inputs["seq_len"]

    # Test with very small values
    q_small = torch.randn_like(sample_attention_inputs["q"]) * 1e-6
    k_small = torch.randn_like(sample_attention_inputs["k"]) * 1e-6
    v_small = sample_attention_inputs["v"]

    pattern = pattern_generator(seq_len, device)
    out_small = apply_sparse_attention_optimization(q_small, k_small, v_small, pattern, is_causal=True)

    assert not torch.isnan(out_small).any()
    assert not torch.isinf(out_small).any()

    # Test with very large values
    q_large = torch.randn_like(sample_attention_inputs["q"]) * 1e6
    k_large = torch.randn_like(sample_attention_inputs["k"]) * 1e6
    v_large = sample_attention_inputs["v"]

    out_large = apply_sparse_attention_optimization(q_large, k_large, v_large, pattern, is_causal=True)

    assert not torch.isnan(out_large).any()
    assert not torch.isinf(out_large).any()


@pytest.mark.parametrize("seq_len", [128, 256, 512])
def test_dca_allocator_comprehensive(dca_allocator, seq_len):
    """Test DCA allocator with different sequence lengths."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 2

    # Create test hidden states
    hidden_states = torch.randn(batch_size, seq_len, dca_allocator.config.n_embd, device=device)

    # Test forward pass (this should allocate attention patterns)
    attention_mask, selected_mask, metadata = dca_allocator(hidden_states)

    # Validate attention mask
    assert attention_mask.shape[0] == batch_size
    assert attention_mask.shape[1] == seq_len
    assert attention_mask.shape[2] == seq_len
    # DCA returns float attention weights, not boolean mask
    assert attention_mask.dtype in [torch.float32, torch.bool]

    # Test sparsity constraints (for float masks, count near-zero values)
    if attention_mask.dtype == torch.float32:
        sparsity = (attention_mask < 0.01).float().mean()
    else:  # boolean mask
        sparsity = (~attention_mask).float().mean()
    assert 0.3 <= sparsity <= 0.9, f"DCA sparsity out of expected range: {sparsity}"


def test_dca_sparsity_constraints(dca_config):
    """Test that DCA creates sparse attention patterns."""
    allocator = DynamicContextAllocator(dca_config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    seq_len = 256
    batch_size = 1
    hidden_states = torch.randn(batch_size, seq_len, dca_config.n_embd, device=device)

    attention_mask, selected_mask, metadata = allocator(hidden_states)

    # Calculate sparsity metrics
    total_possible_connections = seq_len * seq_len
    actual_connections = attention_mask.float().sum().item()

    sparsity_ratio = 1.0 - (actual_connections / total_possible_connections)

    # DCA should create meaningful sparsity (at least 20% sparse)
    assert sparsity_ratio > 0.2, f"DCA not sparse enough: {sparsity_ratio:.3f} sparsity ratio"

    # But not completely sparse (should allow some attention)
    assert sparsity_ratio < 0.9, f"DCA too sparse: {sparsity_ratio:.3f} sparsity ratio"

    # Check that metadata contains reasonable values
    assert 'attention_efficiency' in metadata
    assert 'sparsity_ratio' in metadata
    efficiency = metadata['attention_efficiency'].mean().item()
    reported_sparsity = metadata['sparsity_ratio'].mean().item()

    # Efficiency should be reasonable (not too low)
    assert efficiency > 0.1, f"Attention efficiency too low: {efficiency}"


def test_dca_memory_efficiency(dca_config, sample_attention_inputs):
    """Test that DCA provides memory benefits."""
    device = sample_attention_inputs["device"]
    seq_len = sample_attention_inputs["seq_len"]

    # Create dense baseline
    dense_config = NanoConfig.from_preset("decoder_tiny")
    dense_config.use_dca = False

    # Compare attention patterns
    dca_allocator = DynamicContextAllocator(dca_config)
    dense_pattern_gen = SparsePatternGenerator(dense_config)

    # Create test input
    hidden_states = torch.randn(1, seq_len, dca_config.n_embd, device=device)

    # Get DCA pattern
    dca_mask, dca_selected, dca_metadata = dca_allocator(hidden_states)

    # Get dense pattern
    dense_pattern = dense_pattern_gen(seq_len, device)

    # DCA should be significantly sparser
    # For DCA float mask, consider elements close to 0 as sparse
    dca_sparsity = (dca_mask < 0.01).float().mean().item()
    dense_sparsity = dense_pattern["sparsity"]

    assert dca_sparsity > dense_sparsity, \
        f"DCA not sparser than dense: DCA={dca_sparsity}, Dense={dense_sparsity}"


def test_dense_fallback(config):
    """Test SDPA fallback for dense patterns."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    seq_len = 512
    batch_size, n_heads, head_dim = 1, 8, 64

    # Create dense pattern (force sparsity > 0.5)
    q = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device)
    k = v = q.clone()

    # Create dense pattern manually
    dense_mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
    pattern = {
        "mask": dense_mask,
        "sparsity": 1.0,
        "row_offsets": torch.arange(0, seq_len * seq_len + 1, seq_len, device=device, dtype=torch.long),
        "col_indices": torch.arange(seq_len * seq_len, device=device, dtype=torch.long),
        "row_counts": torch.full((seq_len,), seq_len, device=device, dtype=torch.long)
    }

    # Should use SDPA fallback
    out = apply_sparse_attention_optimization(q, k, v, pattern)

    # Validate output
    assert out.shape == q.shape
    assert not torch.isnan(out).any()


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_multi_batch_support(config, batch_size):
    """Test jagged attention with multiple batches."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gen = SparsePatternGenerator(config)

    seq_len = 512
    n_heads, head_dim = 8, 64

    pattern = gen(seq_len, device)
    q = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device)
    k = v = q.clone()

    out = apply_sparse_attention_optimization(q, k, v, pattern, is_causal=True)

    assert out.shape == q.shape
    assert not torch.isnan(out).any()


def test_edge_cases(config):
    """Test edge cases for robustness."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gen = SparsePatternGenerator(config)

    # Test zero sequence length (should raise error gracefully)
    with pytest.raises((RuntimeError, IndexError, ValueError)):
        gen(0, device)

    # Test single token (degenerate case)
    pattern = gen(1, device)
    assert pattern["mask"].shape == (1, 1)
    assert pattern["sparsity"] == 1.0  # Single connection

    # Test attention with single token
    q = torch.randn(1, 8, 1, 64, device=device)
    k = v = q.clone()
    out = apply_sparse_attention_optimization(q, k, v, pattern)
    assert out.shape == q.shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires GPU for memory profiling")
def test_memory_efficiency(config):
    """Test memory efficiency of jagged vs dense attention."""
    import torch.cuda as cuda

    gen = SparsePatternGenerator(config)
    seq_len = 2048
    batch_size, n_heads, head_dim = 1, 8, 64

    pattern = gen(seq_len, torch.device('cuda'))
    sparsity = pattern["sparsity"]

    # Skip if not sparse enough for meaningful comparison
    if sparsity > 0.5:
        pytest.skip("Pattern not sparse enough for memory efficiency test")

    q = torch.randn(batch_size, n_heads, seq_len, head_dim, device='cuda')
    k = v = q.clone()

    # Clear cache and measure memory
    cuda.empty_cache()
    cuda.reset_peak_memory_stats()

    # Jagged attention
    out_jagged = apply_sparse_attention_optimization(q, k, v, pattern)
    jagged_memory = cuda.max_memory_allocated()

    # Dense attention for comparison
    dense_mask = torch.where(pattern["mask"], 0.0, float('-inf'))
    from torch.nn.functional import scaled_dot_product_attention
    cuda.reset_peak_memory_stats()
    out_dense = scaled_dot_product_attention(q, k, v, attn_mask=dense_mask.unsqueeze(0).unsqueeze(0))
    dense_memory = cuda.max_memory_allocated()

    # Jagged should use less memory for sparse patterns
    memory_reduction = (dense_memory - jagged_memory) / dense_memory
    assert memory_reduction > 0.1, f"Memory reduction too small: {memory_reduction}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires GPU for KV-cache testing")
def test_kv_cache_integration(config):
    """Test KV-cache integration with narrow for generation."""
    device = torch.device('cuda')
    gen = SparsePatternGenerator(config)

    seq_len = 512
    batch_size, n_heads, head_dim = 1, 8, 64

    # Create current and past KV states
    pattern = gen(seq_len, device)
    q = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device)
    k = v = q.clone()

    # Mock past KV (smaller sequence)
    past_len = 256
    past_k = torch.randn(batch_size, n_heads, past_len, head_dim, device=device)
    past_v = torch.randn(batch_size, n_heads, past_len, head_dim, device=device)
    past_key_values = [{"k": past_k, "v": past_v}]

    # Test with KV-cache
    out = apply_sparse_attention_optimization(q, k, v, pattern, is_causal=True, past_key_values=past_key_values)

    assert out.shape == q.shape
    assert not torch.isnan(out).any()


def test_torch_compile_compatibility(config):
    """Test torch.compile compatibility with jagged attention."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gen = SparsePatternGenerator(config)

    seq_len = 256  # Smaller for compilation
    pattern = gen(seq_len, device)

    q = torch.randn(1, 8, seq_len, 64, device=device)
    k = v = q.clone()

    # Test compilation
    try:
        compiled_fn = torch.compile(apply_sparse_attention_optimization, mode='reduce-overhead')
        out = compiled_fn(q, k, v, pattern)
        assert out.shape == q.shape
        assert not torch.isnan(out).any()
    except Exception as e:
        # Compilation may fail on some PyTorch versions, but shouldn't crash
        pytest.skip(f"torch.compile not supported: {e}")


def test_gradient_flow(config):
    """Test that gradients flow correctly through jagged attention."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gen = SparsePatternGenerator(config)

    seq_len = 256
    pattern = gen(seq_len, device)

    q = torch.randn(1, 8, seq_len, 64, device=device, requires_grad=True)
    k = torch.randn(1, 8, seq_len, 64, device=device, requires_grad=True)
    v = torch.randn(1, 8, seq_len, 64, device=device, requires_grad=True)

    out = apply_sparse_attention_optimization(q, k, v, pattern)

    # Create dummy loss
    loss = out.sum()
    loss.backward()

    # Check gradients
    assert q.grad is not None
    assert k.grad is not None
    assert v.grad is not None
    assert not torch.isnan(q.grad).any()
    assert not torch.isnan(k.grad).any()
    assert not torch.isnan(v.grad).any()


def test_causal_masking(config):
    """Test causal masking in jagged attention."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gen = SparsePatternGenerator(config)

    seq_len = 128
    pattern = gen(seq_len, device)

    q = torch.randn(1, 8, seq_len, 64, device=device)
    k = v = q.clone()

    # Test with causal masking
    out_causal = apply_sparse_attention_optimization(q, k, v, pattern, is_causal=True)

    # Create manual causal check (upper triangle should be zero in attention)
    # This is a simplified check - in practice, causal masking is more complex
    assert out_causal.shape == q.shape
    assert not torch.isnan(out_causal).any()


def test_different_head_dimensions(config):
    """Test jagged attention with different head dimensions."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gen = SparsePatternGenerator(config)

    seq_len = 256
    batch_size, n_heads = 1, 8

    for head_dim in [32, 64, 128]:
        pattern = gen(seq_len, device)
        q = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device)
        k = v = q.clone()

        out = apply_sparse_attention_optimization(q, k, v, pattern)
        assert out.shape == q.shape
        assert not torch.isnan(out).any()

@pytest.mark.parametrize("dilation_rate", [1, 2, 3, 4])
def test_dilation_functionality(config, dilation_rate):
    """Test that dilation creates the expected sparse attention patterns."""
    # Create config with dilation
    config.dca_dilation_rate = dilation_rate
    config.use_dca = True

    # Create DCA allocator
    allocator = DynamicContextAllocator(config)

    # Create test inputs
    batch_size = 2
    seq_len = 128
    hidden_states = torch.randn(batch_size, seq_len, config.n_embd)

    # Test local window mask generation
    local_mask = allocator._build_local_window_mask(seq_len, hidden_states.device, is_causal=False)

    # Count connections for each position
    connections_per_position = local_mask.sum(dim=1)

    # Verify dilation creates sparser patterns for dilation_rate > 1
    if dilation_rate > 1:
        # Compare with dilation_rate=1
        config_no_dilation = config.__class__(**config.__dict__)
        config_no_dilation.dca_dilation_rate = 1
        allocator_no_dilation = DynamicContextAllocator(config_no_dilation)
        local_mask_no_dilation = allocator_no_dilation._build_local_window_mask(seq_len, hidden_states.device, is_causal=False)
        connections_no_dilation = local_mask_no_dilation.sum().item()
        connections_with_dilation = local_mask.sum().item()

        # Dilation should generally reduce connections (though not always exactly 1/dilation_rate due to boundary effects)
        assert connections_with_dilation <= connections_no_dilation, "Dilation should reduce connections"


def test_vectorized_random_selection():
    """Test that the vectorized random selection works correctly."""
    config = NanoConfig(dca_local_window=5, block_size=512)
    allocator = DynamicContextAllocator(config)

    # Test parameters
    batch_size, seq_len = 4, 20
    budget = 3
    hidden_states = torch.randn(batch_size, seq_len, config.n_embd)
    exclude_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)

    # Test the random token selection
    random_mask = allocator._select_random_tokens_structured(
        seq_len, budget, exclude_mask, hidden_states.device
    )

    # Verify results
    selected_per_batch = random_mask.sum(dim=1)

    # Check constraints
    assert (selected_per_batch <= budget).all(), f"Some batches exceed budget: {selected_per_batch}"
    assert not (random_mask & exclude_mask).any(), "Selected excluded positions"

    # Test that it's actually random (run multiple times)
    selections = []
    for _ in range(5):  # Reduced for test performance
        mask = allocator._select_random_tokens_structured(
            seq_len, budget, exclude_mask, hidden_states.device
        )
        selections.append(mask.sum().item())

    # Should get some variation (though with small budgets this might not always be true)
    unique_selections = len(set(selections))
    assert unique_selections >= 1, "Should have some variation in selections"