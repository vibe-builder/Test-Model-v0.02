# Nano XYZ Profiling Suite

Comprehensive performance benchmarking and analysis toolkit for the Nano XYZ transformer implementation.

## Overview

This profiling suite provides detailed performance analysis of Nano XYZ components:

- **Attention Mechanisms**: Dense vs sparse attention latency and memory usage
- **Model Performance**: End-to-end latency and memory usage
- **Full Models**: End-to-end performance with memory and latency tracking
- **Visual Reports**: Charts and tables comparing different configurations

## Quick Start

### Run Complete Profiling Suite

```bash
# Quick profiling (development mode)
python profiling/run_profiling.py --quick

# Full comprehensive profiling
python profiling/run_profiling.py --full
```

### Individual Profiling Scripts

```bash
# Profile attention mechanisms only
python profiling/profile_attention.py --seq_lens 1024 4096 --attention_types dense sparse

# Profile full models only
python profiling/profile_model.py --presets decoder_tiny decoder_small --seq_lens 512 1024
```

### Generate Reports from Existing Data

```bash
# Generate visualizations and reports from profiling results
python profiling/generate_report.py --input_dir profiling/results --output_dir profiling/reports
```

## Output Files

### Raw Profiling Data (`profiling/results/`)

- `attention_profile.json`: Attention mechanism performance data
- `model_profile.json`: Full model performance data

### Reports and Visualizations (`profiling/reports/`)

- `attention_latency.png`: Attention latency vs sequence length
- `attention_memory.png`: Attention memory usage comparison
- `attention_efficiency.png`: Sparsity and memory efficiency charts
- `model_performance.png`: Complete model performance dashboard
- `*summary.csv/md`: Detailed performance tables in CSV and Markdown formats
- `profiling_report.md`: Comprehensive analysis report

## Profiling Metrics

### Attention Mechanisms

- **Latency**: End-to-end attention computation time (ms)
- **Memory Usage**: Peak GPU memory consumption (MB)
- **Sparsity Ratio**: Fraction of attention connections removed (0-1)
- **FLOPs**: Estimated floating point operations

### Full Models

- **Inference Latency**: Forward pass time (ms)
- **Training Latency**: Forward + backward pass time (ms)
- **Peak Memory**: Maximum memory usage during computation (MB)
- **Model Size**: Parameter count and memory footprint
- **Loss Values**: Training loss tracking

## Configuration Options

### Sequence Lengths

Test different context sizes:
- Small: 256-512 tokens (typical for many tasks)
- Medium: 1024-2048 tokens (longer contexts)
- Large: 4096-8192 tokens (very long contexts)

### Batch Sizes

- Default: 1 (single sequence)
- Larger batches: 4, 8, 16 (for throughput testing)

### Attention Types

- `dense`: Standard full attention (O(n²))
- `sparse`: DCA-based sparse attention (O(n*k))


## Interpreting Results

### Attention Performance

**Memory Efficiency**: Sparse attention should use significantly less memory than dense attention, especially for long sequences (>2048 tokens).

**Latency**: Sparse attention may be faster for very long sequences due to reduced computation, but dense attention often performs better for shorter sequences.

**Sparsity**: Higher sparsity ratios (closer to 1.0) indicate more aggressive attention pruning, which reduces memory usage but may impact model quality.


### Model Performance

**Memory Scaling**: Monitor how memory usage grows with sequence length. Linear growth is ideal; quadratic growth indicates attention bottlenecks.

**Training Efficiency**: Compare inference vs training latency ratios. Good implementations maintain reasonable training overhead.

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or sequence length
2. **Torch Compile Errors**: Profiling scripts disable torch.compile by default to avoid compilation issues
3. **Missing Dependencies**: Install matplotlib and seaborn for visualizations

### Performance Tips

1. **GPU Memory**: Profile on target hardware to understand real-world limits
2. **Sequence Length**: Start with shorter sequences and gradually increase
3. **Batch Size**: Use batch_size=1 for memory-limited GPUs
4. **Multiple Runs**: Run profiling multiple times and average results for stability

## Integration with Nano XYZ

The profiling suite is designed to work seamlessly with the Nano XYZ configuration system:

```python
from nano_xyz.configuration_nano import NanoConfig

# Create config for profiling
config = NanoConfig.from_preset("decoder_small")
config.attention_type = "sparse"

# Profile with this configuration
# (profiling scripts will automatically use config settings)
```

## Extending the Suite

### Adding New Metrics

1. Add metric calculation in the appropriate profiling script
2. Update the results JSON structure
3. Modify report generation to include new visualizations

### Custom Profiling Scenarios

1. Create new profiling script following the existing patterns
2. Add results to the report generator
3. Update the main runner script

## Requirements

- Python 3.8+
- PyTorch 2.1+
- CUDA (recommended for GPU profiling)
- matplotlib, seaborn, pandas (for visualizations)

Install visualization dependencies:
```bash
pip install matplotlib seaborn pandas
```
