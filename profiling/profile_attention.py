#!/usr/bin/env python3
"""
Profile attention mechanisms in Nano XYZ.

This script benchmarks the performance of different attention implementations:
- Dense attention (standard PyTorch SDPA)
- Sparse attention (DCA + pattern-based)
- Memory usage and latency for various sequence lengths

Usage:
    python profiling/profile_attention.py --seq_lens 1024 4096 8192 --attention_types dense sparse
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Any
import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity

# Add the nano_xyz module to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from nano_xyz.configuration_nano import NanoConfig
from nano_xyz.attention_utils import SparsePatternGenerator, apply_sparse_attention_optimization
from nano_xyz.model import AttentionLayer, RotaryEmbedding


class AttentionProfiler:
    """Profile attention mechanisms with memory and latency tracking."""

    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = torch.device(device)
        print(f"Using device: {self.device}")

        # Create a standard config for profiling
        self.config = NanoConfig.from_preset("decoder_medium")  # Has DCA enabled

    def create_test_inputs(self, batch_size: int, seq_len: int, n_heads: int, head_dim: int) -> Dict[str, torch.Tensor]:
        """Create test inputs for attention profiling."""
        return {
            'query': torch.randn(batch_size, n_heads, seq_len, head_dim, device=self.device),
            'key': torch.randn(batch_size, n_heads, seq_len, head_dim, device=self.device),
            'value': torch.randn(batch_size, n_heads, seq_len, head_dim, device=self.device),
        }

    def create_attention_layer(self, attention_type: str) -> AttentionLayer:
        """Create attention layer with specified type."""
        config = self.config.copy()
        # Config already has DCA enabled via preset

        # Create minimal attention layer for profiling
        layer = AttentionLayer(config, layer_idx=0)
        layer.to(self.device)
        layer.eval()
        return layer

    def profile_dense_attention(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> Dict[str, Any]:
        """Profile dense attention using PyTorch SDPA."""
        with torch.no_grad():
            # Warmup
            for _ in range(5):
                torch.nn.functional.scaled_dot_product_attention(query, key, value)

            # Profile
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                profile_memory=True,
                record_shapes=True,
                with_stack=True
            ) as prof:
                with record_function("dense_attention"):
                    start_time = time.time()
                    output = torch.nn.functional.scaled_dot_product_attention(query, key, value)
                    torch.cuda.synchronize() if self.device.type == "cuda" else None
                    end_time = time.time()

            # Memory peak
            memory_stats = prof.key_averages()
            peak_memory = 0
            for event in memory_stats:
                if hasattr(event, 'cuda_memory_usage'):
                    peak_memory = max(peak_memory, event.cuda_memory_usage)

            return {
                'latency_ms': (end_time - start_time) * 1000,
                'peak_memory_mb': peak_memory / (1024**2) if peak_memory > 0 else 0,
                'output_shape': list(output.shape),
                'attention_type': 'dense',
                'flops': self._estimate_flops(query.shape[-2], query.shape[-1]),
            }

    def profile_sparse_attention(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                               attention_type: str) -> Dict[str, Any]:
        """Profile sparse attention with DCA patterns."""
        batch_size, n_heads, seq_len, head_dim = query.shape

        # Create sparse pattern generator
        config = self.config.copy()
        # Config already has DCA parameters set
        pattern_gen = SparsePatternGenerator(config)

        # Generate jagged pattern
        pattern = pattern_gen(seq_len, self.device)
        sparsity = pattern["sparsity"]

        with torch.no_grad():
            # Warmup
            for _ in range(5):
                apply_sparse_attention_optimization(query, key, value, pattern)

            # Profile
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                profile_memory=True,
                record_shapes=True,
                with_stack=True
            ) as prof:
                with record_function("jagged_attention"):
                    start_time = time.time()
                    output = apply_sparse_attention_optimization(query, key, value, pattern, is_causal=True)
                    torch.cuda.synchronize() if self.device.type == "cuda" else None
                    end_time = time.time()

            # Memory peak
            memory_stats = prof.key_averages()
            peak_memory = 0
            for event in memory_stats:
                if hasattr(event, 'cuda_memory_usage'):
                    peak_memory = max(peak_memory, event.cuda_memory_usage)

            return {
                'latency_ms': (end_time - start_time) * 1000,
                'peak_memory_mb': peak_memory / (1024**2) if peak_memory > 0 else 0,
                'output_shape': list(output.shape),
                'attention_type': attention_type,
                'sparsity_ratio': sparsity,
                'connections': pattern["col_indices"].numel(),
                'flops': self._estimate_sparse_flops(seq_len, head_dim, sparsity),
            }

    def _estimate_flops(self, seq_len: int, head_dim: int) -> float:
        """Estimate FLOPs for dense attention: O(seq_len² * head_dim)."""
        return seq_len ** 2 * head_dim * 2  # QK^T + softmax + V aggregation

    def _estimate_sparse_flops(self, seq_len: int, head_dim: int, sparsity: float) -> float:
        """Estimate FLOPs for sparse attention."""
        connections = int(seq_len * seq_len * (1 - sparsity))
        return connections * head_dim * 2  # Sparse connections * head_dim

    def run_profile(self, seq_lens: List[int], attention_types: List[str],
                   batch_size: int = 1) -> Dict[str, Any]:
        """Run comprehensive profiling across configurations."""
        results = {
            'metadata': {
                'device': str(self.device),
                'batch_size': batch_size,
                'config': self.config.dict(),
            },
            'results': []
        }

        print(f"Profiling attention mechanisms with batch_size={batch_size}")
        print("=" * 80)

        for seq_len in seq_lens:
            print(f"\nProfiling sequence length: {seq_len}")
            print("-" * 40)

            # Create test inputs
            n_heads = self.config.n_head
            head_dim = self.config.n_embd // self.config.n_head
            inputs = self.create_test_inputs(batch_size, seq_len, n_heads, head_dim)

            for attn_type in attention_types:
                print(f"  Testing {attn_type} attention...")

                try:
                    if attn_type == "dense":
                        result = self.profile_dense_attention(
                            inputs['query'], inputs['key'], inputs['value']
                        )
                    else:
                        result = self.profile_sparse_attention(
                            inputs['query'], inputs['key'], inputs['value'], attn_type
                        )

                    result.update({
                        'seq_len': seq_len,
                        'batch_size': batch_size,
                        'n_heads': n_heads,
                        'head_dim': head_dim,
                    })

                    results['results'].append(result)
                    print(f"    + {attn_type}: {result['latency_ms']:.1f}ms, {result['peak_memory_mb']:.1f}MB")
                except Exception as e:
                    print(f"    ERROR: {e}")
                    results['results'].append({
                        'seq_len': seq_len,
                        'attention_type': attn_type,
                        'error': str(e),
                    })

        return results

    def save_results(self, results: Dict[str, Any], output_path: str):
        """Save profiling results to JSON file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\nResults saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Profile Nano XYZ attention mechanisms")
    parser.add_argument('--seq_lens', nargs='+', type=int, default=[1024, 4096, 8192],
                       help='Sequence lengths to profile')
    parser.add_argument('--attention_types', nargs='+',
                       default=['dense', 'sparse'],
                       help='Attention types to profile')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for profiling')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cuda, cpu)')
    parser.add_argument('--output', type=str, default='profiling/results/attention_profile.json',
                       help='Output file path')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (alternative to --output)')

    args = parser.parse_args()

    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    # Create profiler
    profiler = AttentionProfiler(device)

    # Run profiling
    results = profiler.run_profile(args.seq_lens, args.attention_types, args.batch_size)

    # Save results
    if args.output_dir:
        import os
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, 'attention_profile.json')
    else:
        output_path = args.output
    profiler.save_results(results, output_path)

    # Print summary
    print("\n" + "=" * 80)
    print("PROFILING SUMMARY")
    print("=" * 80)

    for result in results['results']:
        if 'error' not in result:
            print(f"{result['seq_len']:5d} tokens | {result['attention_type']:8s} | "
                  ".1f"
                  ".1f")

    print("\nProfiling complete!")


def profile_jagged_attention(seq_lens=[512, 1024, 4096, 100000], batch_size=1, n_heads=8, head_dim=64, device='cuda'):
    """
    Profile jagged tensor sparse attention performance and memory usage.

    Tests memory efficiency gains from jagged packing and measures FLOPs
    for different sequence lengths targeting consumer hardware limits.

    Args:
        seq_lens: Sequence lengths to test (including 100K for consumer limits)
        batch_size: Batch size for profiling
        n_heads: Number of attention heads
        head_dim: Head dimension
        device: Device to profile on

    Returns:
        Dictionary with profiling results per sequence length
    """
    if not torch.cuda.is_available() and device == 'cuda':
        device = 'cpu'  # Consumer fallback
        print("Warning: CUDA not available, falling back to CPU")

    config = NanoConfig.from_preset("decoder_tiny")
    gen = SparsePatternGenerator(config)

    results = {}

    for seq_len in seq_lens:
        print(f"Profiling sequence length: {seq_len}")

        pattern = gen(seq_len, torch.device(device))

        # Create test inputs
        q = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device)
        k = v = q.clone()

        # Profile with comprehensive metrics
        with profiler.profile(
            with_flops=True,
            profile_memory=True,
            with_cuda=(device=='cuda'),
            record_shapes=True
        ) as prof:
            out = apply_sparse_attention_optimization(q, k, v, pattern, is_causal=True)

        # Export trace for detailed analysis
        prof.export_chrome_trace(f"jagged_attention_{seq_len}.json")

        # Extract metrics
        avg_stats = prof.key_averages()

        # Calculate memory usage
        if device == 'cuda':
            memory_mb = prof.total_average().cuda_memory_usage / (1024**2)
            latency_ms = prof.total_average().self_cuda_time_total / 1000
        else:
            memory_mb = prof.total_average().cpu_memory_usage / (1024**2)
            latency_ms = prof.total_average().self_cpu_time_total / 1000

        # Calculate theoretical FLOPs (attention computation)
        # Q*K^T: seq_len² * head_dim * n_heads * 2 (for QK and softmax)
        # Q*V: seq_len² * head_dim * n_heads
        theoretical_flops = seq_len**2 * head_dim * n_heads * 3  # Rough estimate

        results[seq_len] = {
            "flops": prof.total_average().flops if hasattr(prof.total_average(), 'flops') else theoretical_flops,
            "theoretical_flops": theoretical_flops,
            "memory_mb": memory_mb,
            "latency_ms": latency_ms,
            "sparsity": pattern["sparsity"],
            "connections": pattern["col_indices"].numel(),
            "total_possible": seq_len * seq_len,
            "memory_efficiency": (1.0 - pattern["sparsity"]) * 100,  # % reduction
        }

        print(f"  Sparsity: {pattern['sparsity']:.3f}, "
              f"Memory: {memory_mb:.1f}MB, "
              f"Latency: {latency_ms:.1f}ms, "
              f"Connections: {pattern['col_indices'].numel()}")

    # Save results
    with open("jagged_attention_profile.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nProfiling complete. Results saved to jagged_attention_profile.json")

    # Validate consumer hardware targets
    max_seq = max(seq_lens)
    if max_seq >= 100000 and device == 'cuda':
        max_memory = results[max_seq]["memory_mb"]
        if max_memory > 16000:  # 16GB limit
            print(f"⚠️  Warning: {max_memory:.1f}MB exceeds 16GB consumer GPU limit")
        else:
            print(f"✅ Memory usage acceptable for consumer GPUs: {max_memory:.1f}MB")

    return results


if __name__ == "__main__":
    main()
