"""
Profiling script for torchao dynamic quantization in Nano XYZ.

Benchmarks memory usage, latency, and throughput improvements from quantization.
Tests various configurations and sequence lengths on consumer hardware.
"""

import torch
import torch.profiler as profiler
import time
import json
from typing import Dict, List, Any
from contextlib import contextmanager

from nano_xyz.configuration_nano import NanoConfig, QuantizationConfig
from nano_xyz.model import NanoModel


@contextmanager
def cuda_memory_monitor():
    """Context manager to monitor CUDA memory usage."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        yield
        peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)  # GB
        print(f"Peak GPU memory: {peak_memory:.2f} GB")
    else:
        yield


class QuantizationProfiler:
    """Profiler for quantization performance benchmarking."""

    def __init__(self, preset: str = "decoder_tiny"):
        """
        Initialize profiler with model preset.

        Args:
            preset: Model configuration preset to use
        """
        self.preset = preset
        self.results = {}

    def create_models(self, quant_config: QuantizationConfig = None) -> tuple:
        """
        Create float and quantized model pair for comparison.

        Args:
            quant_config: Quantization configuration for quantized model

        Returns:
            Tuple of (float_model, quant_model, config)
        """
        # Float model
        float_config = NanoConfig.from_preset(self.preset)
        float_model = NanoModel(float_config)

        # Quantized model
        if quant_config:
            quant_config_copy = NanoConfig.from_preset(self.preset)
            quant_config_copy.quantization_config = quant_config
            quant_model = NanoModel(quant_config_copy)
        else:
            quant_model = None

        return float_model, quant_model, float_config

    def benchmark_inference(
        self,
        model: NanoModel,
        inputs: torch.Tensor,
        num_runs: int = 5,
        warmup_runs: int = 2
    ) -> Dict[str, float]:
        """
        Benchmark inference latency and throughput.

        Args:
            model: Model to benchmark
            inputs: Input tensor
            num_runs: Number of timing runs
            warmup_runs: Number of warmup runs

        Returns:
            Dictionary with latency and throughput metrics
        """
        model.eval()

        # Warmup
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = model(inputs)

        # Benchmark
        latencies = []
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.perf_counter()
                _ = model(inputs)
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                end_time = time.perf_counter()
                latencies.append(end_time - start_time)

        avg_latency = sum(latencies) / len(latencies)
        throughput = inputs.size(0) * inputs.size(1) / avg_latency  # tokens/second

        return {
            "avg_latency_ms": avg_latency * 1000,
            "throughput_tokens_per_sec": throughput,
            "min_latency_ms": min(latencies) * 1000,
            "max_latency_ms": max(latencies) * 1000,
        }

    def profile_memory_usage(
        self,
        model: NanoModel,
        inputs: torch.Tensor
    ) -> Dict[str, float]:
        """
        Profile memory usage during inference.

        Args:
            model: Model to profile
            inputs: Input tensor

        Returns:
            Dictionary with memory metrics
        """
        model.eval()

        with cuda_memory_monitor():
            with torch.no_grad():
                _ = model(inputs)

        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)  # GB
            return {"peak_memory_gb": peak_memory}
        else:
            return {"peak_memory_gb": 0.0}  # CPU placeholder

    def profile_with_torch_profiler(
        self,
        model: NanoModel,
        inputs: torch.Tensor,
        profile_name: str
    ) -> Dict[str, Any]:
        """
        Profile with torch.profiler for detailed metrics.

        Args:
            model: Model to profile
            inputs: Input tensor
            profile_name: Name for the profile output

        Returns:
            Dictionary with profiler metrics
        """
        model.eval()

        with profiler.profile(
            with_flops=True,
            profile_memory=True,
            with_cuda=torch.cuda.is_available(),
            record_shapes=True
        ) as prof:
            with torch.no_grad():
                _ = model(inputs)

        # Export chrome trace
        prof.export_chrome_trace(f"{profile_name}_trace.json")

        # Extract key metrics
        total_flops = prof.total_average().flops if hasattr(prof.total_average(), 'flops') else 0
        total_memory = prof.total_average().cuda_memory_usage if torch.cuda.is_available() else 0

        return {
            "total_flops": total_flops,
            "total_memory_bytes": total_memory,
            "profile_trace": f"{profile_name}_trace.json"
        }

    def run_comprehensive_benchmark(
        self,
        seq_lengths: List[int] = None,
        batch_sizes: List[int] = None,
        quant_configs: List[QuantizationConfig] = None
    ) -> Dict[str, Any]:
        """
        Run comprehensive quantization benchmark.

        Args:
            seq_lengths: Sequence lengths to test
            batch_sizes: Batch sizes to test
            quant_configs: Quantization configurations to test

        Returns:
            Dictionary with all benchmark results
        """
        if seq_lengths is None:
            seq_lengths = [256, 512, 1024]
        if batch_sizes is None:
            batch_sizes = [1, 4]
        if quant_configs is None:
            quant_configs = [
                QuantizationConfig(
                    method="torchao",
                    bits=8,
                    quant_type="int8_dyn_act_int4_weight",
                    group_size=32,
                    calibration_samples=10
                )
            ]

        results = {
            "benchmark_info": {
                "preset": self.preset,
                "torch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "device": "cuda" if torch.cuda.is_available() else "cpu"
            },
            "results": {}
        }

        # Move models to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for seq_len in seq_lengths:
            for batch_size in batch_sizes:
                print(f"\nBenchmarking seq_len={seq_len}, batch_size={batch_size}")

                # Create test inputs
                vocab_size = NanoConfig.from_preset(self.preset).vocab_size
                inputs = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)

                # Benchmark float model
                print("  Float model...")
                float_model, _, config = self.create_models()
                float_model = float_model.to(device)

                float_metrics = self.benchmark_inference(float_model, inputs)
                float_memory = self.profile_memory_usage(float_model, inputs)
                float_profile = self.profile_with_torch_profiler(
                    float_model, inputs, f"float_{seq_len}_{batch_size}"
                )

                # Benchmark quantized models
                for quant_config in quant_configs:
                    print(f"  Quantized model ({quant_config.quant_type})...")
                    _, quant_model, _ = self.create_models(quant_config)
                    quant_model = quant_model.to(device)

                    quant_metrics = self.benchmark_inference(quant_model, inputs)
                    quant_memory = self.profile_memory_usage(quant_model, inputs)
                    quant_profile = self.profile_with_torch_profiler(
                        quant_model, inputs, f"quant_{quant_config.quant_type}_{seq_len}_{batch_size}"
                    )

                    # Calculate improvements
                    speedup = float_metrics["avg_latency_ms"] / quant_metrics["avg_latency_ms"]
                    memory_reduction = (float_memory["peak_memory_gb"] - quant_memory["peak_memory_gb"]) / float_memory["peak_memory_gb"] * 100

                    key = f"{seq_len}_{batch_size}_{quant_config.quant_type}"
                    results["results"][key] = {
                        "seq_len": seq_len,
                        "batch_size": batch_size,
                        "quant_config": quant_config.dict(),
                        "float": {
                            **float_metrics,
                            **float_memory,
                            **float_profile
                        },
                        "quantized": {
                            **quant_metrics,
                            **quant_memory,
                            **quant_profile
                        },
                        "improvements": {
                            "speedup": speedup,
                            "memory_reduction_percent": memory_reduction
                        }
                    }

                    print(".2f"
        return results

    def save_results(self, results: Dict[str, Any], filename: str = "quantization_profile.json"):
        """Save benchmark results to JSON file."""
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results saved to {filename}")


def main():
    """Main profiling function."""
    import argparse

    parser = argparse.ArgumentParser(description="Profile torchao quantization performance")
    parser.add_argument("--preset", default="decoder_tiny", help="Model preset to use")
    parser.add_argument("--seq_lens", nargs="+", type=int, default=[256, 512],
                       help="Sequence lengths to test")
    parser.add_argument("--batch_sizes", nargs="+", type=int, default=[1],
                       help="Batch sizes to test")
    parser.add_argument("--output", default="quantization_profile.json",
                       help="Output JSON file")

    args = parser.parse_args()

    profiler = QuantizationProfiler(preset=args.preset)

    # Run comprehensive benchmark
    results = profiler.run_comprehensive_benchmark(
        seq_lengths=args.seq_lens,
        batch_sizes=args.batch_sizes
    )

    # Save results
    profiler.save_results(results, args.output)

    # Print summary
    print("\n" + "="*60)
    print("QUANTIZATION BENCHMARK SUMMARY")
    print("="*60)

    for key, result in results["results"].items():
        print(f"\n{key}:")
        print(".2f")
        print(".1f")


if __name__ == "__main__":
    main()
