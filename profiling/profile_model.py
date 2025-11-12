#!/usr/bin/env python3
"""
Profile full Nano XYZ model performance.

This script benchmarks complete model forward/backward passes:
- Memory usage during training vs inference
- Latency for different sequence lengths
- Peak memory consumption
- Throughput measurements

Usage:
    python profiling/profile_model.py --presets decoder_tiny decoder_small --seq_lens 512 1024 2048
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
from nano_xyz.model import NanoModel


class ModelProfiler:
    """Profile complete Nano XYZ models."""

    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = torch.device(device)
        print(f"Using device: {self.device}")

    def create_model(self, preset: str) -> NanoModel:
        """Create model from preset."""
        config = NanoConfig.from_preset(preset)
        model = NanoModel(config)
        model.to(self.device)
        return model

    def create_test_inputs(self, batch_size: int, seq_len: int, vocab_size: int) -> Dict[str, torch.Tensor]:
        """Create test inputs for model profiling."""
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=self.device)
        attention_mask = torch.ones_like(input_ids)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
        }

    def profile_inference(self, model: NanoModel, inputs: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Profile model inference performance."""
        model.eval()

        with torch.no_grad():
            # Warmup
            for _ in range(3):
                model(**inputs)

            # Profile inference
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                profile_memory=True,
                record_shapes=True,
                with_stack=True
            ) as prof:
                with record_function("model_inference"):
                    start_time = time.time()
                    outputs = model(**inputs)
                    torch.cuda.synchronize() if self.device.type == "cuda" else None
                    end_time = time.time()

            # Analyze memory usage
            memory_stats = prof.key_averages()
            peak_memory = 0
            for event in memory_stats:
                if hasattr(event, 'cuda_memory_usage'):
                    peak_memory = max(peak_memory, event.cuda_memory_usage)

            return {
                'latency_ms': (end_time - start_time) * 1000,
                'peak_memory_mb': peak_memory / (1024**2) if peak_memory > 0 else 0,
                'output_shape': list(outputs.last_hidden_state.shape),
                'mode': 'inference',
            }

    def profile_training_step(self, model: NanoModel, inputs: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Profile single training step performance."""
        model.train()

        # Create targets for loss computation
        batch_size, seq_len = inputs['input_ids'].shape
        targets = torch.randint(0, model.lm_head.out_features, (batch_size, seq_len), device=self.device)

        # Training step
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        # Warmup
        for _ in range(2):
            optimizer.zero_grad()
            outputs = model(**inputs)
            logits = model.lm_head(outputs.last_hidden_state)
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100
            )
            loss.backward()
            optimizer.step()

        # Profile training step
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            profile_memory=True,
            record_shapes=True,
            with_stack=True
        ) as prof:
            with record_function("training_step"):
                start_time = time.time()

                optimizer.zero_grad()
                outputs = model(**inputs)
                logits = model.lm_head(outputs.last_hidden_state)

                # Compute loss
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1),
                    ignore_index=-100
                )

                total_loss = loss

                total_loss.backward()
                optimizer.step()

                torch.cuda.synchronize() if self.device.type == "cuda" else None
                end_time = time.time()

        # Analyze memory usage
        memory_stats = prof.key_averages()
        peak_memory = 0
        for event in memory_stats:
            if hasattr(event, 'cuda_memory_usage'):
                peak_memory = max(peak_memory, event.cuda_memory_usage)

        return {
            'latency_ms': (end_time - start_time) * 1000,
            'peak_memory_mb': peak_memory / (1024**2) if peak_memory > 0 else 0,
            'loss': loss.item(),
            'total_loss': total_loss.item(),
            'mode': 'training',
        }

    def estimate_model_size(self, model: NanoModel) -> Dict[str, Any]:
        """Estimate model size and parameter count."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Estimate memory usage
        param_memory = 0
        for p in model.parameters():
            param_memory += p.numel() * p.element_size()

        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'parameter_memory_mb': param_memory / (1024**2),
        }

    def run_profile(self, presets: List[str], seq_lens: List[int],
                   batch_size: int = 1, profile_training: bool = True) -> Dict[str, Any]:
        """Run comprehensive model profiling."""
        results = {
            'metadata': {
                'device': str(self.device),
                'batch_size': batch_size,
                'profile_training': profile_training,
            },
            'model_sizes': {},
            'results': []
        }

        print(f"Profiling Nano XYZ models with batch_size={batch_size}")
        print("=" * 80)

        for preset in presets:
            print(f"\nProfiling preset: {preset}")
            print("-" * 40)

            # Create model and get size info
            model = self.create_model(preset)
            size_info = self.estimate_model_size(model)
            results['model_sizes'][preset] = size_info

            print(f"  Model size: {size_info['total_parameters']:,} parameters "
                  ".1f")

            vocab_size = model.embed_tokens.num_embeddings

            for seq_len in seq_lens:
                print(f"  Sequence length: {seq_len}")
                inputs = self.create_test_inputs(batch_size, seq_len, vocab_size)

                # Profile inference
                try:
                    inference_result = self.profile_inference(model, inputs)
                    inference_result.update({
                        'preset': preset,
                        'seq_len': seq_len,
                        'batch_size': batch_size,
                    })
                    results['results'].append(inference_result)
                    print(".1f"
                except Exception as e:
                    print(f"    Inference ERROR: {e}")
                    results['results'].append({
                        'preset': preset,
                        'seq_len': seq_len,
                        'mode': 'inference',
                        'error': str(e),
                    })

                # Profile training if requested
                if profile_training:
                    try:
                        training_result = self.profile_training_step(model, inputs)
                        training_result.update({
                            'preset': preset,
                            'seq_len': seq_len,
                            'batch_size': batch_size,
                        })
                        results['results'].append(training_result)
                        print(".1f"
                    except Exception as e:
                        print(f"    Training ERROR: {e}")
                        results['results'].append({
                            'preset': preset,
                            'seq_len': seq_len,
                            'mode': 'training',
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
    parser = argparse.ArgumentParser(description="Profile Nano XYZ models")
    parser.add_argument('--presets', nargs='+',
                       default=['decoder_tiny', 'decoder_small'],
                       help='Model presets to profile')
    parser.add_argument('--seq_lens', nargs='+', type=int, default=[512, 1024, 2048],
                       help='Sequence lengths to profile')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for profiling')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cuda, cpu)')
    parser.add_argument('--no_training', action='store_true',
                       help='Skip training profiling')
    parser.add_argument('--output', type=str, default='profiling/results/model_profile.json',
                       help='Output file path')

    args = parser.parse_args()

    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    # Create profiler
    profiler = ModelProfiler(device)

    # Run profiling
    results = profiler.run_profile(
        args.presets,
        args.seq_lens,
        args.batch_size,
        not args.no_training
    )

    # Save results
    profiler.save_results(results, args.output)

    # Print summary
    print("\n" + "=" * 80)
    print("MODEL PROFILING SUMMARY")
    print("=" * 80)

    for preset, size_info in results['model_sizes'].items():
        print(f"\n{preset}:")
        print(f"  Parameters: {size_info['total_parameters']:,}")
        print(".1f")

    print(f"\nDetailed results saved to: {args.output}")
    print("\nModel profiling complete!")


if __name__ == "__main__":
    main()
