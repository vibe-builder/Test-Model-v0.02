#!/usr/bin/env python3
"""
Complete Nano XYZ profiling suite runner.

This script runs all profiling tasks in sequence:
1. Attention mechanism profiling
2. Model performance profiling
3. Full model profiling
4. Report generation with visualizations

Usage:
    python profiling/run_profiling.py --quick  # Quick profiling for development
    python profiling/run_profiling.py --full   # Comprehensive profiling
"""

import argparse
import subprocess
import sys
from pathlib import Path


class ProfilingSuiteRunner:
    """Run the complete Nano XYZ profiling suite."""

    def __init__(self, base_dir: str = ".", output_dir: str = None):
        self.base_dir = Path(base_dir)
        self.profiling_dir = self.base_dir / "profiling"
        self.output_dir = Path(output_dir) if output_dir else self.profiling_dir / "results"

    def run_attention_profiling(self, quick: bool = False):
        """Run attention mechanism profiling."""
        print("\n" + "="*80)
        print("🔍 PROFILING ATTENTION MECHANISMS")
        print("="*80)

        cmd = [
            sys.executable,
            str(self.profiling_dir / "profile_attention.py"),
            "--output_dir", str(self.output_dir)
        ]

        if quick:
            # Quick profiling with smaller configs
            cmd.extend([
                "--seq_lens", "512", "1024",
                "--attention_types", "dense", "sparse",
                "--batch_size", "1"
            ])
        else:
            # Full profiling
            cmd.extend([
                "--seq_lens", "512", "1024", "2048", "4096",
                "--attention_types", "dense", "sparse",
                "--batch_size", "1"
            ])

        result = subprocess.run(cmd, cwd=self.base_dir)
        return result.returncode == 0


    def run_model_profiling(self, quick: bool = False):
        """Run full model profiling."""
        print("\n" + "="*80)
        print("🏗️ PROFILING FULL MODELS")
        print("="*80)

        cmd = [
            sys.executable,
            str(self.profiling_dir / "profile_model.py"),
        ]

        if quick:
            cmd.extend([
                "--presets", "decoder_tiny",
                "--seq_lens", "256", "512",
                "--batch_size", "1",
                "--no_training"  # Skip training for quick mode
            ])
        else:
            cmd.extend([
                "--presets", "decoder_tiny", "decoder_small",
                "--seq_lens", "256", "512", "1024",
                "--batch_size", "1"
            ])

        result = subprocess.run(cmd, cwd=self.base_dir)
        return result.returncode == 0

    def generate_reports(self):
        """Generate comprehensive profiling reports."""
        print("\n" + "="*80)
        print("📊 GENERATING PROFILING REPORTS")
        print("="*80)

        cmd = [
            sys.executable,
            str(self.profiling_dir / "generate_report.py"),
        ]

        result = subprocess.run(cmd, cwd=self.base_dir)
        return result.returncode == 0

    def run_full_suite(self, quick: bool = False):
        """Run the complete profiling suite."""
        print("🚀 Nano XYZ Profiling Suite")
        print("="*80)
        print(f"Mode: {'Quick (Development)' if quick else 'Full (Comprehensive)'}")
        print(f"Base directory: {self.base_dir}")
        print("="*80)

        success_count = 0
        total_steps = 3

        # Step 1: Attention profiling
        if self.run_attention_profiling(quick):
            success_count += 1
            print("✅ Attention profiling completed")
        else:
            print("❌ Attention profiling failed")

        # Step 2: Model profiling
        if self.run_model_profiling(quick):
            success_count += 1
            print("✅ Model profiling completed")
        else:
            print("❌ Model profiling failed")

        # Step 3: Report generation
        if self.generate_reports():
            success_count += 1
            print("✅ Report generation completed")
        else:
            print("❌ Report generation failed")

        # Final summary
        print("\n" + "="*80)
        print("🏁 PROFILING SUITE COMPLETE")
        print("="*80)
        print(f"Results: {success_count}/{total_steps} steps successful")

        if success_count == total_steps:
            print("🎉 All profiling tasks completed successfully!")
            print("\nGenerated files:")
            print("- profiling/results/attention_profile.json")
            print("- profiling/results/model_profile.json")
            print("- profiling/reports/*.png (visualizations)")
            print("- profiling/reports/*.csv (data tables)")
            print("- profiling/reports/*.md (markdown reports)")
        else:
            print(f"⚠️ {total_steps - success_count} profiling tasks failed")
            return False

        return True


def main():
    parser = argparse.ArgumentParser(description="Run Nano XYZ profiling suite")
    parser.add_argument('--quick', action='store_true',
                       help='Run quick profiling for development (smaller configs)')
    parser.add_argument('--full', action='store_true',
                       help='Run full comprehensive profiling (default)')
    parser.add_argument('--base_dir', type=str, default='.',
                       help='Base directory containing the nano_xyz project')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for profiling artifacts')

    args = parser.parse_args()

    # Determine mode
    quick_mode = args.quick or not args.full

    # Create and run profiling suite
    runner = ProfilingSuiteRunner(args.base_dir, args.output_dir)

    success = runner.run_full_suite(quick_mode)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
