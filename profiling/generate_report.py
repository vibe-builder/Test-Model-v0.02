#!/usr/bin/env python3
"""
Generate profiling reports and visualizations.

This script analyzes profiling data and generates:
- Performance comparison tables
- Memory usage charts
- Latency vs sequence length plots
- Efficiency analysis reports

Usage:
    python profiling/generate_report.py --input_dir profiling/results/ --output_dir profiling/reports/
"""

import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")


class ProfilingReportGenerator:
    """Generate comprehensive profiling reports and visualizations."""

    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load all profiling data
        self.attention_data = self._load_json("attention_profile.json")
        self.model_data = self._load_json("model_profile.json")

    def _load_json(self, filename: str) -> Dict[str, Any]:
        """Load JSON file if it exists."""
        filepath = self.input_dir / filename
        if filepath.exists():
            with open(filepath, 'r') as f:
                return json.load(f)
        return {}

    def generate_attention_report(self):
        """Generate attention mechanism comparison report."""
        if not self.attention_data or 'results' not in self.attention_data:
            print("No attention profiling data found")
            return

        df = pd.DataFrame(self.attention_data['results'])

        # Filter out error entries
        df = df[df['error'].isna()]

        if df.empty:
            print("No valid attention profiling data")
            return

        # Create comparison plots
        self._plot_attention_latency(df)
        self._plot_attention_memory(df)
        self._plot_attention_efficiency(df)

        # Generate summary table
        self._generate_attention_summary(df)

    def _plot_attention_latency(self, df: pd.DataFrame):
        """Plot latency vs sequence length for different attention types."""
        plt.figure(figsize=(12, 8))

        for attn_type in df['attention_type'].unique():
            type_data = df[df['attention_type'] == attn_type]
            plt.plot(type_data['seq_len'], type_data['latency_ms'],
                    marker='o', label=attn_type, linewidth=2)

        plt.xlabel('Sequence Length')
        plt.ylabel('Latency (ms)')
        plt.title('Attention Latency vs Sequence Length')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xscale('log')
        plt.yscale('log')
        plt.tight_layout()

        plt.savefig(self.output_dir / 'attention_latency.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_attention_memory(self, df: pd.DataFrame):
        """Plot memory usage vs sequence length."""
        plt.figure(figsize=(12, 8))

        for attn_type in df['attention_type'].unique():
            type_data = df[df['attention_type'] == attn_type]
            plt.plot(type_data['seq_len'], type_data['peak_memory_mb'],
                    marker='s', label=attn_type, linewidth=2)

        plt.xlabel('Sequence Length')
        plt.ylabel('Peak Memory (MB)')
        plt.title('Attention Memory Usage vs Sequence Length')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xscale('log')
        plt.yscale('log')
        plt.tight_layout()

        plt.savefig(self.output_dir / 'attention_memory.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_attention_efficiency(self, df: pd.DataFrame):
        """Plot attention efficiency metrics."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Sparsity ratio
        sparse_data = df[df['attention_type'] != 'dense']
        if not sparse_data.empty:
            sparsity_pivot = sparse_data.pivot(index='seq_len', columns='attention_type', values='sparsity_ratio')
            sparsity_pivot.plot(kind='bar', ax=axes[0])
            axes[0].set_title('Attention Sparsity Ratio')
            axes[0].set_xlabel('Sequence Length')
            axes[0].set_ylabel('Sparsity Ratio')
            axes[0].tick_params(axis='x', rotation=45)

        # Memory efficiency (dense memory / sparse memory)
        efficiency_data = []
        for seq_len in df['seq_len'].unique():
            seq_data = df[df['seq_len'] == seq_len]
            dense_memory = seq_data[seq_data['attention_type'] == 'dense']['peak_memory_mb'].iloc[0] if not seq_data[seq_data['attention_type'] == 'dense'].empty else 0

            for _, row in seq_data[seq_data['attention_type'] != 'dense'].iterrows():
                if dense_memory > 0:
                    efficiency = dense_memory / row['peak_memory_mb']
                    efficiency_data.append({
                        'seq_len': seq_len,
                        'attention_type': row['attention_type'],
                        'memory_efficiency': efficiency
                    })

        if efficiency_data:
            eff_df = pd.DataFrame(efficiency_data)
            eff_pivot = eff_df.pivot(index='seq_len', columns='attention_type', values='memory_efficiency')
            eff_pivot.plot(kind='bar', ax=axes[1])
            axes[1].set_title('Memory Efficiency (Dense Memory / Sparse Memory)')
            axes[1].set_xlabel('Sequence Length')
            axes[1].set_ylabel('Memory Efficiency')
            axes[1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'attention_efficiency.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _generate_attention_summary(self, df: pd.DataFrame):
        """Generate attention profiling summary table."""
        summary = []

        for attn_type in df['attention_type'].unique():
            type_data = df[df['attention_type'] == attn_type]

            summary.append({
                'Attention Type': attn_type,
                'Avg Latency (ms)': ".2f",
                'Avg Memory (MB)': ".2f",
                'Max Memory (MB)': ".2f",
                'Avg Sparsity': ".3f" if attn_type != 'dense' and 'sparsity_ratio' in type_data.columns else 'N/A',
            })

        summary_df = pd.DataFrame(summary)
        summary_df.to_csv(self.output_dir / 'attention_summary.csv', index=False)

        # Also save as markdown table
        with open(self.output_dir / 'attention_summary.md', 'w') as f:
            f.write("# Attention Profiling Summary\n\n")
            f.write(summary_df.to_markdown(index=False))
            f.write("\n")



    def generate_model_report(self):
        """Generate full model comparison report."""
        if not self.model_data or 'results' not in self.model_data:
            print("No model profiling data found")
            return

        df = pd.DataFrame(self.model_data['results'])

        # Filter out error entries
        df = df[df['error'].isna()]

        if df.empty:
            print("No valid model profiling data")
            return

        # Create comparison plots
        self._plot_model_performance(df)

        # Generate summary table
        self._generate_model_summary(df, self.model_data.get('model_sizes', {}))

    def _plot_model_performance(self, df: pd.DataFrame):
        """Plot model performance metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Latency comparison
        latency_data = df.pivot_table(
            index=['preset', 'seq_len'],
            columns='mode',
            values='latency_ms',
            aggfunc='mean'
        ).reset_index()

        for mode in ['inference', 'training']:
            if mode in latency_data.columns:
                for preset in df['preset'].unique():
                    preset_data = latency_data[latency_data['preset'] == preset]
                    axes[0, 0].plot(preset_data['seq_len'], preset_data[mode],
                                   marker='o', label=f'{preset} ({mode})', linewidth=2)

        axes[0, 0].set_xlabel('Sequence Length')
        axes[0, 0].set_ylabel('Latency (ms)')
        axes[0, 0].set_title('Model Latency')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_xscale('log')
        axes[0, 0].set_yscale('log')

        # Memory usage
        memory_data = df.pivot_table(
            index=['preset', 'seq_len'],
            columns='mode',
            values='peak_memory_mb',
            aggfunc='mean'
        ).reset_index()

        for mode in ['inference', 'training']:
            if mode in memory_data.columns:
                for preset in df['preset'].unique():
                    preset_data = memory_data[memory_data['preset'] == preset]
                    axes[0, 1].plot(preset_data['seq_len'], preset_data[mode],
                                   marker='s', label=f'{preset} ({mode})', linewidth=2)

        axes[0, 1].set_xlabel('Sequence Length')
        axes[0, 1].set_ylabel('Peak Memory (MB)')
        axes[0, 1].set_title('Model Memory Usage')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Training loss (if available)
        loss_data = df[df['mode'] == 'training']
        if not loss_data.empty:
            for preset in loss_data['preset'].unique():
                preset_data = loss_data[loss_data['preset'] == preset]
                axes[1, 0].plot(preset_data['seq_len'], preset_data['loss'],
                               marker='^', label=preset, linewidth=2)

            axes[1, 0].set_xlabel('Sequence Length')
            axes[1, 0].set_ylabel('Training Loss')
            axes[1, 0].set_title('Training Loss vs Sequence Length')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)


        plt.tight_layout()
        plt.savefig(self.output_dir / 'model_performance.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _generate_model_summary(self, df: pd.DataFrame, model_sizes: Dict[str, Any]):
        """Generate model profiling summary table."""
        summary = []

        for preset in df['preset'].unique():
            preset_data = df[df['preset'] == preset]
            size_info = model_sizes.get(preset, {})

            # Calculate averages
            inference_data = preset_data[preset_data['mode'] == 'inference']
            training_data = preset_data[preset_data['mode'] == 'training']

            summary.append({
                'Preset': preset,
                'Parameters': f"{size_info.get('total_parameters', 0):,}",
                'Model Memory (MB)': ".1f",
                'Avg Inference Latency (ms)': ".1f" if not inference_data.empty else 'N/A',
                'Avg Training Latency (ms)': ".1f" if not training_data.empty else 'N/A',
                'Avg Inference Memory (MB)': ".1f" if not inference_data.empty else 'N/A',
                'Avg Training Memory (MB)': ".1f" if not training_data.empty else 'N/A',
            })

        summary_df = pd.DataFrame(summary)
        summary_df.to_csv(self.output_dir / 'model_summary.csv', index=False)

        # Also save as markdown table
        with open(self.output_dir / 'model_summary.md', 'w') as f:
            f.write("# Model Profiling Summary\n\n")
            f.write(summary_df.to_markdown(index=False))
            f.write("\n")

    def generate_comprehensive_report(self):
        """Generate a comprehensive profiling report."""
        print("Generating comprehensive profiling report...")

        # Generate individual reports
        self.generate_attention_report()
        self.generate_model_report()

        # Generate overall summary
        self._generate_overall_summary()

        print(f"Reports generated in: {self.output_dir}")

    def _generate_overall_summary(self):
        """Generate overall profiling summary."""
        with open(self.output_dir / 'profiling_report.md', 'w') as f:
            f.write("# Nano XYZ Profiling Report\n\n")

            f.write("## Overview\n\n")
            f.write("This report contains comprehensive performance analysis of the Nano XYZ transformer implementation, ")
            f.write("including attention mechanisms and full model performance.\n\n")

            f.write("## Key Findings\n\n")

            # Add attention findings
            if self.attention_data and 'results' in self.attention_data:
                df = pd.DataFrame(self.attention_data['results'])
                df = df[df['error'].isna()]

                if not df.empty:
                    f.write("### Attention Mechanisms\n\n")
                    dense_data = df[df['attention_type'] == 'dense']
                    sparse_data = df[df['attention_type'] != 'dense']

                    if not dense_data.empty and not sparse_data.empty:
                        avg_dense_mem = dense_data['peak_memory_mb'].mean()
                        avg_sparse_mem = sparse_data['peak_memory_mb'].mean()
                        memory_savings = (avg_dense_mem - avg_sparse_mem) / avg_dense_mem * 100

                        f.write(".1f")
                        f.write(".1f")
                        f.write(f"- **Memory Efficiency**: {memory_savings:.1f}% reduction in peak memory usage\n")


            f.write("\n## Recommendations\n\n")
            f.write("1. **For long sequences (>4K tokens)**: Use sparse attention with DCA patterns\n")
            f.write("2. **For memory-constrained GPUs**: Use sparse attention patterns and gradient checkpointing\n")
            f.write("3. **For training**: Use gradient checkpointing with sparse attention\n")
            f.write("4. **For inference**: Use torch.compile for additional speedups\n\n")

            f.write("## Files Generated\n\n")
            f.write("- `attention_latency.png`: Attention latency comparison\n")
            f.write("- `attention_memory.png`: Attention memory usage\n")
            f.write("- `attention_efficiency.png`: Sparsity and efficiency metrics\n")
            f.write("- `model_performance.png`: Full model performance metrics\n")
            f.write("- `*_summary.csv/md`: Detailed performance tables\n")


def main():
    parser = argparse.ArgumentParser(description="Generate Nano XYZ profiling reports")
    parser.add_argument('--input_dir', type=str, default='profiling/results',
                       help='Input directory containing profiling JSON files')
    parser.add_argument('--output_dir', type=str, default='profiling/reports',
                       help='Output directory for reports and visualizations')

    args = parser.parse_args()

    # Create report generator
    generator = ProfilingReportGenerator(args.input_dir, args.output_dir)

    # Generate comprehensive report
    generator.generate_comprehensive_report()


if __name__ == "__main__":
    main()
