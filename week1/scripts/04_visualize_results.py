"""
Visualize profiling results
"""

import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.config_loader import load_config

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)


def plot_batch_size_comparison(results_df: pd.DataFrame, output_dir: str):
    """
    Plot comparison across different batch sizes
    
    Args:
        results_df: DataFrame with profiling results
        output_dir: Directory to save plots
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Batch Size Performance Comparison', fontsize=16, fontweight='bold')
    
    # 1. Latency per sample
    ax = axes[0, 0]
    ax.plot(results_df['batch_size'], results_df['latency_per_sample_mean'] * 1000, marker='o', linewidth=2)
    ax.fill_between(
        results_df['batch_size'],
        (results_df['latency_per_sample_mean'] - results_df['latency_per_sample_std']) * 1000,
        (results_df['latency_per_sample_mean'] + results_df['latency_per_sample_std']) * 1000,
        alpha=0.3
    )
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Latency (ms)')
    ax.set_title('Latency per Sample')
    ax.grid(True, alpha=0.3)
    
    # 2. Throughput
    ax = axes[0, 1]
    ax.plot(results_df['batch_size'], results_df['throughput_samples_per_sec_mean'], marker='o', linewidth=2, color='green')
    ax.fill_between(
        results_df['batch_size'],
        results_df['throughput_samples_per_sec_mean'] - results_df['throughput_samples_per_sec_std'],
        results_df['throughput_samples_per_sec_mean'] + results_df['throughput_samples_per_sec_std'],
        alpha=0.3,
        color='green'
    )
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Samples/Second')
    ax.set_title('Throughput')
    ax.grid(True, alpha=0.3)
    
    # 3. GPU Memory Usage
    ax = axes[0, 2]
    if 'gpu_mem_used_mean' in results_df.columns:
        ax.plot(results_df['batch_size'], results_df['gpu_mem_used_mean'], marker='o', linewidth=2, color='red')
        ax.plot(results_df['batch_size'], results_df['peak_gpu_mem_mean'], marker='s', linewidth=2, color='orange', linestyle='--', label='Peak')
        ax.set_xlabel('Batch Size')
        ax.set_ylabel('GPU Memory (MB)')
        ax.set_title('GPU Memory Usage')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'GPU metrics not available', ha='center', va='center', transform=ax.transAxes)
    
    # 4. CPU Memory Usage
    ax = axes[1, 0]
    ax.plot(results_df['batch_size'], results_df['cpu_mem_used_mean'], marker='o', linewidth=2, color='purple')
    ax.fill_between(
        results_df['batch_size'],
        results_df['cpu_mem_used_mean'] - results_df['cpu_mem_used_std'],
        results_df['cpu_mem_used_mean'] + results_df['cpu_mem_used_std'],
        alpha=0.3,
        color='purple'
    )
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('CPU Memory (MB)')
    ax.set_title('CPU Memory Usage')
    ax.grid(True, alpha=0.3)
    
    # 5. Latency per Batch
    ax = axes[1, 1]
    ax.plot(results_df['batch_size'], results_df['latency_per_batch_mean'] * 1000, marker='o', linewidth=2, color='brown')
    ax.fill_between(
        results_df['batch_size'],
        (results_df['latency_per_batch_mean'] - results_df['latency_per_batch_std']) * 1000,
        (results_df['latency_per_batch_mean'] + results_df['latency_per_batch_std']) * 1000,
        alpha=0.3,
        color='brown'
    )
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Latency (ms)')
    ax.set_title('Latency per Batch')
    ax.grid(True, alpha=0.3)
    
    # 6. Efficiency (Throughput / GPU Memory)
    ax = axes[1, 2]
    if 'gpu_mem_used_mean' in results_df.columns:
        efficiency = results_df['throughput_samples_per_sec_mean'] / (results_df['gpu_mem_used_mean'] + 1)  # +1 to avoid div by zero
        ax.plot(results_df['batch_size'], efficiency, marker='o', linewidth=2, color='teal')
        ax.set_xlabel('Batch Size')
        ax.set_ylabel('Throughput / Memory (samples/sec/MB)')
        ax.set_title('Memory Efficiency')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'GPU metrics not available', ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = Path(output_dir) / 'batch_size_comparison.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {plot_path}")
    plt.close()


def plot_metrics_heatmap(results_df: pd.DataFrame, output_dir: str):
    """
    Create heatmap of normalized metrics
    
    Args:
        results_df: DataFrame with profiling results
        output_dir: Directory to save plots
    """
    # Select key metrics
    metrics_to_plot = [
        'latency_per_sample_mean',
        'throughput_samples_per_sec_mean',
        'cpu_mem_used_mean',
    ]
    
    if 'gpu_mem_used_mean' in results_df.columns:
        metrics_to_plot.append('gpu_mem_used_mean')
    
    # Create subset and normalize
    plot_data = results_df[['batch_size'] + metrics_to_plot].copy()
    plot_data = plot_data.set_index('batch_size')
    
    # Normalize each metric to 0-1 range
    normalized = (plot_data - plot_data.min()) / (plot_data.max() - plot_data.min())
    
    # Create heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        normalized.T,
        annot=True,
        fmt='.3f',
        cmap='RdYlGn_r',
        cbar_kws={'label': 'Normalized Value (0-1)'},
        xticklabels=[f'BS={int(bs)}' for bs in plot_data.index],
        yticklabels=[col.replace('_mean', '').replace('_', ' ').title() for col in plot_data.columns]
    )
    plt.title('Normalized Metrics Heatmap Across Batch Sizes', fontsize=14, fontweight='bold')
    plt.xlabel('Batch Size')
    plt.ylabel('Metrics')
    plt.tight_layout()
    
    # Save
    heatmap_path = Path(output_dir) / 'metrics_heatmap.png'
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    print(f"Heatmap saved to: {heatmap_path}")
    plt.close()


def create_summary_report(baseline_path: str, output_dir: str):
    """
    Create comprehensive summary report
    
    Args:
        baseline_path: Path to baseline metrics JSON
        output_dir: Directory to save report
    """
    with open(baseline_path, 'r') as f:
        baseline = json.load(f)
    
    report_lines = []
    report_lines.append("="*70)
    report_lines.append("WEEK 1: BASELINE PROFILING SUMMARY REPORT")
    report_lines.append("="*70)
    report_lines.append("")
    
    # Model Information
    report_lines.append("MODEL INFORMATION")
    report_lines.append("-"*70)
    model_info = baseline['model_info']
    report_lines.append(f"Model: {baseline.get('model_info', {}).get('model_type', 'N/A')}")
    report_lines.append(f"Total Parameters: {model_info.get('total_params', 0):,} ({model_info.get('total_params_millions', 0):.2f}M)")
    report_lines.append(f"Model Size: {model_info.get('total_size_mb', 0):.2f} MB")
    report_lines.append("")
    
    # Dataset Information
    report_lines.append("DATASET INFORMATION")
    report_lines.append("-"*70)
    dataset_info = baseline['dataset_info']
    report_lines.append(f"Total Samples: {dataset_info.get('num_samples', 0):,}")
    report_lines.append(f"Number of Labels: {dataset_info.get('num_labels', 0)}")
    report_lines.append(f"Sample Size for Profiling: {dataset_info.get('sample_size_profiled', 0)}")
    report_lines.append("")
    
    # Full Dataset Performance
    report_lines.append("FULL DATASET PERFORMANCE")
    report_lines.append("-"*70)
    full_metrics = baseline['full_dataset_metrics']
    report_lines.append(f"Batch Size: {full_metrics.get('batch_size', 0)}")
    report_lines.append(f"Accuracy: {full_metrics.get('accuracy', 0):.4f}")
    report_lines.append(f"F1-Score (Macro): {full_metrics.get('f1_macro', 0):.4f}")
    report_lines.append(f"F1-Score (Weighted): {full_metrics.get('f1_weighted', 0):.4f}")
    report_lines.append(f"\nLatency:")
    report_lines.append(f"  Total: {full_metrics.get('latency_total', 0):.2f} seconds")
    report_lines.append(f"  Per Sample: {full_metrics.get('latency_per_sample', 0)*1000:.2f} ms")
    report_lines.append(f"\nThroughput: {full_metrics.get('throughput_samples_per_sec', 0):.2f} samples/sec")
    report_lines.append(f"\nMemory Usage:")
    report_lines.append(f"  CPU: {full_metrics.get('cpu_mem_used', 0):.2f} MB")
    report_lines.append(f"  GPU: {full_metrics.get('gpu_mem_used', 0):.2f} MB")
    report_lines.append(f"  Peak GPU: {full_metrics.get('peak_gpu_mem', 0):.2f} MB")
    report_lines.append("")
    
    # Batch Size Analysis
    report_lines.append("BATCH SIZE ANALYSIS")
    report_lines.append("-"*70)
    batch_results = baseline['batch_size_profiling']
    
    # Find optimal batch size (best throughput)
    optimal_idx = np.argmax([r['throughput_samples_per_sec_mean'] for r in batch_results])
    optimal_bs = batch_results[optimal_idx]['batch_size']
    
    report_lines.append(f"Tested Batch Sizes: {[r['batch_size'] for r in batch_results]}")
    report_lines.append(f"Optimal Batch Size (highest throughput): {optimal_bs}")
    report_lines.append("")
    
    for result in batch_results:
        bs = result['batch_size']
        report_lines.append(f"Batch Size {bs}:")
        report_lines.append(f"  Latency/sample: {result['latency_per_sample_mean']*1000:.2f} ± {result['latency_per_sample_std']*1000:.2f} ms")
        report_lines.append(f"  Throughput: {result['throughput_samples_per_sec_mean']:.2f} ± {result['throughput_samples_per_sec_std']:.2f} samples/sec")
        if 'gpu_mem_used_mean' in result:
            report_lines.append(f"  GPU Memory: {result['gpu_mem_used_mean']:.2f} ± {result['gpu_mem_used_std']:.2f} MB")
        report_lines.append("")
    
    report_lines.append("="*70)
    
    # Save report
    report_path = Path(output_dir) / 'baseline_summary_report.txt'
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print('\n'.join(report_lines))
    print(f"\nReport saved to: {report_path}")


def main():
    """Main visualization function"""
    config = load_config()
    
    results_dir = config['output']['results_dir']
    plots_dir = config['output']['plots_dir']
    
    # Create plots directory
    Path(plots_dir).mkdir(parents=True, exist_ok=True)
    
    # Load detailed metrics
    detailed_metrics_path = Path(results_dir) / config['output']['detailed_metrics_file']
    if not detailed_metrics_path.exists():
        print(f"Error: Detailed metrics file not found: {detailed_metrics_path}")
        print("Please run 02_baseline_inference.py first")
        return
    
    results_df = pd.read_csv(detailed_metrics_path)
    print(f"Loaded results for {len(results_df)} batch sizes")
    
    # Load baseline summary
    baseline_path = Path(results_dir) / config['output']['metrics_file']
    
    # Create visualizations
    print("\nGenerating visualizations...")
    
    # 1. Batch size comparison
    plot_batch_size_comparison(results_df, plots_dir)
    
    # 2. Metrics heatmap
    plot_metrics_heatmap(results_df, plots_dir)
    
    # 3. Create summary report
    if baseline_path.exists():
        create_summary_report(baseline_path, results_dir)
    
    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE")
    print("="*70)
    print(f"Plots saved in: {plots_dir}")
    print(f"Summary report: {results_dir}/baseline_summary_report.txt")
    print("="*70)


if __name__ == "__main__":
    main()

