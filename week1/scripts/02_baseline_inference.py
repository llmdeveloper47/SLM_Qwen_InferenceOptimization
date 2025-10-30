"""
Baseline inference script - establishes performance baseline
"""

import sys
from pathlib import Path
import pandas as pd
import torch
import json
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.config_loader import load_config
from utils.model_utils import load_model_and_tokenizer, print_model_summary, get_model_info
from utils.profiler import InferenceProfiler


def load_data(config: dict):
    """
    Load validation dataset
    
    Args:
        config: Configuration dictionary
        
    Returns:
        tuple: (texts, labels, label2id, id2label)
    """
    local_path = config['dataset']['local_path']
    
    print("="*70)
    print("LOADING DATASET")
    print("="*70)
    print(f"Loading from: {local_path}")
    
    # Load CSV
    df = pd.read_csv(local_path)
    
    print(f"Loaded {len(df):,} samples")
    print(f"Columns: {list(df.columns)}")
    
    # Load label mappings
    label_mapping_path = Path(local_path).parent / "label_mappings.json"
    with open(label_mapping_path, 'r') as f:
        mappings = json.load(f)
    
    label2id = mappings['label2id']
    id2label = {int(k): v for k, v in mappings['id2label'].items()}
    
    texts = df[config['dataset']['text_column']].tolist()
    labels = df['label_id'].tolist()
    
    print(f"Number of labels: {mappings['num_labels']}")
    print("="*70 + "\n")
    
    return texts, labels, label2id, id2label


def run_baseline_inference(config: dict):
    """
    Run baseline inference and profiling
    
    Args:
        config: Configuration dictionary
    """
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(
        model_name=config['model']['name'],
        device=config['model']['device'],
        dtype=config['model']['dtype']
    )
    
    # Print model summary
    model_info = get_model_info(model, tokenizer)
    print_model_summary(model, tokenizer)
    
    # Load data
    texts, labels, label2id, id2label = load_data(config)
    
    # Initialize profiler
    profiler = InferenceProfiler(model, tokenizer, config['model']['device'])
    
    # Sample subset for profiling
    sample_size = config['dataset']['sample_sizes'][0]  # Start with smallest
    sample_texts = texts[:sample_size]
    sample_labels = labels[:sample_size]
    
    print(f"Using {sample_size} samples for initial profiling\n")
    
    # Profile different batch sizes
    batch_sizes = config['profiling']['batch_sizes']
    num_runs = config['profiling']['num_measurement_runs']
    max_length = config['profiling']['max_length']
    
    results = profiler.profile_multiple_batch_sizes(
        texts=sample_texts,
        batch_sizes=batch_sizes,
        num_runs=num_runs,
        max_length=max_length
    )
    
    # Save detailed results
    results_df = pd.DataFrame(results)
    results_path = Path(config['output']['results_dir']) / config['output']['detailed_metrics_file']
    results_df.to_csv(results_path, index=False)
    print(f"\nDetailed metrics saved to: {results_path}")
    
    # Run full dataset inference with best batch size (typically mid-range)
    optimal_batch_size = batch_sizes[len(batch_sizes) // 2]  # Middle batch size
    print(f"\n\nRunning full dataset inference with batch size {optimal_batch_size}...")
    
    full_metrics = profiler.profile_full_dataset(
        texts=texts,
        labels=labels,
        batch_size=optimal_batch_size,
        max_length=max_length
    )
    
    # Calculate detailed accuracy metrics
    predictions = full_metrics['predictions']
    accuracy = accuracy_score(labels, predictions)
    f1_macro = f1_score(labels, predictions, average='macro')
    f1_weighted = f1_score(labels, predictions, average='weighted')
    
    print(f"\n" + "="*70)
    print("FULL DATASET EVALUATION")
    print("="*70)
    print(f"Samples: {len(texts):,}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-Score (Macro): {f1_macro:.4f}")
    print(f"F1-Score (Weighted): {f1_weighted:.4f}")
    print(f"Inference time: {full_metrics['latency_total']:.2f} seconds")
    print(f"Throughput: {full_metrics['throughput_samples_per_sec']:.2f} samples/sec")
    print("="*70)
    
    # Compile baseline summary
    baseline_summary = {
        'model_info': model_info,
        'dataset_info': {
            'num_samples': len(texts),
            'num_labels': len(label2id),
            'sample_size_profiled': sample_size,
        },
        'batch_size_profiling': results,
        'full_dataset_metrics': {
            'batch_size': optimal_batch_size,
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'latency_total': full_metrics['latency_total'],
            'latency_per_sample': full_metrics['latency_per_sample'],
            'throughput_samples_per_sec': full_metrics['throughput_samples_per_sec'],
            'cpu_mem_used': full_metrics.get('cpu_mem_used', 0),
            'gpu_mem_used': full_metrics.get('gpu_mem_used', 0),
            'peak_gpu_mem': full_metrics.get('peak_gpu_mem', 0),
        }
    }
    
    # Save baseline summary
    baseline_path = Path(config['output']['results_dir']) / config['output']['metrics_file']
    with open(baseline_path, 'w') as f:
        json.dump(baseline_summary, f, indent=2, default=str)
    
    print(f"\nBaseline summary saved to: {baseline_path}")
    
    return baseline_summary


if __name__ == "__main__":
    # Load configuration
    config = load_config()
    
    # Run baseline inference
    baseline_summary = run_baseline_inference(config)

