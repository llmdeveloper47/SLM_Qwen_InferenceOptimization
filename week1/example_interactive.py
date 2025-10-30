"""
Interactive example script for Week 1 profiling
Run this for a guided walkthrough of profiling tasks
"""

import sys
from pathlib import Path
import torch
import pandas as pd
import json

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from utils.config_loader import load_config
from utils.model_utils import load_model_and_tokenizer, print_model_summary, get_model_info
from utils.profiler import InferenceProfiler


def main():
    """Interactive profiling example"""
    
    print("\n" + "="*70)
    print("WEEK 1: INTERACTIVE PROFILING EXAMPLE")
    print("="*70)
    
    # Step 1: Load config
    print("\n[Step 1] Loading configuration...")
    config = load_config()
    print(f"  ✓ Model: {config['model']['name']}")
    print(f"  ✓ Device: {config['model']['device']}")
    
    # Step 2: Load model
    print("\n[Step 2] Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(
        model_name=config['model']['name'],
        device=config['model']['device'],
        dtype=config['model']['dtype']
    )
    print_model_summary(model, tokenizer)
    
    # Step 3: Load sample data
    print("\n[Step 3] Loading sample data...")
    data_path = Path(config['dataset']['local_path'])
    
    if not data_path.exists():
        print(f"  ! Data not found at {data_path}")
        print(f"  ! Run: python scripts/01_download_data.py")
        return
    
    df = pd.read_csv(data_path)
    print(f"  ✓ Loaded {len(df):,} samples")
    
    # Step 4: Quick inference test
    print("\n[Step 4] Running quick inference test...")
    test_samples = df['text'].iloc[:5].tolist()
    
    inputs = tokenizer(
        test_samples,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    
    input_ids = inputs['input_ids'].to(config['model']['device'])
    attention_mask = inputs['attention_mask'].to(config['model']['device'])
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        predictions = torch.argmax(outputs.logits, dim=-1)
    
    print(f"  ✓ Inference successful on {len(test_samples)} samples")
    
    # Step 5: Profile batch sizes
    print("\n[Step 5] Profiling different batch sizes...")
    print("  This will take a few minutes...")
    
    profiler = InferenceProfiler(model, tokenizer, config['model']['device'])
    
    # Use subset
    sample_size = 100
    sample_texts = df['text'].iloc[:sample_size].tolist()
    
    batch_sizes = [1, 8, 32]  # Reduced for quick example
    
    results = profiler.profile_multiple_batch_sizes(
        texts=sample_texts,
        batch_sizes=batch_sizes,
        num_runs=5,  # Reduced for speed
        max_length=512
    )
    
    # Step 6: Show results
    print("\n[Step 6] Results summary:")
    print("\n" + "="*70)
    print(f"{'Batch Size':<12} {'Latency/Sample':<18} {'Throughput':<20} {'GPU Memory':<15}")
    print("-"*70)
    
    for result in results:
        bs = result['batch_size']
        lat = result['latency_per_sample_mean'] * 1000
        thr = result['throughput_samples_per_sec_mean']
        mem = result.get('gpu_mem_used_mean', 0)
        
        print(f"{bs:<12} {lat:.2f} ms{'':<11} {thr:.2f} samples/s{'':<6} {mem:.2f} MB")
    
    print("="*70)
    
    # Step 7: Recommendations
    print("\n[Step 7] Recommendations:")
    
    # Find optimal
    import numpy as np
    optimal_idx = np.argmax([r['throughput_samples_per_sec_mean'] for r in results])
    optimal = results[optimal_idx]
    
    print(f"  ✓ Optimal batch size: {optimal['batch_size']}")
    print(f"  ✓ Best throughput: {optimal['throughput_samples_per_sec_mean']:.2f} samples/sec")
    
    # Bottleneck hints
    if optimal['latency_per_sample_mean'] > 0.050:  # >50ms
        print("  ! Latency is high - consider optimizations in Week 2")
    
    if optimal.get('gpu_mem_used_mean', 0) > 10000:  # >10GB
        print("  ! High GPU memory usage - consider quantization")
    
    print("\n" + "="*70)
    print("INTERACTIVE EXAMPLE COMPLETE")
    print("="*70)
    print("\nNext steps:")
    print("  - Run full profiling: python run_week1.py")
    print("  - Check detailed results in: ./results/")
    print("  - Review week1_instructions.md for more info")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

