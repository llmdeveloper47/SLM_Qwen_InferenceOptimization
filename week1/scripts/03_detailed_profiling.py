"""
Detailed profiling script - identifies bottlenecks using PyTorch profiler
"""

import sys
from pathlib import Path
import pandas as pd
import torch
import json
import time
from torch.profiler import profile, ProfilerActivity, record_function

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.config_loader import load_config
from utils.model_utils import load_model_and_tokenizer, print_model_summary, get_model_info


def profile_with_pytorch_profiler(
    model, 
    tokenizer, 
    texts, 
    batch_size=32, 
    max_length=512,
    output_dir="./results/profiler_traces"
):
    """
    Use PyTorch profiler to identify bottlenecks
    
    Args:
        model: PyTorch model
        tokenizer: HuggingFace tokenizer
        texts: Sample texts
        batch_size: Batch size
        max_length: Max sequence length
        output_dir: Directory to save profiler traces
    """
    print("="*70)
    print("PYTORCH PROFILER - DETAILED ANALYSIS")
    print("="*70)
    
    device = next(model.parameters()).device
    
    # Prepare batch
    sample_texts = texts[:batch_size]
    inputs = tokenizer(
        sample_texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Profile with PyTorch profiler
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    
    print(f"\nProfiling {len(sample_texts)} samples (batch_size={batch_size})...")
    print(f"Activities: {activities}")
    
    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        with record_function("model_inference"):
            with torch.no_grad():
                for _ in range(10):  # Run 10 iterations
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    
    # Print profiler summary
    print("\n" + "="*70)
    print("CPU TIME SUMMARY (Top 10 operations)")
    print("="*70)
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    
    if device.type == "cuda":
        print("\n" + "="*70)
        print("CUDA TIME SUMMARY (Top 10 operations)")
        print("="*70)
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        
        print("\n" + "="*70)
        print("CUDA MEMORY SUMMARY (Top 10 operations)")
        print("="*70)
        print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))
    
    # Export traces
    trace_file = Path(output_dir) / "trace.json"
    prof.export_chrome_trace(str(trace_file))
    print(f"\nChrome trace saved to: {trace_file}")
    print("View in Chrome: chrome://tracing")
    
    # Export stacks
    stacks_file = Path(output_dir) / "stacks.txt"
    with open(stacks_file, 'w') as f:
        f.write(prof.key_averages(group_by_stack_n=5).table(sort_by="self_cpu_time_total", row_limit=20))
    print(f"Stack traces saved to: {stacks_file}")
    
    return prof


def analyze_layer_timing(model, tokenizer, texts, batch_size=32, max_length=512):
    """
    Analyze timing for each model layer
    
    Args:
        model: PyTorch model
        tokenizer: HuggingFace tokenizer
        texts: Sample texts
        batch_size: Batch size
        max_length: Max sequence length
        
    Returns:
        dict: Layer-wise timing information
    """
    print("\n" + "="*70)
    print("LAYER-WISE TIMING ANALYSIS")
    print("="*70)
    
    device = next(model.parameters()).device
    
    # Prepare batch
    sample_texts = texts[:batch_size]
    inputs = tokenizer(
        sample_texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    # Hook to measure layer execution time
    layer_times = {}
    
    def make_hook(name):
        def hook(module, input, output):
            if name not in layer_times:
                layer_times[name] = []
            torch.cuda.synchronize() if device.type == "cuda" else None
        return hook
    
    # Register hooks
    handles = []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            handle = module.register_forward_hook(make_hook(name))
            handles.append(handle)
    
    # Measure timing
    num_iterations = 10
    
    for _ in range(num_iterations):
        start = time.time()
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        if device.type == "cuda":
            torch.cuda.synchronize()
        end = time.time()
    
    # Remove hooks
    for handle in handles:
        handle.remove()
    
    print(f"\nAnalyzed {len(layer_times)} layers over {num_iterations} iterations")
    print("="*70 + "\n")
    
    return layer_times


def main():
    """Main profiling function"""
    # Load configuration
    config = load_config()
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(
        model_name=config['model']['name'],
        device=config['model']['device'],
        dtype=config['model']['dtype']
    )
    
    print_model_summary(model, tokenizer)
    
    # Load data
    local_path = config['dataset']['local_path']
    df = pd.read_csv(local_path)
    
    texts = df[config['dataset']['text_column']].tolist()
    labels = df['label_id'].tolist()
    
    # Use smaller sample for detailed profiling
    sample_size = 100
    sample_texts = texts[:sample_size]
    sample_labels = labels[:sample_size]
    
    print(f"Using {sample_size} samples for detailed profiling\n")
    
    # Run PyTorch profiler
    prof = profile_with_pytorch_profiler(
        model=model,
        tokenizer=tokenizer,
        texts=sample_texts,
        batch_size=32,
        max_length=config['profiling']['max_length'],
        output_dir=config['output']['results_dir'] + "/profiler_traces"
    )
    
    # Analyze layer timing (optional - can be slow)
    # layer_times = analyze_layer_timing(
    #     model=model,
    #     tokenizer=tokenizer,
    #     texts=sample_texts,
    #     batch_size=32,
    #     max_length=config['profiling']['max_length']
    # )
    
    print("\n" + "="*70)
    print("DETAILED PROFILING COMPLETE")
    print("="*70)
    print("\nKey Findings:")
    print("- Check Chrome trace for visualization: ./results/profiler_traces/trace.json")
    print("- Review stack traces: ./results/profiler_traces/stacks.txt")
    print("- Analyze CPU/CUDA time distribution in the summary above")
    print("="*70)


if __name__ == "__main__":
    main()

