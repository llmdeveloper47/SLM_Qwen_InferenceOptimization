# Week 1: Baseline Profiling and Understanding Bottlenecks

## Overview
This week focuses on establishing baseline inference performance metrics and identifying computational bottlenecks for the Qwen 2.5 0.5B classification model.

## Objectives
- ✅ Establish baseline inference performance metrics
- ✅ Understand computational bottlenecks  
- ✅ Profile model execution
- ✅ Measure latency, throughput, and memory usage

---

## Prerequisites

### Hardware Requirements
- **GPU**: NVIDIA A100 (or similar) with CUDA support
- **RAM**: Minimum 16GB
- **Storage**: 10GB free space

### Software Requirements
- Python 3.8 or higher
- CUDA 11.8+ (for GPU support)
- Internet connection (for downloading model and data)

---

## Setup Instructions

### 1. Environment Setup on RunPod

```bash
# SSH into your RunPod instance
# Navigate to your workspace
cd /workspace

# Clone or upload this week1 directory
# If uploading, use RunPod's file manager or rsync

# Navigate to week1 directory
cd week1
```

### 2. Install Dependencies

```bash
# Install required packages
pip install -r setup/requirements.txt

# Alternatively, install in development mode
pip install -e .
```

**Expected packages:**
- `torch>=2.0.0` - PyTorch for model inference
- `transformers>=4.35.0` - HuggingFace models
- `datasets>=2.14.0` - Dataset loading
- `nvidia-ml-py3` - GPU monitoring
- `psutil` - CPU/memory monitoring
- `pandas`, `numpy`, `scikit-learn` - Data processing
- `matplotlib`, `seaborn` - Visualization
- `pyyaml` - Configuration loading
- `tqdm` - Progress bars

### 3. Verify GPU Access

```bash
# Check if CUDA is available
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

**Expected output:**
```
CUDA Available: True
GPU: NVIDIA A100-SXM4-40GB
```

---

## Directory Structure

```
week1/
├── setup/
│   └── requirements.txt          # Python dependencies
├── scripts/
│   ├── 01_download_data.py       # Download and prepare dataset
│   ├── 02_baseline_inference.py  # Run baseline profiling
│   ├── 03_detailed_profiling.py  # Detailed bottleneck analysis
│   └── 04_visualize_results.py   # Generate plots and reports
├── configs/
│   └── config.yaml               # Configuration file
├── utils/
│   ├── __init__.py
│   ├── config_loader.py          # Config utilities
│   ├── model_utils.py            # Model loading utilities
│   ├── metrics.py                # Metrics collection
│   └── profiler.py               # Inference profiler
├── data/                         # Created automatically
│   ├── val_df.csv               # Validation dataset
│   └── label_mappings.json      # Label mappings
├── results/                      # Created automatically
│   ├── baseline_metrics.json    # Baseline summary
│   ├── detailed_metrics.csv     # Batch size profiling
│   ├── baseline_summary_report.txt
│   ├── plots/                   # Visualization plots
│   └── profiler_traces/         # PyTorch profiler outputs
├── setup.py                     # Package setup
├── run_week1.py                 # Main runner script
└── week1_instructions.md        # This file
```

---

## Configuration

Edit `configs/config.yaml` to customize:

### Model Configuration
```yaml
model:
  name: "codefactory4791/intent-classification-qwen"  # Your fine-tuned model
  device: "cuda"  # or "cpu"
  dtype: "float32"  # Options: float32, float16, bfloat16
```

### Dataset Configuration
```yaml
dataset:
  hf_name: "codefactory4791/amazon_test"
  split: "test"
  local_path: "./data/val_df.csv"
  sample_sizes: [100, 500, 1000]  # For profiling
```

### Profiling Configuration
```yaml
profiling:
  batch_sizes: [1, 4, 8, 16, 32, 64]  # Test different batch sizes
  num_warmup_runs: 3
  num_measurement_runs: 10  # Runs per batch size
  max_length: 512
```

---

## Running the Code

### Option 1: Run All Steps (Recommended for First Time)

```bash
# Run complete Week 1 pipeline
python run_week1.py
```

This will execute all 4 steps in sequence:
1. Download data
2. Baseline inference
3. Detailed profiling
4. Visualization

**Estimated time**: 20-30 minutes on A100

### Option 2: Run Individual Steps

```bash
# Step 1: Download dataset (run once)
python scripts/01_download_data.py

# Step 2: Baseline inference and profiling
python scripts/02_baseline_inference.py

# Step 3: Detailed profiling (optional)
python scripts/03_detailed_profiling.py

# Step 4: Generate visualizations
python scripts/04_visualize_results.py
```

### Option 3: Skip Certain Steps

```bash
# Skip data download if already downloaded
python run_week1.py --skip-download

# Skip detailed profiling (saves time)
python run_week1.py --skip-profiling

# Skip visualization
python run_week1.py --skip-viz
```

---

## What Each Script Does

### `01_download_data.py`
**Purpose**: Download dataset from HuggingFace and save locally

**Outputs**:
- `data/val_df.csv` - Validation dataset with columns: text, labels, label_id
- `data/label_mappings.json` - Label to ID mappings

**What it measures**: N/A (data preparation only)

### `02_baseline_inference.py`
**Purpose**: Establish baseline performance metrics

**Outputs**:
- `results/baseline_metrics.json` - Comprehensive baseline summary
- `results/detailed_metrics.csv` - Batch size profiling results

**What it measures**:
- **Latency metrics**:
  - Per sample latency (ms)
  - Per batch latency (ms)
  - Total inference time
- **Throughput metrics**:
  - Samples per second
  - Batches per second
- **Memory usage**:
  - CPU memory (MB)
  - GPU memory allocated (MB)
  - Peak GPU memory (MB)
- **Model metrics**:
  - Total parameters
  - Model size (MB)
  - Accuracy on full dataset

### `03_detailed_profiling.py`
**Purpose**: Identify computational bottlenecks

**Outputs**:
- `results/profiler_traces/trace.json` - Chrome trace for visualization
- `results/profiler_traces/stacks.txt` - Stack traces with timing

**What it measures**:
- CPU time per operation
- CUDA time per operation
- Memory allocation per operation
- Call stacks and hotspots

**How to use traces**:
1. Open Chrome browser
2. Go to `chrome://tracing`
3. Load `trace.json`
4. Analyze timeline and operation costs

### `04_visualize_results.py`
**Purpose**: Generate plots and summary reports

**Outputs**:
- `results/plots/batch_size_comparison.png` - 6-panel comparison
- `results/plots/metrics_heatmap.png` - Normalized metrics heatmap
- `results/baseline_summary_report.txt` - Text summary

**Plots generated**:
1. Latency per sample vs batch size
2. Throughput vs batch size
3. GPU memory usage vs batch size
4. CPU memory usage vs batch size
5. Latency per batch vs batch size
6. Memory efficiency vs batch size

---

## Understanding the Results

### Key Metrics to Analyze

#### 1. **Latency**
- **Per Sample**: Time to process one sample (lower is better)
- **Per Batch**: Time to process one batch (affected by batch size)
- **Target**: <50ms per sample for real-time applications

#### 2. **Throughput**
- **Samples/sec**: How many samples processed per second (higher is better)
- **Target**: >100 samples/sec for production

#### 3. **Memory Usage**
- **GPU Memory**: VRAM used during inference
- **Peak GPU Memory**: Maximum VRAM allocated
- **Target**: Fit within your GPU capacity with headroom

#### 4. **Batch Size Sweet Spot**
- **Too Small (1-4)**: Underutilizes GPU, low throughput
- **Optimal (8-32)**: Best throughput/memory balance
- **Too Large (64+)**: Marginal throughput gains, high memory

### Expected Baseline Results

For Qwen 2.5 0.5B on A100:
- **Model Size**: ~1-2 GB
- **Parameters**: ~500M
- **Latency/sample**: 10-30ms (batch size 32)
- **Throughput**: 50-200 samples/sec
- **GPU Memory**: 2-6 GB (varies with batch size)
- **Accuracy**: ~92% (based on your notebook)

---

## Troubleshooting

### Issue: CUDA Out of Memory
**Solution**:
```bash
# Reduce batch size in config.yaml
batch_sizes: [1, 2, 4, 8, 16]  # Remove larger sizes

# Or use smaller model dtype
model:
  dtype: "float16"  # Instead of float32
```

### Issue: Module not found
**Solution**:
```bash
# Reinstall dependencies
pip install -r setup/requirements.txt

# Or install in development mode
pip install -e .
```

### Issue: Data download fails
**Solution**:
```bash
# Check internet connection
# Verify HuggingFace access
# Try manual download:
python -c "from datasets import load_dataset; ds = load_dataset('codefactory4791/amazon_test')"
```

### Issue: Slow inference
**Expected**: First run includes model download and compilation
- Model download: ~2-5 minutes (first time only)
- CUDA warmup: ~30 seconds
- Profiling: ~15-20 minutes total

### Issue: No GPU metrics shown
**Check**:
```bash
# Verify CUDA availability
nvidia-smi

# Check PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

---

## Interpreting Bottlenecks

After running the scripts, review these files to identify bottlenecks:

### 1. Chrome Trace (`trace.json`)
Look for:
- **Long operations**: Operations taking >10ms
- **GPU idle time**: Gaps between CUDA kernels
- **Memory allocation**: Frequent malloc/free operations

### 2. Summary Report (`baseline_summary_report.txt`)
Check:
- **Optimal batch size**: Where throughput peaks
- **Memory scaling**: How memory grows with batch size
- **Latency trends**: Linear or sub-linear growth

### 3. Profiling Tables (console output)
Identify:
- **Top CPU operations**: Matrix multiplications, attention
- **Top CUDA operations**: Kernel launches, memory transfers
- **Memory hotspots**: Layers with high memory usage

---

## Expected Outputs

After running `python run_week1.py`, you should have:

```
week1/
├── data/
│   ├── val_df.csv                    # 369,475 samples
│   └── label_mappings.json           # 23 labels
├── results/
│   ├── baseline_metrics.json         # Comprehensive metrics
│   ├── detailed_metrics.csv          # Batch size profiling
│   ├── baseline_summary_report.txt   # Human-readable summary
│   ├── plots/
│   │   ├── batch_size_comparison.png # 6-panel plot
│   │   └── metrics_heatmap.png       # Metrics heatmap
│   └── profiler_traces/
│       ├── trace.json                # Chrome trace
│       └── stacks.txt                # Stack traces
```

---

## Key Findings to Document

After completing Week 1, document these findings:

### 1. Model Characteristics
- [ ] Total parameters
- [ ] Model size on disk
- [ ] Model size in GPU memory
- [ ] Inference accuracy

### 2. Performance Baseline
- [ ] Optimal batch size
- [ ] Latency per sample at optimal batch size
- [ ] Maximum throughput achieved
- [ ] GPU memory usage at optimal batch size

### 3. Identified Bottlenecks
- [ ] Primary bottleneck (compute vs memory vs I/O)
- [ ] Slowest operations (from profiler)
- [ ] Memory allocation patterns
- [ ] GPU utilization percentage

### 4. Optimization Opportunities
- [ ] Batch size tuning potential
- [ ] Memory optimization needs
- [ ] Precision reduction viability (float16/bfloat16)
- [ ] Operator fusion opportunities

---

## Next Steps (Week 2 Preview)

Based on Week 1 findings, Week 2 will focus on:
- **Quantization**: Reduce model size and memory
- **Pruning**: Remove redundant parameters
- **Distillation**: Create smaller student model

The baseline metrics from Week 1 will be used to measure improvement.

---

## Support and Debugging

### Enable Debug Mode
Add to scripts:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Save Intermediate Results
All scripts save results automatically. Check:
- `results/baseline_metrics.json` for summary
- `results/detailed_metrics.csv` for raw data
- Console output for real-time progress

### Contact/Issues
- Review error messages carefully
- Check `results/logs/` for detailed logs
- Verify GPU memory with `nvidia-smi`
- Ensure all dependencies installed: `pip list | grep torch`

---

## Quick Start Checklist

- [ ] Install dependencies: `pip install -r setup/requirements.txt`
- [ ] Verify GPU: `python -c "import torch; print(torch.cuda.is_available())"`
- [ ] Run full pipeline: `python run_week1.py`
- [ ] Review results: `cat results/baseline_summary_report.txt`
- [ ] Analyze plots: `ls results/plots/`
- [ ] Check profiler: Open `results/profiler_traces/trace.json` in Chrome

---

## Time Estimates

| Task | Estimated Time | Can Skip? |
|------|----------------|-----------|
| Data Download | 5-10 minutes | No (first run only) |
| Baseline Inference | 10-15 minutes | No |
| Detailed Profiling | 5-10 minutes | Yes (use `--skip-profiling`) |
| Visualization | 1-2 minutes | Yes (use `--skip-viz`) |
| **Total** | **20-40 minutes** | |

*Times are estimates for A100 GPU with full dataset*

---

## Advanced Usage

### Custom Batch Sizes
Edit `configs/config.yaml`:
```yaml
profiling:
  batch_sizes: [2, 8, 16, 32, 128]  # Your custom sizes
```

### Smaller Sample for Testing
For quick testing:
```yaml
dataset:
  sample_sizes: [100]  # Just 100 samples
```

### Different Precision
Test with half precision:
```yaml
model:
  dtype: "float16"  # Or "bfloat16"
```

### Re-run Specific Steps

```bash
# Re-download data
python scripts/01_download_data.py

# Re-run profiling only
python scripts/02_baseline_inference.py

# Re-generate plots
python scripts/04_visualize_results.py
```

---

## Deliverables

At the end of Week 1, you should have:

1. **Baseline Metrics** (`baseline_metrics.json`)
   - Model size and architecture
   - Performance across batch sizes
   - Full dataset accuracy and timing

2. **Visual Analysis** (`plots/`)
   - Batch size comparison charts
   - Metrics heatmap
   - Identify optimal batch size

3. **Bottleneck Analysis** (`profiler_traces/`)
   - Chrome trace for detailed view
   - Stack traces showing hotspots
   - Operation-level timing

4. **Summary Report** (`baseline_summary_report.txt`)
   - Human-readable findings
   - Key statistics
   - Recommendations for Week 2

---

## Success Criteria

Week 1 is complete when you can answer:

- ✅ What is the baseline latency per sample?
- ✅ What is the maximum throughput achievable?
- ✅ What is the optimal batch size?
- ✅ How much GPU memory does inference require?
- ✅ What are the top 3 computational bottlenecks?
- ✅ What is the model's accuracy on the validation set?

---

## Example Output

```
======================================================================
WEEK 1: BASELINE PROFILING AND BOTTLENECK ANALYSIS
======================================================================

MODEL SUMMARY
======================================================================
Model Type: qwen2
Total Parameters: 494,033,920 (494.03M)
Model Size: 1,976.14 MB
======================================================================

BATCH SIZE ANALYSIS
Batch Size 1:
  Latency/sample: 25.34 ± 1.23 ms
  Throughput: 39.47 ± 1.91 samples/sec
  GPU Memory: 1,234.56 MB

Batch Size 32:
  Latency/sample: 8.76 ± 0.45 ms
  Throughput: 114.28 ± 5.86 samples/sec
  GPU Memory: 3,456.78 MB

...

FULL DATASET PERFORMANCE
Accuracy: 0.9207
Total time: 90.23 seconds
Throughput: 109.45 samples/sec

======================================================================
WEEK 1 COMPLETE!
======================================================================
```

---

## Model Information

**Model**: codefactory4791/intent-classification-qwen
- **Base**: Qwen 2.5 0.5B Instruct
- **Task**: Intent Classification (23 classes)
- **Dataset**: Amazon product reviews
- **Fine-tuning**: LoRA adapter (merged)

**Dataset**: val_df.csv
- **Source**: codefactory4791/amazon_test
- **Samples**: ~369K
- **Classes**: 23 (electronics, groceries, arts & crafts, musical instruments, video games)
- **Columns**: text, labels, label_id

---

## Tips for Success

1. **Start Small**: Test with 100 samples first, then scale up
2. **Monitor Resources**: Use `nvidia-smi` in separate terminal
3. **Save Frequently**: Scripts auto-save, but keep backups
4. **Document Findings**: Take notes on bottlenecks discovered
5. **Compare Results**: Track metrics for different configurations

---

## Common Commands Reference

```bash
# Full run
python run_week1.py

# Quick test (skip profiling)
python run_week1.py --skip-profiling

# Re-run after data download
python run_week1.py --skip-download

# Check GPU usage
watch -n 1 nvidia-smi

# View results
cat results/baseline_summary_report.txt
ls -lh results/plots/

# Check logs (if created)
tail -f results/logs/*.log
```

---

## Validation

To verify Week 1 completion:

```bash
# Check all required files exist
ls data/val_df.csv
ls results/baseline_metrics.json
ls results/detailed_metrics.csv
ls results/plots/batch_size_comparison.png
ls results/profiler_traces/trace.json

# Verify metrics file has data
python -c "import json; print(json.load(open('results/baseline_metrics.json'))['full_dataset_metrics']['accuracy'])"
```

**Expected**: Accuracy ~0.92

---

## Notes

- First run downloads ~2GB model from HuggingFace
- Subsequent runs use cached model
- GPU warmup takes 3-5 iterations
- Profiling uses 10 runs per batch size for statistical significance
- All timing measurements synchronized with `torch.cuda.synchronize()`

---

**Ready to start?** Run: `python run_week1.py`

**Questions?** Review this file or check console output for errors.

