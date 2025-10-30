# Week 1: Baseline Profiling - Project Overview

## Goal
Establish baseline inference performance for the **Qwen 2.5 0.5B** intent classification model and identify bottlenecks for future optimization.

## Model Details
- **Name**: `codefactory4791/intent-classification-qwen`
- **Base**: Qwen 2.5 0.5B Instruct
- **Task**: Multi-class classification (23 intent categories)
- **Training**: LoRA fine-tuned on Amazon product reviews
- **Expected Accuracy**: ~92%

## Dataset Details
- **Source**: `codefactory4791/amazon_test` (HuggingFace)
- **Size**: ~369,000 samples
- **Format**: CSV with columns: text, labels, label_id
- **Categories**: 23 classes across electronics, groceries, arts & crafts, musical instruments, video games

## What We Measure

### Primary Metrics
1. **Latency**
   - Per-sample inference time (ms)
   - Per-batch inference time (ms)
   - Total inference time (s)

2. **Throughput**
   - Samples processed per second
   - Batches processed per second

3. **Memory Usage**
   - GPU memory allocated (MB)
   - Peak GPU memory (MB)
   - CPU memory usage (MB)

4. **Model Characteristics**
   - Total parameters (millions)
   - Model size on disk (MB)
   - Model size in GPU memory (MB)

### Secondary Metrics
- GPU utilization percentage
- Operation-level timing (via PyTorch profiler)
- Layer-wise execution time
- Memory allocation patterns

## Key Questions We Answer

1. **What is the baseline performance?**
   - Current latency and throughput without optimization

2. **What is the optimal batch size?**
   - Balance between throughput and memory usage

3. **Where are the bottlenecks?**
   - Compute-bound vs memory-bound
   - Specific operations that are slow

4. **How much room for optimization?**
   - Identify areas for Week 2+ improvements

## Expected Baseline Results

Based on Qwen 2.5 0.5B on A100:

| Metric | Expected Range | Status |
|--------|----------------|--------|
| Model Parameters | ~494M | Measured |
| Model Size | 1-2 GB | Measured |
| Latency/sample (BS=32) | 8-15 ms | To measure |
| Throughput (BS=32) | 100-200 samples/s | To measure |
| GPU Memory (BS=32) | 3-6 GB | To measure |
| Accuracy | ~92% | Known from training |

## Profiling Approach

### Phase 1: Quick Testing
- Load model and verify inference works
- Test with 5 samples
- **Time**: 2-3 minutes

### Phase 2: Batch Size Profiling
- Test batch sizes: [1, 4, 8, 16, 32, 64]
- Run 10 iterations per batch size
- Measure latency, throughput, memory
- **Time**: 10-15 minutes

### Phase 3: Full Dataset Evaluation
- Use optimal batch size from Phase 2
- Process all ~369K samples
- Calculate accuracy metrics
- **Time**: 5-10 minutes

### Phase 4: Detailed Profiling
- Use PyTorch profiler for operation-level analysis
- Generate Chrome traces
- Identify specific bottlenecks
- **Time**: 5-10 minutes

### Phase 5: Analysis & Reporting
- Generate plots and heatmaps
- Create summary report
- Document findings
- **Time**: 2-5 minutes

**Total Estimated Time**: 25-45 minutes

## Deliverables

### 1. Metrics Files
- `baseline_metrics.json` - Complete baseline summary
- `detailed_metrics.csv` - Batch size profiling data
- `baseline_summary_report.txt` - Human-readable report

### 2. Visualizations
- `batch_size_comparison.png` - 6-panel comparison chart
- `metrics_heatmap.png` - Normalized metrics heatmap

### 3. Profiler Outputs
- `trace.json` - Chrome trace for timeline view
- `stacks.txt` - Call stacks with timing

### 4. Documentation
- Identified bottlenecks
- Optimal configuration
- Recommendations for Week 2

## Success Criteria

Week 1 is successful when you can answer:

- ✅ **Performance**: What is baseline latency and throughput?
- ✅ **Optimization**: What is the optimal batch size?
- ✅ **Bottlenecks**: What operations are slowest?
- ✅ **Memory**: How much GPU/CPU memory is used?
- ✅ **Accuracy**: Does model maintain 92% accuracy?
- ✅ **Scalability**: How does performance scale with batch size?

## Next Week Preview

Week 2 will use these baseline metrics to measure improvement from:
- **Quantization**: INT8, FP16 precision
- **Pruning**: Remove redundant parameters  
- **Distillation**: Compress model further

Target improvements:
- 2-3x speedup in latency
- 2-4x reduction in memory
- Minimal accuracy loss (<1%)

## File Organization

```
week1/
├── README.md                    # Quick start
├── week1_instructions.md        # Detailed instructions (PRIMARY DOC)
├── RUNPOD_SETUP.md             # This file
├── PROJECT_OVERVIEW.md         # Project context
├── setup.py                    # Package installation
├── run_week1.py                # Main runner
├── test_setup.py               # Verify setup
├── example_interactive.py      # Interactive example
├── setup_environment.sh        # Automated setup
│
├── setup/
│   └── requirements.txt        # Dependencies
│
├── configs/
│   └── config.yaml            # Configuration
│
├── utils/                     # Utility modules
│   ├── __init__.py
│   ├── config_loader.py
│   ├── model_utils.py
│   ├── metrics.py
│   └── profiler.py
│
├── scripts/                   # Executable scripts
│   ├── 00_test_model_loading.py
│   ├── 01_download_data.py
│   ├── 02_baseline_inference.py
│   ├── 03_detailed_profiling.py
│   └── 04_visualize_results.py
│
├── data/                      # Generated
│   ├── val_df.csv
│   └── label_mappings.json
│
└── results/                   # Generated
    ├── baseline_metrics.json
    ├── detailed_metrics.csv
    ├── baseline_summary_report.txt
    ├── plots/
    │   ├── batch_size_comparison.png
    │   └── metrics_heatmap.png
    └── profiler_traces/
        ├── trace.json
        └── stacks.txt
```

## Key Scripts Explained

| Script | Purpose | Runtime | Skippable? |
|--------|---------|---------|------------|
| `00_test_model_loading.py` | Verify model loads | 2 min | Yes (testing only) |
| `01_download_data.py` | Download dataset | 10 min | No (first run) |
| `02_baseline_inference.py` | Main profiling | 15 min | No |
| `03_detailed_profiling.py` | Bottleneck analysis | 10 min | Yes (optional) |
| `04_visualize_results.py` | Generate plots | 2 min | Yes (can view raw data) |

**Recommended**: Run `python run_week1.py` for everything

## Data Flow

```
HuggingFace Hub → 01_download_data.py → val_df.csv
                                            ↓
Model (HF Hub) → 02_baseline_inference.py → baseline_metrics.json
                          ↓                → detailed_metrics.csv
                  03_detailed_profiling.py → profiler_traces/
                          ↓
                  04_visualize_results.py  → plots/
                                            → summary_report.txt
```

## What to Review After Completion

1. **`baseline_summary_report.txt`**
   - Start here for high-level findings
   - Shows optimal batch size and key metrics

2. **`plots/batch_size_comparison.png`**
   - Visual analysis of performance vs batch size
   - Identify sweet spot

3. **`profiler_traces/trace.json`**
   - Open in Chrome (chrome://tracing)
   - See detailed operation timeline
   - Find slow operations

4. **`baseline_metrics.json`**
   - Complete raw data
   - Use for Week 2 comparisons

## Recommended Workflow

### First Time:
```bash
1. ./setup_environment.sh          # 5 min
2. python test_setup.py            # 1 min
3. python scripts/00_test_model_loading.py  # 2 min
4. python run_week1.py             # 30 min
5. Review: results/baseline_summary_report.txt
6. Download results to local machine
```

### Subsequent Runs (re-profiling):
```bash
1. python run_week1.py --skip-download  # 20 min
2. Review updated results
```

### Quick Testing:
```bash
python example_interactive.py      # 5 min with reduced samples
```

---

## Dependencies Explained

| Package | Purpose | Why Needed |
|---------|---------|------------|
| `torch` | PyTorch framework | Model inference |
| `transformers` | HuggingFace models | Load Qwen model |
| `datasets` | HuggingFace datasets | Download data |
| `nvidia-ml-py3` | GPU monitoring | Track GPU utilization |
| `psutil` | System monitoring | Track CPU/memory |
| `pandas` | Data manipulation | Process datasets |
| `scikit-learn` | ML metrics | Calculate accuracy, F1 |
| `matplotlib/seaborn` | Visualization | Generate plots |
| `pyyaml` | Config parsing | Load config.yaml |

All versions specified in `setup/requirements.txt`

---

## Support Resources

- **Primary**: `week1_instructions.md` - Complete guide
- **RunPod**: `RUNPOD_SETUP.md` - This file
- **Testing**: `test_setup.py` - Verify setup
- **Quick Start**: `README.md` - TL;DR version

---

**Ready to start?** → See `RUNPOD_SETUP.md` Step 6

