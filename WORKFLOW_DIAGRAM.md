# Week 1 Workflow Diagram

## 📊 Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    Week 1: Baseline Profiling                    │
└─────────────────────────────────────────────────────────────────┘

┌──────────────────┐
│  HuggingFace Hub │
│  amazon_test     │
└────────┬─────────┘
         │
         ▼
┌──────────────────────────────────────────────┐
│  01_download_data.py                         │
│  • Downloads dataset                         │
│  • Converts to CSV                           │
│  • Creates label mappings                    │
└────────┬─────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────┐
│  data/                                       │
│  ├── val_df.csv (369K samples)              │
│  └── label_mappings.json (23 labels)        │
└────────┬─────────────────────────────────────┘
         │
         ├─────────────────────────────────────┐
         │                                      │
         ▼                                      ▼
┌──────────────────┐              ┌─────────────────────┐
│  HuggingFace Hub │              │  Model Loading      │
│  intent-class    │              │  • Load from HF     │
│  -qwen model     │              │  • Setup tokenizer  │
└────────┬─────────┘              │  • Move to GPU      │
         │                        │  • Set eval mode    │
         │                        └──────────┬──────────┘
         │                                   │
         └───────────────┬───────────────────┘
                         │
                         ▼
         ┌───────────────────────────────────┐
         │  02_baseline_inference.py         │
         │  ┌─────────────────────────────┐  │
         │  │ Profile Multiple Batch Sizes│  │
         │  │  • BS: 1, 4, 8, 16, 32, 64  │  │
         │  │  • 10 runs each             │  │
         │  │  • Measure:                 │  │
         │  │    - Latency                │  │
         │  │    - Throughput             │  │
         │  │    - Memory                 │  │
         │  └─────────────────────────────┘  │
         │  ┌─────────────────────────────┐  │
         │  │ Full Dataset Evaluation     │  │
         │  │  • All 369K samples         │  │
         │  │  • Optimal batch size       │  │
         │  │  • Calculate accuracy       │  │
         │  └─────────────────────────────┘  │
         └──────────┬────────────────────────┘
                    │
                    ▼
         ┌────────────────────────────────────┐
         │  results/                          │
         │  ├── baseline_metrics.json         │
         │  └── detailed_metrics.csv          │
         └──────────┬─────────────────────────┘
                    │
                    ├──────────────────┐
                    │                  │
                    ▼                  ▼
    ┌──────────────────────┐  ┌────────────────────────┐
    │ 03_detailed_         │  │ 04_visualize_          │
    │    profiling.py      │  │    results.py          │
    │                      │  │                        │
    │ • PyTorch Profiler   │  │ • Batch size plots     │
    │ • Operation timing   │  │ • Metrics heatmap      │
    │ • Memory tracking    │  │ • Summary report       │
    │ • Stack traces       │  │ • Analysis charts      │
    └──────────┬───────────┘  └────────┬───────────────┘
               │                       │
               ▼                       ▼
    ┌─────────────────────┐  ┌──────────────────────┐
    │ profiler_traces/    │  │ plots/               │
    │ ├── trace.json      │  │ ├── batch_size_      │
    │ └── stacks.txt      │  │ │   comparison.png   │
    └─────────────────────┘  │ └── metrics_         │
                             │     heatmap.png      │
                             └──────────────────────┘
```

---

## 🔄 Execution Flow

```
run_week1.py (Main Orchestrator)
    │
    ├─► [Step 1] 01_download_data.py
    │   └─► Downloads HuggingFace dataset
    │       Creates val_df.csv
    │       Creates label mappings
    │       Time: ~10 minutes
    │
    ├─► [Step 2] 02_baseline_inference.py
    │   ├─► Load model
    │   ├─► Profile batch sizes: [1, 4, 8, 16, 32, 64]
    │   │   └─► For each batch size:
    │   │       ├─► Warmup (3 runs)
    │   │       ├─► Measure (10 runs)
    │   │       └─► Aggregate metrics
    │   ├─► Full dataset inference
    │   └─► Calculate accuracy
    │   Time: ~15 minutes
    │
    ├─► [Step 3] 03_detailed_profiling.py (Optional)
    │   ├─► PyTorch profiler
    │   ├─► Chrome trace generation
    │   └─► Stack analysis
    │   Time: ~10 minutes
    │
    └─► [Step 4] 04_visualize_results.py
        ├─► Load metrics
        ├─► Generate 6-panel plot
        ├─► Create heatmap
        └─► Write summary report
        Time: ~2 minutes

Total: 30-40 minutes
```

---

## 🧩 Module Dependencies

```
run_week1.py
    │
    ├─► utils/config_loader.py
    │   └─► configs/config.yaml
    │
    ├─► utils/model_utils.py
    │   ├─► torch
    │   └─► transformers
    │
    ├─► utils/metrics.py
    │   ├─► torch
    │   ├─► psutil
    │   └─► pynvml (optional)
    │
    └─► utils/profiler.py
        ├─► utils/metrics.py
        ├─► torch
        └─► sklearn (for accuracy)

All scripts import from utils/
All utils are independent and reusable
```

---

## 📈 Metrics Collection Flow

```
Start Measurement
    │
    ├─► Record start time
    ├─► Record CPU memory (psutil)
    ├─► Record GPU memory (torch.cuda)
    └─► Record GPU utilization (pynvml)
    │
    ▼
Run Inference
    │
    ├─► Tokenize text
    ├─► Move to GPU
    ├─► Model forward pass
    ├─► Get predictions
    └─► Sync GPU (torch.cuda.synchronize)
    │
    ▼
End Measurement
    │
    ├─► Record end time
    ├─► Calculate elapsed time
    ├─► Record final memory
    └─► Calculate metrics:
        ├─► Latency = elapsed / samples
        ├─► Throughput = samples / elapsed
        ├─► Memory = final - start
        └─► Utilization = average
    │
    ▼
Aggregate Results (10 runs)
    │
    ├─► Calculate mean
    ├─► Calculate std deviation
    ├─► Calculate min/max
    └─► Calculate median
    │
    ▼
Save Results
    │
    ├─► JSON (baseline_metrics.json)
    ├─► CSV (detailed_metrics.csv)
    └─► TXT (baseline_summary_report.txt)
```

---

## 🎯 Optimization Journey

```
Week 1 (Baseline)
    ↓
Measure current performance
• Latency: ~20ms/sample
• Throughput: ~100 samples/sec
• Memory: ~4GB GPU
    ↓
Identify bottlenecks
• Matrix multiplications (slow)
• Attention layers (memory)
• Data transfer (I/O)
    ↓
Document baseline
• All metrics saved
• Bottlenecks identified
• Optimal config found
    ↓
Week 2 (Optimization)
• Apply quantization
• Reduce precision
• Optimize batch size
    ↓
Compare improvements
• 2-3x faster
• 50% less memory
• <1% accuracy loss
```

---

## 🔍 Profiling Methodology

### Why Multiple Runs?
- **Warmup (3 runs)**: Eliminate CUDA compilation, cache effects
- **Measurement (10 runs)**: Statistical significance
- **Aggregation**: Mean ± std for reliability

### Why Multiple Batch Sizes?
- **Small (1-4)**: Baseline latency, minimal memory
- **Medium (8-16)**: Balanced performance
- **Large (32-64)**: Maximum throughput
- **Analysis**: Find sweet spot

### Why Full Dataset?
- **Real-world performance**: Not just sample
- **Accuracy verification**: Ensure model works
- **End-to-end timing**: Complete pipeline
- **Production estimate**: Actual deployment metrics

---

## 🎁 Included Utilities

### Testing & Validation
- `test_setup.py` - Verify environment
- `00_test_model_loading.py` - Test model
- `validate_dataset.py` - Validate data
- `example_interactive.py` - Quick profiling

### Automation
- `setup_environment.sh` - One-command setup
- `run_week1.py` - One-command execution
- `show_structure.py` - View directory tree

### Documentation
- 7 markdown files covering all aspects
- Every script has docstrings
- Every function documented
- Examples included

---

## 📦 Deliverables After Week 1

You will have:

1. **Quantitative Metrics**
   - Exact latency measurements
   - Throughput numbers
   - Memory usage data
   - Model size statistics

2. **Qualitative Analysis**
   - Bottleneck identification
   - Optimization opportunities
   - Configuration recommendations
   - Week 2 preparation

3. **Artifacts**
   - Metrics files (JSON, CSV)
   - Visualization plots (PNG)
   - Profiler traces (Chrome)
   - Summary report (TXT)

4. **Knowledge**
   - Current performance understood
   - Bottlenecks identified
   - Baseline established
   - Ready for optimization

---

## 🏁 Your Action Items

### Right Now:
1. Read `START_HERE.md` (2 min)
2. Upload files to RunPod
3. Follow `RUNPOD_SETUP.md`

### On RunPod:
1. `pip install -r setup/requirements.txt`
2. `python test_setup.py`
3. `python run_week1.py`

### After Completion:
1. Review `results/baseline_summary_report.txt`
2. Analyze `results/plots/`
3. Download results
4. Document findings for Week 2

---

**Everything is ready!**

**Next**: Upload to RunPod and run `python run_week1.py`

**Documentation**: All questions answered in markdown files

**Support**: Detailed troubleshooting in `week1_instructions.md`

---

Good luck with Week 1! 🚀

