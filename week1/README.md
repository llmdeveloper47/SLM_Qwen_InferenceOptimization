# Week 1: Baseline Profiling and Bottleneck Analysis

## Quick Start

```bash
# 1. Install dependencies
pip install -r setup/requirements.txt

# 2. Run complete pipeline
python run_week1.py
```

## What This Week Covers

Establishes performance baseline for **Qwen 2.5 0.5B** classification model:
- ✅ Measure latency, throughput, memory usage
- ✅ Test multiple batch sizes
- ✅ Identify computational bottlenecks
- ✅ Profile with PyTorch profiler
- ✅ Generate visualization reports

## Key Files

- **`run_week1.py`** - Main runner (start here)
- **`week1_instructions.md`** - Detailed instructions
- **`configs/config.yaml`** - Configuration
- **`scripts/`** - Individual step scripts
- **`utils/`** - Utility modules

## Model & Dataset

**Model**: `codefactory4791/intent-classification-qwen`
- Qwen 2.5 0.5B fine-tuned for intent classification
- 23 classes (Amazon product categories)
- ~92% accuracy baseline

**Dataset**: `val_df.csv` (from HuggingFace)
- 369K samples for evaluation
- Columns: text, labels, label_id

## Results

After running, check:
- `results/baseline_summary_report.txt` - Key findings
- `results/plots/` - Visualizations
- `results/baseline_metrics.json` - Full metrics

## Time Required

**Total**: ~20-40 minutes on A100 GPU
- Data download: 5-10 min (first time only)
- Profiling: 10-15 min
- Analysis: 5-10 min
- Visualization: 1-2 min

## Documentation

See **`week1_instructions.md`** for:
- Complete setup guide
- Step-by-step instructions
- Troubleshooting
- Understanding results
- Advanced usage

---

**Ready?** → `python run_week1.py`

