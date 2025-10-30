# Getting Started with Week 1

## 🚀 Quick Start (5 Steps)

### 1️⃣ On RunPod: Create Instance
- GPU: **A100 40GB or 80GB**
- Template: **PyTorch 2.0+**
- Storage: **20GB minimum**

### 2️⃣ Upload Files
```bash
# SSH to RunPod
ssh root@<pod-ip> -p <port>

# Navigate to workspace
cd /workspace

# Upload week1 folder (use web interface or scp)
```

### 3️⃣ Install Dependencies
```bash
cd /workspace/week1

# Quick install
pip install -r setup/requirements.txt

# Or use automated setup
./setup_environment.sh
```

### 4️⃣ Verify Setup
```bash
python test_setup.py
```

**Expected**: All tests pass ✓

### 5️⃣ Run Week 1
```bash
python run_week1.py
```

**Duration**: 25-40 minutes
**Output**: `results/` folder with all metrics and plots

---

## 📚 Documentation Guide

**Start here depending on your needs:**

| If you want... | Read this file |
|----------------|----------------|
| **Quick 5-minute overview** | `README.md` |
| **Complete step-by-step guide** | `week1_instructions.md` ⭐ |
| **RunPod-specific setup** | `RUNPOD_SETUP.md` |
| **Project context & goals** | `PROJECT_OVERVIEW.md` |
| **Just run it** | Run `python run_week1.py` |

**Primary Documentation**: `week1_instructions.md` (most comprehensive)

---

## 🎯 What You'll Get

After running Week 1, you'll have:

### Metrics
- ✅ Baseline latency: XX ms per sample
- ✅ Throughput: XX samples/second
- ✅ Memory usage: XX MB GPU
- ✅ Optimal batch size: XX
- ✅ Model accuracy: ~92%

### Files
```
results/
├── baseline_metrics.json          # Complete metrics
├── detailed_metrics.csv           # Batch size data
├── baseline_summary_report.txt    # Human-readable
├── plots/
│   ├── batch_size_comparison.png  # Main visualization
│   └── metrics_heatmap.png        # Metrics overview
└── profiler_traces/
    ├── trace.json                 # Chrome timeline
    └── stacks.txt                 # Call stacks
```

### Insights
- Primary bottleneck identified
- Optimal inference configuration
- Areas for Week 2 optimization

---

## ⚡ Quick Commands

```bash
# Complete setup and run
./setup_environment.sh && python run_week1.py

# Just profiling (if data exists)
python run_week1.py --skip-download

# Fast test (reduced samples)
python example_interactive.py

# Verify setup
python test_setup.py

# Test model only
python scripts/00_test_model_loading.py

# Check results
cat results/baseline_summary_report.txt
```

---

## 🔧 Customization

### Test Faster (for debugging)
Edit `configs/config.yaml`:
```yaml
dataset:
  sample_sizes: [100]  # Just 100 samples

profiling:
  batch_sizes: [8, 32]  # Just 2 batch sizes
  num_measurement_runs: 3  # Fewer runs
```

Then run:
```bash
python run_week1.py
```

### Use Different Precision
```yaml
model:
  dtype: "float16"  # Faster, less memory
```

### Profile Larger Batches
```yaml
profiling:
  batch_sizes: [1, 4, 8, 16, 32, 64, 128]
```

---

## ❗ Common Issues

### "CUDA out of memory"
```bash
# Reduce batch size in config.yaml
batch_sizes: [1, 4, 8, 16]  # Remove larger sizes
```

### "Module not found"
```bash
pip install -r setup/requirements.txt
# or
pip install -e .
```

### "Dataset not found"
```bash
python scripts/01_download_data.py
```

### "Model download failed"
```bash
# Check internet
ping huggingface.co

# Check HuggingFace access
python -c "from huggingface_hub import HfApi; print(HfApi().whoami())"
```

---

## 📊 Interpreting Results

### Good Performance Indicators
- ✅ Latency < 50ms per sample
- ✅ Throughput > 100 samples/sec
- ✅ GPU memory < 50% of capacity
- ✅ Accuracy maintained at ~92%

### Optimization Needed If
- ⚠️ Latency > 100ms per sample
- ⚠️ Throughput < 50 samples/sec
- ⚠️ GPU memory > 80% of capacity
- ⚠️ Accuracy drops below 90%

### Where to Look
1. **High latency** → Check `profiler_traces/trace.json` for slow ops
2. **Low throughput** → Try larger batch sizes
3. **High memory** → Consider quantization (Week 2)
4. **Low accuracy** → Verify model loaded correctly

---

## 📁 File Structure Reference

```
week1/
│
├── 📖 Documentation
│   ├── README.md                    ← Quick start (start here)
│   ├── week1_instructions.md        ← Complete guide (main doc)
│   ├── RUNPOD_SETUP.md              ← RunPod specific
│   ├── PROJECT_OVERVIEW.md          ← Context & goals
│   └── GETTING_STARTED.md           ← This file
│
├── ⚙️ Configuration
│   ├── configs/config.yaml          ← Main config
│   ├── setup.py                     ← Package setup
│   └── setup/requirements.txt       ← Dependencies
│
├── 🛠️ Utilities
│   └── utils/                       ← Reusable modules
│       ├── config_loader.py
│       ├── model_utils.py
│       ├── metrics.py
│       └── profiler.py
│
├── 📜 Scripts
│   └── scripts/                     ← Executable scripts
│       ├── 00_test_model_loading.py    (test)
│       ├── 01_download_data.py         (required)
│       ├── 02_baseline_inference.py    (required)
│       ├── 03_detailed_profiling.py    (optional)
│       ├── 04_visualize_results.py     (optional)
│       └── validate_dataset.py         (utility)
│
├── 🎬 Runners
│   ├── run_week1.py                 ← Main runner (use this)
│   ├── example_interactive.py       ← Quick example
│   ├── test_setup.py                ← Verify setup
│   └── setup_environment.sh         ← Auto setup
│
└── 📊 Generated (after running)
    ├── data/
    │   ├── val_df.csv
    │   └── label_mappings.json
    └── results/
        ├── baseline_metrics.json
        ├── detailed_metrics.csv
        ├── baseline_summary_report.txt
        ├── plots/
        └── profiler_traces/
```

---

## 🎓 Learning Path

### For First-Time Users:
1. Read `README.md` (5 min)
2. Follow `RUNPOD_SETUP.md` (10 min)
3. Run `python run_week1.py` (30 min)
4. Review `baseline_summary_report.txt` (5 min)
5. Analyze plots in `results/plots/` (5 min)

### For Experienced Users:
```bash
./setup_environment.sh
python run_week1.py
cat results/baseline_summary_report.txt
```

### For Debugging:
1. `python test_setup.py` - Verify environment
2. `python scripts/00_test_model_loading.py` - Test model
3. `python scripts/validate_dataset.py` - Validate data
4. `python example_interactive.py` - Quick profiling

---

## 🔍 What Each Script Does

### Data Preparation
- **`01_download_data.py`**: Downloads from HuggingFace, saves as CSV
  - Input: HuggingFace dataset ID
  - Output: `val_df.csv`, `label_mappings.json`
  - Time: ~10 minutes

### Baseline Profiling
- **`02_baseline_inference.py`**: Main profiling script
  - Input: `val_df.csv`
  - Output: `baseline_metrics.json`, `detailed_metrics.csv`
  - Measures: Latency, throughput, memory across batch sizes
  - Time: ~15 minutes

### Bottleneck Analysis
- **`03_detailed_profiling.py`**: PyTorch profiler analysis
  - Input: Model + sample data
  - Output: `trace.json`, `stacks.txt`
  - Identifies: Slow operations, memory hotspots
  - Time: ~10 minutes

### Visualization
- **`04_visualize_results.py`**: Generate plots and reports
  - Input: Metrics files
  - Output: PNG plots, summary report
  - Time: ~2 minutes

---

## 💡 Tips for Success

1. **Start with test script**: `python test_setup.py`
2. **Use small samples first**: Test with 100 samples before full run
3. **Monitor GPU**: Open `nvidia-smi` in another terminal
4. **Save results**: Download from RunPod before terminating
5. **Read outputs**: Check `baseline_summary_report.txt` for insights

---

## 🆘 Need Help?

1. **Setup issues** → `RUNPOD_SETUP.md`
2. **Detailed instructions** → `week1_instructions.md`
3. **Understanding results** → `PROJECT_OVERVIEW.md`
4. **Quick reference** → `README.md`

---

## ✅ Checklist Before Running

- [ ] RunPod A100 instance created
- [ ] Files uploaded to `/workspace/week1`
- [ ] Dependencies installed: `pip install -r setup/requirements.txt`
- [ ] Setup verified: `python test_setup.py` passes
- [ ] GPU available: `nvidia-smi` shows A100

**All checked?** → `python run_week1.py`

---

## 📈 What to Expect

```
$ python run_week1.py

======================================================================
WEEK 1: BASELINE PROFILING AND BOTTLENECK ANALYSIS
======================================================================

STEP 1: DOWNLOADING DATA
  → Downloading from HuggingFace...
  → Saved 369,475 samples to data/val_df.csv
  ✓ Complete (10 minutes)

STEP 2: BASELINE INFERENCE AND PROFILING
  → Loading model...
  → Profiling batch sizes: [1, 4, 8, 16, 32, 64]
  → Running 10 iterations per batch size...
  ✓ Complete (15 minutes)

STEP 3: DETAILED PROFILING
  → Running PyTorch profiler...
  → Generating trace.json...
  ✓ Complete (10 minutes)

STEP 4: GENERATING VISUALIZATIONS
  → Creating plots...
  → Generating report...
  ✓ Complete (2 minutes)

======================================================================
WEEK 1 COMPLETE!
======================================================================
Results: ./results/
  - baseline_summary_report.txt
  - plots/batch_size_comparison.png
  - baseline_metrics.json
======================================================================
```

---

**Ready?** → `python run_week1.py`

**Questions?** → Read `week1_instructions.md`

**Issues?** → Run `python test_setup.py`

