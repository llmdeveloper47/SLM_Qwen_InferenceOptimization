# RunPod Setup Guide for Week 1

## Quick Setup on RunPod

### Step 1: Create RunPod Instance

1. Go to [RunPod.io](https://runpod.io)
2. Select **GPU Instances** → **+ Deploy**
3. Choose template: **PyTorch** (recommended) or **RunPod PyTorch**
4. Select GPU: **A100 40GB** or **A100 80GB**
5. Click **Deploy**

### Step 2: Connect to Instance

```bash
# Connect via SSH (get SSH command from RunPod dashboard)
ssh root@<your-pod-ip> -p <port> -i ~/.ssh/id_ed25519

# Or use Web Terminal in RunPod dashboard
```

### Step 3: Upload Week 1 Files

**Option A: Using RunPod Web Interface**
1. Click on your Pod → **Connect** → **HTTP Service [Port 8888]**
2. Use Jupyter interface to upload `week1/` folder
3. Navigate to terminal in Jupyter

**Option B: Using SCP** (from your local machine)
```bash
# From your local machine where week1 folder is
scp -P <port> -i ~/.ssh/id_ed25519 -r week1/ root@<pod-ip>:/workspace/
```

**Option C: Using Git** (if you have a repo)
```bash
# On RunPod instance
cd /workspace
git clone <your-repo-url>
cd <repo-name>/week1
```

### Step 4: Verify Environment

```bash
# Navigate to week1 directory
cd /workspace/week1  # Adjust path if needed

# Check Python
python --version  # Should be 3.8+

# Check CUDA
nvidia-smi

# Verify PyTorch CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### Step 5: Run Setup Script

```bash
# Make executable
chmod +x setup_environment.sh

# Run setup
./setup_environment.sh
```

**Or manually:**
```bash
# Install dependencies
pip install -r setup/requirements.txt

# Verify installation
python test_setup.py
```

### Step 6: Run Week 1 Profiling

```bash
# Full pipeline (recommended)
python run_week1.py

# Or step by step:
python scripts/01_download_data.py
python scripts/02_baseline_inference.py
python scripts/03_detailed_profiling.py
python scripts/04_visualize_results.py
```

---

## Expected Timeline on RunPod

| Step | Time | Notes |
|------|------|-------|
| Instance startup | 1-2 min | Auto-configured |
| Upload files | 2-5 min | Depends on connection |
| Install dependencies | 3-5 min | Cached after first time |
| Download model | 3-5 min | ~2GB, first time only |
| Download dataset | 5-10 min | ~1GB, first time only |
| Run profiling | 15-25 min | Full profiling |
| **Total** | **30-55 min** | First run |

Subsequent runs: ~15-20 minutes (cached model/data)

---

## RunPod-Specific Tips

### 1. Persistent Storage
```bash
# RunPod has /workspace as persistent storage
# Save all results there
cd /workspace/week1

# Your data and results will persist across pod restarts
```

### 2. Monitor GPU Usage
```bash
# Open a second terminal/tab
watch -n 1 nvidia-smi

# Or in Python:
python -c "import torch; print(torch.cuda.memory_summary())"
```

### 3. Download Results

After profiling, download results to your local machine:

```bash
# From your local machine
scp -P <port> -i ~/.ssh/id_ed25519 -r root@<pod-ip>:/workspace/week1/results ./week1_results
```

Or use RunPod web interface:
1. Navigate to files in Jupyter
2. Select `results/` folder
3. Download as zip

### 4. Jupyter Notebook Access

If you prefer notebooks:
```bash
# On RunPod instance
cd /workspace/week1
jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root --no-browser

# Access via RunPod dashboard:
# Connect → HTTP Service [Port 8888]
```

Then create a new notebook and run:
```python
exec(open('example_interactive.py').read())
```

---

## Troubleshooting on RunPod

### Issue: Out of Memory
```bash
# Check GPU memory
nvidia-smi

# Reduce batch size in config.yaml
# Or use smaller sample size for testing
```

### Issue: Slow Download
```bash
# RunPod usually has fast internet
# But if slow, you can reduce sample size:
# Edit configs/config.yaml:
#   sample_sizes: [100]  # Instead of [100, 500, 1000]
```

### Issue: Module Not Found
```bash
# Reinstall in workspace
cd /workspace/week1
pip install --upgrade -r setup/requirements.txt

# Or install from setup.py
pip install -e .
```

### Issue: CUDA Initialization Failed
```bash
# Restart pod or try:
sudo rmmod nvidia_uvm
sudo modprobe nvidia_uvm

# Then test:
python -c "import torch; print(torch.cuda.is_available())"
```

---

## Monitoring During Profiling

### Terminal 1: Run profiling
```bash
cd /workspace/week1
python run_week1.py
```

### Terminal 2: Monitor GPU
```bash
watch -n 1 nvidia-smi
```

### Terminal 3: Monitor logs (optional)
```bash
tail -f results/logs/*.log  # If logging enabled
```

---

## Saving Work

RunPod pods can be paused/terminated. To save work:

### Before Pausing Pod:
```bash
# Ensure results are in /workspace
ls /workspace/week1/results

# All files in /workspace persist when paused
```

### Before Terminating Pod:
```bash
# Download results first!
# Use SCP or RunPod web interface

# Or commit to git:
cd /workspace/week1
git add results/
git commit -m "Week 1 baseline results"
git push
```

---

## Template Selection

**Recommended RunPod Templates:**

1. **PyTorch 2.0+** (Official)
   - Pre-installed: PyTorch, CUDA, cuDNN
   - Just install: `pip install -r setup/requirements.txt`

2. **RunPod PyTorch**
   - Includes Jupyter
   - Good for interactive work

3. **Custom Template** (Advanced)
   - Base: `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime`
   - Add requirements.txt to template

---

## Cost Optimization

### Reduce Costs:
1. **Use Spot Instances**: 50-80% cheaper
2. **Pause When Not Running**: Only pay for storage
3. **Download Results**: Then terminate pod
4. **Use Smaller Sample**: For testing, use 100-1000 samples

### Estimated Costs (A100 40GB):
- **On-Demand**: ~$1.50-2.50/hour
- **Spot**: ~$0.50-1.00/hour
- **Storage**: ~$0.10/GB/month

**Week 1 total cost**: $1-3 (including setup time)

---

## Quick Commands Reference

```bash
# Setup
cd /workspace/week1
./setup_environment.sh

# Verify
python test_setup.py

# Quick test
python scripts/00_test_model_loading.py

# Full run
python run_week1.py

# Interactive example (quick)
python example_interactive.py

# Check results
cat results/baseline_summary_report.txt
ls -lh results/plots/

# Monitor GPU
watch -n 1 nvidia-smi

# Check CUDA
python -c "import torch; print(torch.cuda.get_device_properties(0))"
```

---

## Pre-flight Checklist

Before running Week 1:

- [ ] RunPod instance created (A100 recommended)
- [ ] SSH/web terminal access working
- [ ] Files uploaded to `/workspace/week1`
- [ ] Setup script run: `./setup_environment.sh`
- [ ] Test passed: `python test_setup.py`
- [ ] GPU verified: `nvidia-smi` shows A100

**All clear?** → `python run_week1.py`

---

## Getting Help

1. **Check logs**: Console output shows detailed errors
2. **Test setup**: `python test_setup.py`
3. **Verify GPU**: `nvidia-smi`
4. **Check space**: `df -h` (need ~10GB free)
5. **Review config**: `cat configs/config.yaml`

---

## After Week 1

1. **Download results**:
   ```bash
   # Results are in /workspace/week1/results
   # Download via web interface or SCP
   ```

2. **Save pod state**:
   - **Pause** if continuing later (preserves /workspace)
   - **Terminate** if done (download results first!)

3. **Prepare for Week 2**:
   - Keep pod paused
   - Review baseline metrics
   - Ready for optimization work

---

**Need help?** Check `week1_instructions.md` for detailed troubleshooting.

