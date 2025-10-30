"""
Test script to verify Week 1 setup is correct
"""

import sys
from pathlib import Path

def test_imports():
    """Test all required imports"""
    print("Testing imports...")
    
    try:
        import torch
        print(f"  ✓ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"  ✗ PyTorch import failed: {e}")
        return False
    
    try:
        import transformers
        print(f"  ✓ Transformers {transformers.__version__}")
    except ImportError as e:
        print(f"  ✗ Transformers import failed: {e}")
        return False
    
    try:
        import datasets
        print(f"  ✓ Datasets {datasets.__version__}")
    except ImportError as e:
        print(f"  ✗ Datasets import failed: {e}")
        return False
    
    try:
        import pandas
        print(f"  ✓ Pandas {pandas.__version__}")
    except ImportError as e:
        print(f"  ✗ Pandas import failed: {e}")
        return False
    
    try:
        import numpy
        print(f"  ✓ NumPy {numpy.__version__}")
    except ImportError as e:
        print(f"  ✗ NumPy import failed: {e}")
        return False
    
    try:
        import matplotlib
        print(f"  ✓ Matplotlib {matplotlib.__version__}")
    except ImportError as e:
        print(f"  ✗ Matplotlib import failed: {e}")
        return False
    
    try:
        import yaml
        print(f"  ✓ PyYAML installed")
    except ImportError as e:
        print(f"  ✗ PyYAML import failed: {e}")
        return False
    
    try:
        import sklearn
        print(f"  ✓ Scikit-learn {sklearn.__version__}")
    except ImportError as e:
        print(f"  ✗ Scikit-learn import failed: {e}")
        return False
    
    return True


def test_cuda():
    """Test CUDA availability"""
    print("\nTesting CUDA/GPU...")
    
    import torch
    
    if torch.cuda.is_available():
        print(f"  ✓ CUDA available")
        print(f"  ✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"  ✓ CUDA version: {torch.version.cuda}")
        
        # Test basic tensor operation on GPU
        try:
            x = torch.randn(100, 100).cuda()
            y = torch.randn(100, 100).cuda()
            z = torch.matmul(x, y)
            torch.cuda.synchronize()
            print(f"  ✓ GPU tensor operations working")
            return True
        except Exception as e:
            print(f"  ✗ GPU tensor operation failed: {e}")
            return False
    else:
        print(f"  ✗ CUDA not available - will run on CPU")
        print(f"  ! Warning: Profiling on CPU will be much slower")
        return False


def test_config():
    """Test configuration loading"""
    print("\nTesting configuration...")
    
    try:
        sys.path.append(str(Path(__file__).parent))
        from utils.config_loader import load_config
        
        config = load_config()
        print(f"  ✓ Config loaded successfully")
        print(f"  ✓ Model: {config['model']['name']}")
        print(f"  ✓ Dataset: {config['dataset']['hf_name']}")
        print(f"  ✓ Batch sizes: {config['profiling']['batch_sizes']}")
        return True
    except Exception as e:
        print(f"  ✗ Config loading failed: {e}")
        return False


def test_directories():
    """Test directory structure"""
    print("\nTesting directory structure...")
    
    required_dirs = [
        'setup',
        'scripts',
        'configs',
        'utils',
    ]
    
    required_files = [
        'setup/requirements.txt',
        'configs/config.yaml',
        'utils/__init__.py',
        'utils/config_loader.py',
        'utils/model_utils.py',
        'utils/metrics.py',
        'utils/profiler.py',
        'scripts/01_download_data.py',
        'scripts/02_baseline_inference.py',
        'scripts/03_detailed_profiling.py',
        'scripts/04_visualize_results.py',
        'run_week1.py',
        'setup.py',
    ]
    
    all_ok = True
    
    for dir_name in required_dirs:
        if Path(dir_name).exists():
            print(f"  ✓ Directory: {dir_name}")
        else:
            print(f"  ✗ Missing directory: {dir_name}")
            all_ok = False
    
    for file_name in required_files:
        if Path(file_name).exists():
            print(f"  ✓ File: {file_name}")
        else:
            print(f"  ✗ Missing file: {file_name}")
            all_ok = False
    
    return all_ok


def test_utils_import():
    """Test importing utility modules"""
    print("\nTesting utility modules...")
    
    try:
        sys.path.append(str(Path(__file__).parent))
        
        from utils import load_config
        print("  ✓ load_config imported")
        
        from utils import InferenceProfiler
        print("  ✓ InferenceProfiler imported")
        
        from utils import MetricsCollector
        print("  ✓ MetricsCollector imported")
        
        from utils import load_model_and_tokenizer
        print("  ✓ load_model_and_tokenizer imported")
        
        from utils import get_model_size
        print("  ✓ get_model_size imported")
        
        return True
    except Exception as e:
        print(f"  ✗ Utility import failed: {e}")
        return False


def main():
    """Run all tests"""
    print("="*70)
    print("WEEK 1 SETUP VERIFICATION")
    print("="*70)
    
    results = {}
    
    # Run tests
    results['imports'] = test_imports()
    results['cuda'] = test_cuda()
    results['directories'] = test_directories()
    results['utils'] = test_utils_import()
    results['config'] = test_config()
    
    # Summary
    print("\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)
    
    all_passed = all(results.values())
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        color = "\033[0;32m" if passed else "\033[0;31m"
        print(f"{color}{status}\033[0m - {test_name.title()}")
    
    print("="*70)
    
    if all_passed:
        print("\n✓ All tests passed! You're ready to run Week 1.")
        print("\nRun: python run_week1.py")
    else:
        print("\n✗ Some tests failed. Please review errors above.")
        print("\nCommon fixes:")
        print("  - Install dependencies: pip install -r setup/requirements.txt")
        print("  - Check CUDA: nvidia-smi")
        print("  - Verify file structure: ls -R")
    
    print("")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

