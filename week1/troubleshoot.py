"""
Quick troubleshooting script to diagnose common issues
"""

import sys
import os
from pathlib import Path

def check_directory_structure():
    """Verify all required files exist"""
    print("Checking directory structure...")
    
    required = {
        'configs/config.yaml': 'Configuration file',
        'utils/__init__.py': 'Utils module init',
        'utils/config_loader.py': 'Config loader',
        'utils/model_utils.py': 'Model utilities',
        'utils/metrics.py': 'Metrics collector',
        'utils/profiler.py': 'Profiler',
        'scripts/01_download_data.py': 'Data download script',
        'scripts/02_baseline_inference.py': 'Baseline script',
        'scripts/03_detailed_profiling.py': 'Profiling script',
        'scripts/04_visualize_results.py': 'Visualization script',
    }
    
    all_ok = True
    for file_path, description in required.items():
        if Path(file_path).exists():
            print(f"  ✓ {description}: {file_path}")
        else:
            print(f"  ✗ MISSING {description}: {file_path}")
            all_ok = False
    
    return all_ok

def check_python_path():
    """Check Python path"""
    print("\nChecking Python path...")
    print(f"  Python executable: {sys.executable}")
    print(f"  Python version: {sys.version}")
    print(f"  Current directory: {os.getcwd()}")
    print(f"  Script directory: {Path(__file__).parent}")

def check_imports():
    """Test critical imports"""
    print("\nChecking critical imports...")
    
    try:
        import torch
        print(f"  ✓ torch {torch.__version__}")
    except ImportError as e:
        print(f"  ✗ torch: {e}")
        return False
    
    try:
        import transformers
        print(f"  ✓ transformers {transformers.__version__}")
    except ImportError as e:
        print(f"  ✗ transformers: {e}")
        return False
    
    try:
        sys.path.append(str(Path(__file__).parent))
        from utils import load_config
        print(f"  ✓ utils.load_config")
    except ImportError as e:
        print(f"  ✗ utils.load_config: {e}")
        return False
    
    return True

def check_config():
    """Verify config can be loaded"""
    print("\nChecking configuration...")
    
    try:
        sys.path.append(str(Path(__file__).parent))
        from utils.config_loader import load_config
        config = load_config()
        print(f"  ✓ Config loaded")
        print(f"    Model: {config['model']['name']}")
        print(f"    Dataset: {config['dataset']['hf_name']}")
        return True
    except Exception as e:
        print(f"  ✗ Config loading failed: {e}")
        return False

def test_script_execution():
    """Test if scripts can be executed"""
    print("\nTesting script execution...")
    
    # Test running a simple Python command
    import subprocess
    
    try:
        result = subprocess.run(
            [sys.executable, "-c", "print('Test OK')"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print(f"  ✓ Can execute Python scripts")
        else:
            print(f"  ✗ Script execution failed")
            return False
    except Exception as e:
        print(f"  ✗ Subprocess error: {e}")
        return False
    
    # Test running a week1 script
    script_path = Path(__file__).parent / "scripts" / "00_test_model_loading.py"
    if script_path.exists():
        print(f"  Testing: {script_path}")
        try:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                print(f"  ✓ Test script runs successfully")
            else:
                print(f"  ⚠ Test script returned code {result.returncode}")
                if result.stderr:
                    print(f"    Error: {result.stderr[:200]}")
        except subprocess.TimeoutExpired:
            print(f"  ⚠ Test script timed out (may need GPU)")
        except Exception as e:
            print(f"  ✗ Script execution error: {e}")
    
    return True

def main():
    """Run all checks"""
    print("="*70)
    print("WEEK 1 TROUBLESHOOTING")
    print("="*70)
    
    results = {}
    
    results['directory'] = check_directory_structure()
    check_python_path()
    results['imports'] = check_imports()
    results['config'] = check_config()
    results['execution'] = test_script_execution()
    
    print("\n" + "="*70)
    print("DIAGNOSIS")
    print("="*70)
    
    if all(results.values()):
        print("\n✓ All checks passed!")
        print("\nYou should be able to run:")
        print("  python run_week1.py")
    else:
        print("\n✗ Some checks failed:")
        for check, passed in results.items():
            status = "✓" if passed else "✗"
            print(f"  {status} {check}")
        
        print("\nSuggested fixes:")
        if not results.get('directory'):
            print("  - Verify you're in the week1 directory")
            print("  - Check all files were uploaded correctly")
        if not results.get('imports'):
            print("  - Run: pip install -r setup/requirements.txt")
        if not results.get('config'):
            print("  - Check configs/config.yaml exists")
            print("  - Verify YAML syntax is correct")
    
    print("="*70)

if __name__ == "__main__":
    main()

