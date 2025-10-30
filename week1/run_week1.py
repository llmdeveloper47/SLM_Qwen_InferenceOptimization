"""
Main runner script for Week 1: Baseline Profiling

This script runs all Week 1 tasks in sequence:
1. Download data
2. Run baseline inference
3. Run detailed profiling
4. Visualize results
"""

import sys
import argparse
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from utils.config_loader import load_config


def main(skip_download=False, skip_profiling=False, skip_viz=False):
    """
    Run all Week 1 steps
    
    Args:
        skip_download: Skip data download step
        skip_profiling: Skip detailed profiling step
        skip_viz: Skip visualization step
    """
    print("\n" + "="*70)
    print("WEEK 1: BASELINE PROFILING AND BOTTLENECK ANALYSIS")
    print("="*70)
    print("\nThis script will:")
    print("  1. Download and prepare dataset")
    print("  2. Run baseline inference with multiple batch sizes")
    print("  3. Perform detailed profiling")
    print("  4. Generate visualization and reports")
    print("="*70 + "\n")
    
    # Load config
    config = load_config()
    
    # Step 1: Download data
    if not skip_download:
        print("\n" + "="*70)
        print("STEP 1: DOWNLOADING DATA")
        print("="*70)
        from scripts import download_data
        import importlib
        
        # Import and run download script
        sys.path.insert(0, str(Path(__file__).parent / "scripts"))
        download_module = importlib.import_module("01_download_data")
        download_module.download_and_prepare_data(config)
    else:
        print("\n[SKIPPED] Step 1: Download data")
    
    # Step 2: Baseline inference
    print("\n" + "="*70)
    print("STEP 2: BASELINE INFERENCE AND PROFILING")
    print("="*70)
    import importlib
    sys.path.insert(0, str(Path(__file__).parent / "scripts"))
    baseline_module = importlib.import_module("02_baseline_inference")
    baseline_module.run_baseline_inference(config)
    
    # Step 3: Detailed profiling
    if not skip_profiling:
        print("\n" + "="*70)
        print("STEP 3: DETAILED PROFILING")
        print("="*70)
        profiling_module = importlib.import_module("03_detailed_profiling")
        profiling_module.main()
    else:
        print("\n[SKIPPED] Step 3: Detailed profiling")
    
    # Step 4: Visualization
    if not skip_viz:
        print("\n" + "="*70)
        print("STEP 4: GENERATING VISUALIZATIONS")
        print("="*70)
        viz_module = importlib.import_module("04_visualize_results")
        viz_module.main()
    else:
        print("\n[SKIPPED] Step 4: Visualization")
    
    print("\n" + "="*70)
    print("WEEK 1 COMPLETE!")
    print("="*70)
    print(f"\nResults saved in: {config['output']['results_dir']}")
    print(f"Plots saved in: {config['output']['plots_dir']}")
    print("\nNext steps:")
    print("  - Review baseline_summary_report.txt")
    print("  - Analyze batch_size_comparison.png")
    print("  - Check profiler traces for bottlenecks")
    print("="*70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Week 1: Baseline Profiling")
    parser.add_argument("--skip-download", action="store_true", help="Skip data download")
    parser.add_argument("--skip-profiling", action="store_true", help="Skip detailed profiling")
    parser.add_argument("--skip-viz", action="store_true", help="Skip visualization")
    
    args = parser.parse_args()
    
    main(
        skip_download=args.skip_download,
        skip_profiling=args.skip_profiling,
        skip_viz=args.skip_viz
    )

