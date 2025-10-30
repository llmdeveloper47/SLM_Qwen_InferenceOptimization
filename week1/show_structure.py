"""
Display week1 directory structure
"""

import os
from pathlib import Path


def print_tree(directory, prefix="", max_depth=4, current_depth=0, ignore_dirs={'.git', '__pycache__', '.ipynb_checkpoints', 'results', 'data'}):
    """Print directory tree structure"""
    
    if current_depth >= max_depth:
        return
    
    try:
        entries = sorted(Path(directory).iterdir(), key=lambda x: (not x.is_dir(), x.name))
    except PermissionError:
        return
    
    for i, entry in enumerate(entries):
        is_last = i == len(entries) - 1
        
        # Skip ignored directories
        if entry.is_dir() and entry.name in ignore_dirs:
            continue
        
        # Print entry
        connector = "└── " if is_last else "├── "
        print(f"{prefix}{connector}{entry.name}")
        
        # Recurse into directories
        if entry.is_dir():
            extension = "    " if is_last else "│   "
            print_tree(entry, prefix + extension, max_depth, current_depth + 1, ignore_dirs)


def main():
    """Display week1 structure"""
    print("="*70)
    print("WEEK 1 DIRECTORY STRUCTURE")
    print("="*70)
    print("\nweek1/")
    
    week1_dir = Path(__file__).parent
    print_tree(week1_dir, ignore_dirs={'.git', '__pycache__', '.ipynb_checkpoints', 'egg-info'})
    
    print("\n" + "="*70)
    print("NOTES:")
    print("  - 'data/' and 'results/' will be created when scripts run")
    print("  - Start with: python run_week1.py")
    print("  - Read: week1_instructions.md for details")
    print("="*70)


if __name__ == "__main__":
    main()

