"""
Validate dataset structure and content
"""

import sys
from pathlib import Path
import pandas as pd
import json

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.config_loader import load_config


def validate_dataset(config: dict):
    """
    Validate dataset structure and content
    
    Args:
        config: Configuration dictionary
    """
    print("="*70)
    print("DATASET VALIDATION")
    print("="*70)
    
    local_path = config['dataset']['local_path']
    
    # Check if file exists
    if not Path(local_path).exists():
        print(f"✗ Dataset file not found: {local_path}")
        print(f"  Run: python scripts/01_download_data.py")
        return False
    
    print(f"✓ Dataset file found: {local_path}")
    
    # Load dataset
    try:
        df = pd.read_csv(local_path)
        print(f"✓ Dataset loaded successfully")
        print(f"  Shape: {df.shape}")
    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        return False
    
    # Check required columns
    text_col = config['dataset']['text_column']
    label_col = config['dataset']['label_column']
    
    required_cols = [text_col, label_col, 'label_id']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"✗ Missing required columns: {missing_cols}")
        print(f"  Found columns: {list(df.columns)}")
        return False
    
    print(f"✓ All required columns present: {required_cols}")
    
    # Check for missing values
    missing_counts = df[required_cols].isnull().sum()
    if missing_counts.any():
        print(f"⚠ Warning: Missing values detected:")
        print(missing_counts[missing_counts > 0])
    else:
        print(f"✓ No missing values")
    
    # Check label mappings
    label_mapping_path = Path(local_path).parent / "label_mappings.json"
    
    if not label_mapping_path.exists():
        print(f"✗ Label mappings not found: {label_mapping_path}")
        return False
    
    print(f"✓ Label mappings found")
    
    with open(label_mapping_path, 'r') as f:
        mappings = json.load(f)
    
    num_labels = mappings['num_labels']
    print(f"  Number of labels: {num_labels}")
    
    # Validate label consistency
    unique_labels = df['label_id'].nunique()
    if unique_labels != num_labels:
        print(f"⚠ Warning: Label count mismatch")
        print(f"  Expected: {num_labels}, Found: {unique_labels}")
    else:
        print(f"✓ Label count matches: {num_labels}")
    
    # Check data distribution
    print(f"\n✓ Label distribution:")
    print(df['labels'].value_counts().head(10))
    
    # Check text samples
    print(f"\n✓ Sample texts:")
    for i in range(min(3, len(df))):
        text = df[text_col].iloc[i]
        label = df[label_col].iloc[i]
        print(f"  [{i+1}] {label}: {text[:60]}...")
    
    # Dataset statistics
    print(f"\n✓ Text length statistics:")
    text_lengths = df[text_col].str.len()
    print(f"  Mean: {text_lengths.mean():.1f} characters")
    print(f"  Median: {text_lengths.median():.1f} characters")
    print(f"  Min: {text_lengths.min()} characters")
    print(f"  Max: {text_lengths.max()} characters")
    
    print("\n" + "="*70)
    print("✓ DATASET VALIDATION PASSED")
    print("="*70)
    
    return True


if __name__ == "__main__":
    config = load_config()
    success = validate_dataset(config)
    
    if success:
        print("\nDataset is ready for profiling!")
    else:
        print("\n✗ Dataset validation failed. Please fix issues above.")
    
    sys.exit(0 if success else 1)

