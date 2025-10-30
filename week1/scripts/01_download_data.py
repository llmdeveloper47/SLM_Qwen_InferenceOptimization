"""
Script to download dataset from HuggingFace and save as val_df.csv
"""

import sys
from pathlib import Path
import pandas as pd
from datasets import load_dataset

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.config_loader import load_config


def download_and_prepare_data(config: dict):
    """
    Download dataset from HuggingFace and save locally
    
    Args:
        config: Configuration dictionary
    """
    dataset_name = config['dataset']['hf_name']
    split = config['dataset']['split']
    local_path = config['dataset']['local_path']
    
    print("="*70)
    print("DOWNLOADING DATASET")
    print("="*70)
    print(f"Dataset: {dataset_name}")
    print(f"Split: {split}")
    print(f"Saving to: {local_path}")
    
    # Load dataset from HuggingFace
    print("\nLoading from HuggingFace Hub...")
    dataset = load_dataset(dataset_name)
    
    # Convert to pandas DataFrame
    df = dataset[split].to_pandas()
    
    print(f"Loaded {len(df):,} samples")
    print(f"\nDataset columns: {list(df.columns)}")
    print(f"\nFirst few rows:")
    print(df.head())
    
    # Rename columns if needed
    if 'query' in df.columns and 'label' in df.columns:
        df = df.rename(columns={'query': 'text', 'label': 'labels'})
        print("\nRenamed columns: 'query' -> 'text', 'label' -> 'labels'")
    
    # Ensure required columns exist
    required_columns = [config['dataset']['text_column'], config['dataset']['label_column']]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in dataset")
    
    # Create label mappings
    labels = sorted(df['labels'].unique())
    label2id = {lbl: idx for idx, lbl in enumerate(labels)}
    id2label = {idx: lbl for idx, lbl in enumerate(labels)}
    
    # Add label_id column
    df['label_id'] = df['labels'].map(label2id)
    
    print(f"\nNumber of unique labels: {len(labels)}")
    print(f"Label distribution:")
    print(df['labels'].value_counts().head(10))
    
    # Create data directory if needed
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    df.to_csv(local_path, index=False)
    print(f"\nDataset saved to: {local_path}")
    
    # Save label mappings
    label_mapping_path = Path(local_path).parent / "label_mappings.json"
    import json
    with open(label_mapping_path, 'w') as f:
        json.dump({
            'label2id': label2id,
            'id2label': {str(k): v for k, v in id2label.items()},  # JSON keys must be strings
            'num_labels': len(labels)
        }, f, indent=2)
    
    print(f"Label mappings saved to: {label_mapping_path}")
    
    print("\n" + "="*70)
    print("DATA DOWNLOAD COMPLETE")
    print("="*70)
    
    return df, label2id, id2label


if __name__ == "__main__":
    # Load configuration
    config = load_config()
    
    # Download and prepare data
    df, label2id, id2label = download_and_prepare_data(config)
    
    print(f"\nDataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

