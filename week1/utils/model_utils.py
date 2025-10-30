"""
Model loading and utility functions
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
from pathlib import Path


def load_model_and_tokenizer(model_name: str, device: str = "cuda", dtype: str = "float32"):
    """
    Load model and tokenizer from HuggingFace Hub
    
    Args:
        model_name: HuggingFace model identifier
        device: Device to load model on ('cuda' or 'cpu')
        dtype: Data type for model weights ('float32', 'float16', 'bfloat16')
        
    Returns:
        tuple: (model, tokenizer)
    """
    print(f"Loading model: {model_name}")
    print(f"Device: {device}")
    print(f"Data type: {dtype}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Fix missing pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Determine torch dtype
    torch_dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }.get(dtype, torch.float32)
    
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
    )
    
    # Set pad_token_id in model config
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # Move to device
    if device == "cuda" and torch.cuda.is_available():
        model = model.to(device)
    elif device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU")
        model = model.to("cpu")
    else:
        model = model.to(device)
    
    # Set to evaluation mode
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"Model device: {next(model.parameters()).device}")
    
    return model, tokenizer


def get_model_size(model):
    """
    Calculate model size metrics
    
    Args:
        model: PyTorch model
        
    Returns:
        dict: Dictionary containing size metrics
    """
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Calculate memory footprint
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024**2
    
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": total_params - trainable_params,
        "param_size_mb": param_size / 1024**2,
        "buffer_size_mb": buffer_size / 1024**2,
        "total_size_mb": size_all_mb,
        "total_params_millions": total_params / 1e6,
    }


def get_model_info(model, tokenizer):
    """
    Get comprehensive model information
    
    Args:
        model: PyTorch model
        tokenizer: HuggingFace tokenizer
        
    Returns:
        dict: Model information
    """
    size_info = get_model_size(model)
    
    info = {
        "model_type": model.config.model_type,
        "vocab_size": model.config.vocab_size,
        "hidden_size": model.config.hidden_size,
        "num_hidden_layers": model.config.num_hidden_layers,
        "num_attention_heads": model.config.num_attention_heads,
        "max_position_embeddings": model.config.max_position_embeddings,
        **size_info
    }
    
    return info


def print_model_summary(model, tokenizer):
    """
    Print formatted model summary
    
    Args:
        model: PyTorch model
        tokenizer: HuggingFace tokenizer
    """
    info = get_model_info(model, tokenizer)
    
    print("\n" + "="*70)
    print("MODEL SUMMARY")
    print("="*70)
    print(f"Model Type: {info['model_type']}")
    print(f"Vocabulary Size: {info['vocab_size']:,}")
    print(f"Hidden Size: {info['hidden_size']}")
    print(f"Number of Layers: {info['num_hidden_layers']}")
    print(f"Attention Heads: {info['num_attention_heads']}")
    print(f"Max Position Embeddings: {info['max_position_embeddings']}")
    print(f"\nTotal Parameters: {info['total_params']:,} ({info['total_params_millions']:.2f}M)")
    print(f"Trainable Parameters: {info['trainable_params']:,}")
    print(f"Non-trainable Parameters: {info['non_trainable_params']:,}")
    print(f"\nModel Size: {info['total_size_mb']:.2f} MB")
    print(f"  - Parameters: {info['param_size_mb']:.2f} MB")
    print(f"  - Buffers: {info['buffer_size_mb']:.2f} MB")
    print("="*70 + "\n")

