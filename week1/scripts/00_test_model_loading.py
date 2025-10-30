"""
Quick test script to verify model can be loaded and run inference
This is useful for debugging before running full profiling
"""

import sys
from pathlib import Path
import torch

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.config_loader import load_config
from utils.model_utils import load_model_and_tokenizer, print_model_summary


def test_model_loading():
    """Test if model can be loaded successfully"""
    print("="*70)
    print("MODEL LOADING TEST")
    print("="*70)
    
    # Load config
    config = load_config()
    
    # Load model and tokenizer
    try:
        model, tokenizer = load_model_and_tokenizer(
            model_name=config['model']['name'],
            device=config['model']['device'],
            dtype=config['model']['dtype']
        )
        print("\n✓ Model loaded successfully!")
    except Exception as e:
        print(f"\n✗ Model loading failed: {e}")
        return False
    
    # Print summary
    print_model_summary(model, tokenizer)
    
    return True, model, tokenizer


def test_single_inference(model, tokenizer, device="cuda"):
    """Test inference on a single sample"""
    print("="*70)
    print("SINGLE INFERENCE TEST")
    print("="*70)
    
    # Test samples from different categories
    test_samples = [
        "This guitar has amazing sound quality and great build",
        "The headphones are comfortable and have excellent noise cancellation",
        "These candy bars are delicious and the perfect size",
        "The Xbox controller works perfectly with great battery life",
    ]
    
    print(f"\nTesting with {len(test_samples)} samples...\n")
    
    try:
        # Tokenize
        inputs = tokenizer(
            test_samples,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # Move to device
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        
        print(f"Input shape: {input_ids.shape}")
        print(f"Device: {input_ids.device}")
        
        # Run inference
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=-1)
            probs = torch.softmax(outputs.logits, dim=-1)
        
        print(f"Output logits shape: {outputs.logits.shape}")
        print(f"Predictions shape: {predictions.shape}")
        
        # Print results
        print("\nPredictions:")
        for i, (text, pred, prob) in enumerate(zip(test_samples, predictions, probs)):
            pred_label = model.config.id2label.get(pred.item(), f"Class_{pred.item()}")
            confidence = prob[pred].item()
            print(f"\n  Sample {i+1}:")
            print(f"    Text: {text[:60]}...")
            print(f"    Predicted: {pred_label}")
            print(f"    Confidence: {confidence:.4f}")
        
        print("\n✓ Inference successful!")
        return True
        
    except Exception as e:
        print(f"\n✗ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_batch_inference(model, tokenizer, device="cuda"):
    """Test batched inference"""
    print("\n" + "="*70)
    print("BATCH INFERENCE TEST")
    print("="*70)
    
    # Create a larger batch
    test_text = "This is a test product review for classification"
    test_samples = [test_text] * 32  # Batch of 32
    
    print(f"\nTesting batch size: {len(test_samples)}")
    
    try:
        # Tokenize
        inputs = tokenizer(
            test_samples,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        
        # Warmup
        print("Running warmup...")
        for _ in range(3):
            with torch.no_grad():
                _ = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Timed inference
        if device == "cuda":
            torch.cuda.synchronize()
        
        import time
        start = time.time()
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        if device == "cuda":
            torch.cuda.synchronize()
        
        end = time.time()
        elapsed = end - start
        
        print(f"\nBatch inference time: {elapsed*1000:.2f} ms")
        print(f"Time per sample: {elapsed/len(test_samples)*1000:.2f} ms")
        print(f"Throughput: {len(test_samples)/elapsed:.2f} samples/sec")
        
        if device == "cuda":
            print(f"GPU memory allocated: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
            print(f"GPU memory reserved: {torch.cuda.memory_reserved()/1024**2:.2f} MB")
        
        print("\n✓ Batch inference successful!")
        return True
        
    except Exception as e:
        print(f"\n✗ Batch inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("WEEK 1 SETUP VERIFICATION")
    print("="*70)
    
    # Test imports
    if not test_imports():
        print("\n✗ Import test failed. Install dependencies:")
        print("  pip install -r setup/requirements.txt")
        return False
    
    # Test CUDA
    cuda_ok = test_cuda()
    
    # Test config
    if not test_config():
        print("\n✗ Config test failed. Check configs/config.yaml")
        return False
    
    # Test model loading
    success, model, tokenizer = test_model_loading()
    if not success:
        print("\n✗ Model loading failed. Check internet connection and HuggingFace access.")
        return False
    
    # Test inference
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if not test_single_inference(model, tokenizer, device):
        print("\n✗ Single inference test failed")
        return False
    
    if not test_batch_inference(model, tokenizer, device):
        print("\n✗ Batch inference test failed")
        return False
    
    # Final summary
    print("\n" + "="*70)
    print("ALL TESTS PASSED!")
    print("="*70)
    print("\n✓ Setup is correct and ready to use")
    print("\nYou can now run:")
    print("  python run_week1.py")
    print("\nOr run individual scripts:")
    print("  python scripts/01_download_data.py")
    print("  python scripts/02_baseline_inference.py")
    print("="*70 + "\n")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

