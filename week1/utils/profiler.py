"""
Inference profiler for detailed performance analysis
"""

import torch
import time
import numpy as np
from typing import List, Dict, Any
from tqdm import tqdm
from .metrics import MetricsCollector


class InferenceProfiler:
    """Profile model inference performance"""
    
    def __init__(self, model, tokenizer, device: str = "cuda"):
        """
        Initialize profiler
        
        Args:
            model: PyTorch model
            tokenizer: HuggingFace tokenizer
            device: Device to run inference on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.metrics_collector = MetricsCollector(device)
    
    def warmup(self, texts: List[str], batch_size: int = 8, num_warmup: int = 3):
        """
        Perform warmup runs
        
        Args:
            texts: Sample texts for warmup
            batch_size: Batch size for warmup
            num_warmup: Number of warmup iterations
        """
        print(f"Warming up model ({num_warmup} iterations)...")
        
        sample_texts = texts[:batch_size] * (batch_size // len(texts[:batch_size]) + 1)
        sample_texts = sample_texts[:batch_size]
        
        for _ in range(num_warmup):
            self._inference_batch(sample_texts)
        
        # Clear GPU cache after warmup
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("Warmup complete!")
    
    def _inference_batch(self, texts: List[str], max_length: int = 512) -> torch.Tensor:
        """
        Run inference on a batch of texts
        
        Args:
            texts: List of text strings
            max_length: Maximum sequence length
            
        Returns:
            torch.Tensor: Model predictions
        """
        # Tokenize
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        # Move to device
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=-1)
        
        return predictions
    
    def profile_batch_size(
        self, 
        texts: List[str], 
        batch_size: int,
        num_runs: int = 10,
        max_length: int = 512
    ) -> Dict[str, Any]:
        """
        Profile inference with a specific batch size
        
        Args:
            texts: List of text samples
            batch_size: Batch size to test
            num_runs: Number of runs for averaging
            max_length: Maximum sequence length
            
        Returns:
            dict: Profiling metrics
        """
        print(f"\nProfiling batch size: {batch_size}")
        
        # Ensure we have enough samples
        if len(texts) < batch_size:
            # Repeat texts to match batch size
            texts = texts * (batch_size // len(texts) + 1)
        
        texts = texts[:batch_size]
        
        # Run multiple times for statistical significance
        run_metrics = []
        
        for run in range(num_runs):
            # Clear cache before each run
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            
            # Start measurement
            measurement = self.metrics_collector.start_measurement()
            
            # Run inference
            predictions = self._inference_batch(texts, max_length)
            
            # End measurement
            metrics = self.metrics_collector.end_measurement(
                measurement, 
                batch_size=batch_size, 
                num_samples=len(texts)
            )
            
            run_metrics.append(metrics)
        
        # Aggregate results
        aggregated = self.metrics_collector.aggregate_metrics(run_metrics)
        
        return aggregated
    
    def profile_multiple_batch_sizes(
        self,
        texts: List[str],
        batch_sizes: List[int],
        num_runs: int = 10,
        max_length: int = 512
    ) -> List[Dict[str, Any]]:
        """
        Profile inference across multiple batch sizes
        
        Args:
            texts: List of text samples
            batch_sizes: List of batch sizes to test
            num_runs: Number of runs per batch size
            max_length: Maximum sequence length
            
        Returns:
            list: List of profiling results for each batch size
        """
        print("="*70)
        print("PROFILING MULTIPLE BATCH SIZES")
        print("="*70)
        
        # Warmup
        self.warmup(texts, batch_size=min(batch_sizes), num_warmup=3)
        
        results = []
        
        for batch_size in batch_sizes:
            result = self.profile_batch_size(
                texts=texts,
                batch_size=batch_size,
                num_runs=num_runs,
                max_length=max_length
            )
            results.append(result)
            
            # Print summary for this batch size
            print(f"\nBatch Size {batch_size}:")
            print(f"  Latency/sample: {result['latency_per_sample_mean']*1000:.2f} ms")
            print(f"  Throughput: {result['throughput_samples_per_sec_mean']:.2f} samples/sec")
            if 'gpu_mem_used_mean' in result:
                print(f"  GPU Memory: {result['gpu_mem_used_mean']:.2f} MB")
        
        print("\n" + "="*70)
        print("PROFILING COMPLETE")
        print("="*70)
        
        return results
    
    def profile_full_dataset(
        self,
        texts: List[str],
        labels: List[int],
        batch_size: int = 32,
        max_length: int = 512
    ) -> Dict[str, Any]:
        """
        Profile inference on full dataset with accuracy metrics
        
        Args:
            texts: List of text samples
            labels: True labels
            batch_size: Batch size for inference
            max_length: Maximum sequence length
            
        Returns:
            dict: Profiling results with accuracy
        """
        from sklearn.metrics import accuracy_score, classification_report
        
        print(f"\nProfiling full dataset ({len(texts)} samples, batch size={batch_size})")
        
        all_predictions = []
        
        # Start overall measurement
        measurement = self.metrics_collector.start_measurement()
        
        # Process in batches
        for i in tqdm(range(0, len(texts), batch_size), desc="Processing"):
            batch_texts = texts[i:i+batch_size]
            predictions = self._inference_batch(batch_texts, max_length)
            all_predictions.extend(predictions.cpu().numpy())
        
        # End measurement
        metrics = self.metrics_collector.end_measurement(
            measurement,
            batch_size=batch_size,
            num_samples=len(texts)
        )
        
        # Calculate accuracy
        accuracy = accuracy_score(labels, all_predictions)
        metrics['accuracy'] = accuracy
        metrics['predictions'] = all_predictions
        
        print(f"\nAccuracy: {accuracy:.4f}")
        print(f"Total time: {metrics['latency_total']:.2f} seconds")
        print(f"Throughput: {metrics['throughput_samples_per_sec']:.2f} samples/sec")
        
        return metrics
    
    def get_all_metrics(self) -> List[Dict[str, Any]]:
        """Get all collected metrics"""
        return self.metrics_collector.metrics
    
    def save_results(self, filepath: str, aggregated: bool = True):
        """
        Save profiling results
        
        Args:
            filepath: Path to save results
            aggregated: Whether to aggregate results by batch size
        """
        self.metrics_collector.save_metrics(filepath, aggregated)

