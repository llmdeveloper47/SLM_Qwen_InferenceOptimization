"""
Metrics collection and calculation utilities
"""

import time
import torch
import psutil
import numpy as np
from typing import List, Dict, Any
import json
import pandas as pd
from pathlib import Path

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    print("Warning: pynvml not available. GPU metrics will not be collected.")


class MetricsCollector:
    """Collects and manages inference metrics"""
    
    def __init__(self, device: str = "cuda"):
        """
        Initialize metrics collector
        
        Args:
            device: Device being used ('cuda' or 'cpu')
        """
        self.device = device
        self.metrics = []
        self.gpu_available = device == "cuda" and torch.cuda.is_available()
        
        # Initialize NVML for GPU monitoring
        if self.gpu_available and NVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self.nvml_initialized = True
            except Exception as e:
                print(f"Warning: Could not initialize NVML: {e}")
                self.nvml_initialized = False
        else:
            self.nvml_initialized = False
    
    def start_measurement(self) -> Dict[str, Any]:
        """Start a new measurement session"""
        measurement = {
            'start_time': time.time(),
            'start_cpu_mem': psutil.Process().memory_info().rss / 1024**2,  # MB
        }
        
        if self.gpu_available:
            torch.cuda.synchronize()
            measurement['start_gpu_mem'] = torch.cuda.memory_allocated() / 1024**2  # MB
            measurement['start_gpu_mem_reserved'] = torch.cuda.memory_reserved() / 1024**2  # MB
            
            if self.nvml_initialized:
                try:
                    gpu_util = pynvml.nvmlDeviceGetUtilizationRates(self.nvml_handle)
                    measurement['start_gpu_util'] = gpu_util.gpu
                except:
                    pass
        
        return measurement
    
    def end_measurement(self, measurement: Dict[str, Any], batch_size: int, num_samples: int) -> Dict[str, Any]:
        """
        End measurement and calculate metrics
        
        Args:
            measurement: Started measurement dict
            batch_size: Batch size used
            num_samples: Total number of samples processed
            
        Returns:
            dict: Complete metrics
        """
        if self.gpu_available:
            torch.cuda.synchronize()
        
        end_time = time.time()
        elapsed_time = end_time - measurement['start_time']
        
        measurement['end_time'] = end_time
        measurement['elapsed_time'] = elapsed_time
        measurement['batch_size'] = batch_size
        measurement['num_samples'] = num_samples
        
        # Latency metrics
        measurement['latency_total'] = elapsed_time
        measurement['latency_per_sample'] = elapsed_time / num_samples
        measurement['latency_per_batch'] = elapsed_time / (num_samples / batch_size)
        
        # Throughput metrics
        measurement['throughput_samples_per_sec'] = num_samples / elapsed_time
        measurement['throughput_batches_per_sec'] = (num_samples / batch_size) / elapsed_time
        
        # Memory metrics
        measurement['end_cpu_mem'] = psutil.Process().memory_info().rss / 1024**2
        measurement['cpu_mem_used'] = measurement['end_cpu_mem'] - measurement['start_cpu_mem']
        
        if self.gpu_available:
            measurement['end_gpu_mem'] = torch.cuda.memory_allocated() / 1024**2
            measurement['end_gpu_mem_reserved'] = torch.cuda.memory_reserved() / 1024**2
            measurement['gpu_mem_used'] = measurement['end_gpu_mem'] - measurement['start_gpu_mem']
            measurement['gpu_mem_reserved_used'] = measurement['end_gpu_mem_reserved'] - measurement['start_gpu_mem_reserved']
            measurement['peak_gpu_mem'] = torch.cuda.max_memory_allocated() / 1024**2
            
            if self.nvml_initialized:
                try:
                    gpu_util = pynvml.nvmlDeviceGetUtilizationRates(self.nvml_handle)
                    measurement['end_gpu_util'] = gpu_util.gpu
                    measurement['avg_gpu_util'] = (measurement.get('start_gpu_util', 0) + gpu_util.gpu) / 2
                except:
                    pass
        
        self.metrics.append(measurement)
        return measurement
    
    def aggregate_metrics(self, metrics_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate multiple measurements
        
        Args:
            metrics_list: List of measurement dictionaries
            
        Returns:
            dict: Aggregated statistics
        """
        if not metrics_list:
            return {}
        
        # Extract numeric metrics
        numeric_keys = [
            'latency_total', 'latency_per_sample', 'latency_per_batch',
            'throughput_samples_per_sec', 'throughput_batches_per_sec',
            'cpu_mem_used', 'gpu_mem_used', 'peak_gpu_mem'
        ]
        
        aggregated = {}
        
        for key in numeric_keys:
            values = [m[key] for m in metrics_list if key in m]
            if values:
                aggregated[f'{key}_mean'] = np.mean(values)
                aggregated[f'{key}_std'] = np.std(values)
                aggregated[f'{key}_min'] = np.min(values)
                aggregated[f'{key}_max'] = np.max(values)
                aggregated[f'{key}_median'] = np.median(values)
        
        # Add batch size and num_samples from first measurement
        aggregated['batch_size'] = metrics_list[0]['batch_size']
        aggregated['num_samples'] = metrics_list[0]['num_samples']
        aggregated['num_runs'] = len(metrics_list)
        
        return aggregated
    
    def save_metrics(self, filepath: str, aggregated: bool = True):
        """
        Save metrics to file
        
        Args:
            filepath: Path to save metrics
            aggregated: Whether to save aggregated or raw metrics
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if aggregated and len(self.metrics) > 0:
            # Group by batch size and aggregate
            batch_sizes = list(set(m['batch_size'] for m in self.metrics))
            aggregated_results = []
            
            for bs in sorted(batch_sizes):
                bs_metrics = [m for m in self.metrics if m['batch_size'] == bs]
                agg = self.aggregate_metrics(bs_metrics)
                aggregated_results.append(agg)
            
            # Save as JSON
            if filepath.suffix == '.json':
                with open(filepath, 'w') as f:
                    json.dump(aggregated_results, f, indent=2)
            # Save as CSV
            elif filepath.suffix == '.csv':
                df = pd.DataFrame(aggregated_results)
                df.to_csv(filepath, index=False)
        else:
            # Save raw metrics
            if filepath.suffix == '.json':
                with open(filepath, 'w') as f:
                    json.dump(self.metrics, f, indent=2)
            elif filepath.suffix == '.csv':
                df = pd.DataFrame(self.metrics)
                df.to_csv(filepath, index=False)
        
        print(f"Metrics saved to: {filepath}")
    
    def print_summary(self, batch_size: int = None):
        """
        Print summary of collected metrics
        
        Args:
            batch_size: If specified, only show metrics for this batch size
        """
        if not self.metrics:
            print("No metrics collected yet")
            return
        
        if batch_size is not None:
            metrics_to_show = [m for m in self.metrics if m['batch_size'] == batch_size]
            if not metrics_to_show:
                print(f"No metrics found for batch size {batch_size}")
                return
        else:
            metrics_to_show = self.metrics
        
        agg = self.aggregate_metrics(metrics_to_show)
        
        print("\n" + "="*70)
        print(f"METRICS SUMMARY (Batch Size: {agg['batch_size']})")
        print("="*70)
        print(f"Number of runs: {agg['num_runs']}")
        print(f"Total samples: {agg['num_samples']}")
        print(f"\nLatency Metrics:")
        print(f"  Per Sample: {agg['latency_per_sample_mean']*1000:.2f} ± {agg['latency_per_sample_std']*1000:.2f} ms")
        print(f"  Per Batch:  {agg['latency_per_batch_mean']*1000:.2f} ± {agg['latency_per_batch_std']*1000:.2f} ms")
        print(f"\nThroughput Metrics:")
        print(f"  Samples/sec: {agg['throughput_samples_per_sec_mean']:.2f} ± {agg['throughput_samples_per_sec_std']:.2f}")
        print(f"  Batches/sec: {agg['throughput_batches_per_sec_mean']:.2f} ± {agg['throughput_batches_per_sec_std']:.2f}")
        print(f"\nMemory Usage:")
        print(f"  CPU: {agg['cpu_mem_used_mean']:.2f} ± {agg['cpu_mem_used_std']:.2f} MB")
        
        if 'gpu_mem_used_mean' in agg:
            print(f"  GPU: {agg['gpu_mem_used_mean']:.2f} ± {agg['gpu_mem_used_std']:.2f} MB")
            print(f"  Peak GPU: {agg['peak_gpu_mem_mean']:.2f} MB")
        
        if 'avg_gpu_util_mean' in agg:
            print(f"\nGPU Utilization: {agg['avg_gpu_util_mean']:.2f}%")
        
        print("="*70 + "\n")
    
    def __del__(self):
        """Cleanup NVML"""
        if self.nvml_initialized:
            try:
                pynvml.nvmlShutdown()
            except:
                pass

