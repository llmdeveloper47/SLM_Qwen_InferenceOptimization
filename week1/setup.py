"""
Setup script for Week 1: Baseline Profiling
"""

from setuptools import setup, find_packages

setup(
    name="week1-baseline-profiling",
    version="1.0.0",
    description="Week 1: Baseline profiling and bottleneck analysis for Qwen 2.5 0.5B inference optimization",
    author="ML Engineering Course",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "datasets>=2.14.0",
        "huggingface-hub>=0.19.0",
        "nvidia-ml-py3>=7.352.0",
        "psutil>=5.9.0",
        "py3nvml>=0.2.7",
        "memory_profiler>=0.61.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "tqdm>=4.66.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "pyyaml>=6.0",
        "python-dotenv>=1.0.0",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)

