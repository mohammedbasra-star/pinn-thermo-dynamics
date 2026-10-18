"""
Physics-Informed Neural Networks (PINNs) Benchmark Suite - PyTorch 2.x
Module: data
"""

from data.download_benchmarks import (
    download_dataset,
    download_all_benchmarks,
    BENCHMARK_URLS,
)
from data.dataset import (
    BurgersBenchmarkData,
    SchrodingerBenchmarkData,
    CollocationDataset,
    BoundaryDataset,
)

__all__ = [
    "download_dataset",
    "download_all_benchmarks",
    "BENCHMARK_URLS",
    "BurgersBenchmarkData",
    "SchrodingerBenchmarkData",
    "CollocationDataset",
    "BoundaryDataset",
]
