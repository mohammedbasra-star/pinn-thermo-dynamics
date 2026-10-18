"""
Physics-Informed Neural Networks (PINNs) Benchmark Suite - PyTorch 2.x
Module: data/download_benchmarks.py

Automated download utility for the historic benchmark datasets from
Maziar Raissi et al. (2019, Journal of Computational Physics):
1. 1D Burgers' Equation: 'burgers_shock.mat'
2. 1D Nonlinear Schrödinger Equation: 'NLS.mat'
3. 2D Incompressible Navier-Stokes (Cylinder Wake): 'cylinder_nektar_wake.mat'
"""

import os
import sys
import argparse
import urllib.request
from typing import Dict, List, Optional
import scipy.io as sio


BENCHMARK_URLS: Dict[str, Dict[str, str]] = {
    "burgers": {
        "filename": "burgers_shock.mat",
        "url": "https://raw.githubusercontent.com/maziarraissi/PINNs/master/appendix/Data/burgers_shock.mat",
        "description": "Burgers' Equation shock benchmark (x: 256, t: 100, usol: 256x100)",
        "expected_keys": ["x", "t", "usol"],
    },
    "schrodinger": {
        "filename": "NLS.mat",
        "url": "https://raw.githubusercontent.com/maziarraissi/PINNs/master/main/Data/NLS.mat",
        "description": "Nonlinear Schrödinger (NLS) soliton benchmark (x: 1x256, tt: 1x201, uu: 256x201)",
        "expected_keys": ["x", "tt", "uu"],
    },
    "navier_stokes": {
        "filename": "cylinder_nektar_wake.mat",
        "url": "https://raw.githubusercontent.com/maziarraissi/PINNs/master/main/Data/cylinder_nektar_wake.mat",
        "description": "2D Navier-Stokes cylinder wake benchmark (X_star: 5000x2, t: 200x1, U_star, p_star)",
        "expected_keys": ["X_star", "t", "U_star", "p_star"],
    },
}


class DownloadProgressBar:
    """Console progress tracker for dataset downloads."""

    def __init__(self, filename: str):
        self.filename = filename
        self.total_size = 0
        self.downloaded = 0

    def __call__(self, block_num: int, block_size: int, total_size: int):
        self.total_size = total_size
        self.downloaded += block_size
        if total_size > 0:
            percent = min(100.0, self.downloaded * 100.0 / total_size)
            mb_downloaded = self.downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            sys.stdout.write(
                f"\r  [FETCH] {self.filename}: {percent:5.1f}% ({mb_downloaded:6.2f} MB / {mb_total:6.2f} MB)"
            )
            sys.stdout.flush()
        else:
            mb_downloaded = self.downloaded / (1024 * 1024)
            sys.stdout.write(f"\r  [FETCH] {self.filename}: {mb_downloaded:6.2f} MB downloaded")
            sys.stdout.flush()


def download_file(url: str, destination: str, description: str, force: bool = False) -> str:
    """
    Downloads a single benchmark dataset file with progress tracking and validation.
    """
    filename = os.path.basename(destination)
    if os.path.exists(destination) and not force:
        print(f"[FOUND] {filename} already exists at: {destination}")
        return destination

    os.makedirs(os.path.dirname(os.path.abspath(destination)), exist_ok=True)
    print(f"[DOWNLOADING] {description}")
    print(f"  URL: {url}")
    print(f"  Target: {destination}")

    progress_bar = DownloadProgressBar(filename)
    try:
        urllib.request.urlretrieve(url, destination, reporthook=progress_bar)
        print()  # Newline after carriage return
        print(f"  [SUCCESS] Downloaded {filename} ({os.path.getsize(destination) / (1024*1024):.2f} MB)")
    except Exception as exc:
        print()
        if os.path.exists(destination):
            os.remove(destination)
        raise RuntimeError(f"Failed to download benchmark from {url}: {exc}") from exc

    return destination


def verify_dataset(filepath: str, expected_keys: List[str]) -> bool:
    """Verifies that the downloaded .mat file can be loaded and contains expected keys."""
    try:
        data = sio.loadmat(filepath)
        missing = [k for k in expected_keys if k not in data]
        if missing:
            print(f"  [WARNING] File {filepath} is missing expected keys: {missing}")
            return False
        key_shapes = {k: data[k].shape for k in expected_keys}
        print(f"  [VERIFIED] {os.path.basename(filepath)} verified successfully. Shapes: {key_shapes}")
        return True
    except Exception as exc:
        print(f"  [ERROR] Verification failed for {filepath}: {exc}")
        return False


def download_dataset(
    name: str,
    target_dir: str = "data/benchmarks",
    force: bool = False,
) -> str:
    """
    Download a specific benchmark dataset by name ('burgers', 'schrodinger', 'navier_stokes').
    """
    if name not in BENCHMARK_URLS:
        raise ValueError(f"Unknown benchmark dataset '{name}'. Available: {list(BENCHMARK_URLS.keys())}")

    info = BENCHMARK_URLS[name]
    target_path = os.path.join(target_dir, info["filename"])
    download_file(
        url=info["url"],
        destination=target_path,
        description=info["description"],
        force=force,
    )
    verify_dataset(target_path, info["expected_keys"])
    return target_path


def download_all_benchmarks(
    target_dir: str = "data/benchmarks",
    force: bool = False,
) -> Dict[str, str]:
    """Download all historic benchmark datasets."""
    paths = {}
    print("=" * 70)
    print("Downloading Raissi et al. (2019) Historic PINN Benchmark Datasets")
    print("=" * 70)
    for name in BENCHMARK_URLS:
        paths[name] = download_dataset(name, target_dir=target_dir, force=force)
        print("-" * 70)
    return paths


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download historic Raissi et al. (2019) PINN benchmark datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["all", "burgers", "schrodinger", "navier_stokes"],
        default="burgers",
        help="Benchmark dataset to download",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/benchmarks",
        help="Target directory to store .mat files",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if files exist",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.dataset == "all":
        download_all_benchmarks(target_dir=args.data_dir, force=args.force)
    else:
        download_dataset(name=args.dataset, target_dir=args.data_dir, force=args.force)
