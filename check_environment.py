#!/usr/bin/env python3
"""Environment validation script for HPC experiments.

Run this before experiments to catch dependency and environment issues early.
Usage: python check_environment.py [--exp1] [--exp2]
"""

import argparse
import os
import sys
from pathlib import Path


class Colors:
    """ANSI color codes for terminal output."""
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    END = "\033[0m"


def ok(msg: str) -> None:
    print(f"  {Colors.GREEN}[OK]{Colors.END} {msg}")


def warn(msg: str) -> None:
    print(f"  {Colors.YELLOW}[WARN]{Colors.END} {msg}")


def fail(msg: str) -> None:
    print(f"  {Colors.RED}[FAIL]{Colors.END} {msg}")


def header(msg: str) -> None:
    print(f"\n{Colors.BOLD}=== {msg} ==={Colors.END}")


def check_python_version() -> bool:
    """Check Python version is compatible."""
    header("Python Version")
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"

    if version.major == 3 and version.minor >= 9:
        ok(f"Python {version_str}")
        return True
    else:
        fail(f"Python {version_str} (need >= 3.9)")
        return False


def check_cuda() -> bool:
    """Check CUDA availability and version."""
    header("CUDA / GPU")
    success = True

    try:
        import torch
        ok(f"PyTorch {torch.__version__}")

        if torch.cuda.is_available():
            ok(f"CUDA available (version {torch.version.cuda})")
            ok(f"cuDNN version {torch.backends.cudnn.version()}")

            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                mem_gb = props.total_memory / (1024**3)
                ok(f"GPU {i}: {props.name} ({mem_gb:.1f} GB)")

            # Test CUDA operations work
            try:
                x = torch.randn(10, 10).cuda()
                y = x @ x.T
                del x, y
                torch.cuda.empty_cache()
                ok("CUDA tensor operations work")
            except Exception as e:
                fail(f"CUDA tensor operations failed: {e}")
                success = False
        else:
            warn("CUDA not available - will use CPU (slow)")
            success = True  # Not a fatal error

    except ImportError as e:
        fail(f"PyTorch import failed: {e}")
        success = False
    except Exception as e:
        fail(f"Error checking CUDA: {e}")
        success = False

    return success


def check_core_packages() -> bool:
    """Check core dependencies are importable."""
    header("Core Packages")
    success = True

    packages = [
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("sklearn", "scikit-learn"),
        ("mne", "mne"),
    ]

    for import_name, display_name in packages:
        try:
            mod = __import__(import_name)
            version = getattr(mod, "__version__", "unknown")
            ok(f"{display_name} {version}")
        except ImportError as e:
            fail(f"{display_name}: {e}")
            success = False

    return success


def check_exp1_packages() -> bool:
    """Check Experiment 1 specific packages."""
    header("Experiment 1 Packages")
    success = True

    packages = [
        ("transformers", "transformers"),
        ("torch", "torch"),
    ]

    for import_name, display_name in packages:
        try:
            mod = __import__(import_name)
            version = getattr(mod, "__version__", "unknown")
            ok(f"{display_name} {version}")
        except ImportError as e:
            fail(f"{display_name}: {e}")
            success = False

    return success


def check_exp2_packages() -> bool:
    """Check Experiment 2 specific packages (EEG processing)."""
    header("Experiment 2 Packages")
    success = True

    # Check MNE (required)
    try:
        import mne
        ok(f"mne {mne.__version__}")
    except ImportError as e:
        fail(f"mne: {e}")
        success = False

    # Check braindecode (optional, for LaBraM encoder)
    try:
        import braindecode
        ok(f"braindecode {braindecode.__version__}")

        # Try importing LaBraM specifically
        try:
            from braindecode.models import Labram
            ok("braindecode.models.Labram importable")
        except ImportError as e:
            warn(f"Labram import failed: {e}")
            warn("LaBraM encoder will not be available, use --eeg-encoder simplecnn")

    except ImportError as e:
        warn(f"braindecode not available: {e}")
        warn("LaBraM encoder will not be available, use --eeg-encoder simplecnn")
        # Not a fatal error - SimpleCNN can be used instead
    except Exception as e:
        warn(f"braindecode import error: {e}")
        warn("LaBraM encoder may not work correctly")

    return success


def check_data_files() -> bool:
    """Check required data files exist."""
    header("Data Files")
    success = True

    # Get the experiments directory
    script_dir = Path(__file__).parent.resolve()

    # Check for SMILES embeddings
    embeddings_files = [
        "exp1_outputs/asm_smiles_embeddings_chemberta.npy",
        "exp1_outputs/asm_smiles_embeddings_smilestrf.npy",
    ]

    for emb_file in embeddings_files:
        path = script_dir / emb_file
        if path.exists():
            ok(f"Found {emb_file}")
        else:
            warn(f"Missing {emb_file} (run Experiment 1 first)")

    # Check for EEG data directory
    eeg_dirs = [
        script_dir / "data" / "eeg",
        script_dir.parent / "data" / "eeg",
    ]

    found_eeg = False
    for eeg_dir in eeg_dirs:
        if eeg_dir.exists() and list(eeg_dir.glob("*.edf")):
            n_files = len(list(eeg_dir.glob("*.edf")))
            ok(f"Found EEG directory with {n_files} EDF files: {eeg_dir}")
            found_eeg = True
            break

    if not found_eeg:
        warn("EEG data directory not found or empty")
        success = False

    # Check for CSV file
    csv_paths = [
        script_dir / "data" / "asm_outcomes.csv",
        script_dir.parent / "data" / "asm_outcomes.csv",
    ]

    found_csv = False
    for csv_path in csv_paths:
        if csv_path.exists():
            ok(f"Found outcomes CSV: {csv_path}")
            found_csv = True
            break

    if not found_csv:
        warn("Outcomes CSV not found")

    return success


def check_slurm_env() -> bool:
    """Check SLURM environment variables if running under SLURM."""
    header("SLURM Environment")

    if "SLURM_JOB_ID" in os.environ:
        ok(f"SLURM_JOB_ID: {os.environ['SLURM_JOB_ID']}")
        ok(f"SLURM_JOB_NAME: {os.environ.get('SLURM_JOB_NAME', 'N/A')}")
        ok(f"SLURM_NODELIST: {os.environ.get('SLURM_NODELIST', 'N/A')}")
        ok(f"SLURM_GPUS_ON_NODE: {os.environ.get('SLURM_GPUS_ON_NODE', 'N/A')}")
        return True
    else:
        warn("Not running under SLURM (local execution)")
        return True


def main():
    parser = argparse.ArgumentParser(description="Validate environment for experiments")
    parser.add_argument("--exp1", action="store_true", help="Check Experiment 1 requirements")
    parser.add_argument("--exp2", action="store_true", help="Check Experiment 2 requirements")
    parser.add_argument("--all", action="store_true", help="Check all requirements")
    args = parser.parse_args()

    # Default to checking all if nothing specified
    if not args.exp1 and not args.exp2:
        args.all = True

    print(f"{Colors.BOLD}Environment Validation{Colors.END}")
    print("=" * 50)

    all_passed = True

    # Always check these
    all_passed &= check_python_version()
    all_passed &= check_cuda()
    all_passed &= check_core_packages()
    check_slurm_env()  # Informational only

    # Experiment-specific checks
    if args.exp1 or args.all:
        all_passed &= check_exp1_packages()

    if args.exp2 or args.all:
        all_passed &= check_exp2_packages()

    all_passed &= check_data_files()

    # Summary
    print()
    print("=" * 50)
    if all_passed:
        print(f"{Colors.GREEN}{Colors.BOLD}All checks passed!{Colors.END}")
        sys.exit(0)
    else:
        print(f"{Colors.RED}{Colors.BOLD}Some checks failed - see above for details{Colors.END}")
        sys.exit(1)


if __name__ == "__main__":
    main()
