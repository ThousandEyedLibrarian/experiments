"""Logging utilities for HPC experiment tracking."""

import logging
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional


def setup_logging(
    experiment_name: str,
    log_dir: str = "logs",
    level: int = logging.INFO,
    console_level: int = logging.INFO,
) -> logging.Logger:
    """Set up logging with both file and console output.

    Args:
        experiment_name: Name for the log file.
        log_dir: Directory for log files.
        level: File logging level.
        console_level: Console logging level.

    Returns:
        Configured logger.
    """
    # Create logs directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Create timestamped log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_path / f"{experiment_name}_{timestamp}.log"

    # Create logger
    logger = logging.getLogger(experiment_name)
    logger.setLevel(logging.DEBUG)  # Capture all levels

    # Clear any existing handlers
    logger.handlers.clear()

    # File handler - detailed output
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_format = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)

    # Console handler - cleaner output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_format = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # Log the log file path
    logger.info(f"Logging to: {log_file}")

    return logger


def log_environment_info(logger: logging.Logger) -> None:
    """Log environment information for debugging HPC issues."""
    logger.info("=" * 60)
    logger.info("ENVIRONMENT INFORMATION")
    logger.info("=" * 60)

    # System info
    import platform
    logger.info(f"Hostname: {platform.node()}")
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Python: {sys.version}")

    # SLURM info
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "N/A (not running under SLURM)")
    slurm_job_name = os.environ.get("SLURM_JOB_NAME", "N/A")
    slurm_nodelist = os.environ.get("SLURM_NODELIST", "N/A")
    logger.info(f"SLURM Job ID: {slurm_job_id}")
    logger.info(f"SLURM Job Name: {slurm_job_name}")
    logger.info(f"SLURM Nodes: {slurm_nodelist}")

    # PyTorch and CUDA
    try:
        import torch
        logger.info(f"PyTorch: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"cuDNN version: {torch.backends.cudnn.version()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                mem_total = props.total_memory / (1024**3)
                logger.info(f"GPU {i}: {props.name} ({mem_total:.1f} GB)")
    except ImportError as e:
        logger.error(f"PyTorch import failed: {e}")
    except Exception as e:
        logger.warning(f"Error getting PyTorch/CUDA info: {e}")

    # Key package versions
    packages = ["numpy", "sklearn", "mne", "pandas"]
    for pkg in packages:
        try:
            mod = __import__(pkg)
            version = getattr(mod, "__version__", "unknown")
            logger.info(f"{pkg}: {version}")
        except ImportError:
            logger.warning(f"{pkg}: NOT INSTALLED")

    # Check braindecode (the problematic import)
    try:
        import braindecode
        logger.info(f"braindecode: {braindecode.__version__}")
    except ImportError as e:
        logger.warning(f"braindecode: NOT AVAILABLE - {e}")
    except Exception as e:
        logger.warning(f"braindecode: IMPORT ERROR - {e}")

    logger.info("=" * 60)


def log_gpu_memory(logger: logging.Logger, prefix: str = "") -> None:
    """Log current GPU memory usage."""
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / (1024**3)
                reserved = torch.cuda.memory_reserved(i) / (1024**3)
                logger.debug(f"{prefix}GPU {i} memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    except Exception:
        pass  # Silently ignore if we can't get memory info


def log_exception(logger: logging.Logger, e: Exception, context: str = "") -> None:
    """Log an exception with full traceback."""
    if context:
        logger.error(f"{context}: {type(e).__name__}: {e}")
    else:
        logger.error(f"{type(e).__name__}: {e}")
    logger.error(f"Traceback:\n{traceback.format_exc()}")


def get_logger(name: str = "exp2") -> logging.Logger:
    """Get an existing logger or create a basic one if not set up."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        # Create a basic console handler if not configured
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%H:%M:%S",
        ))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
