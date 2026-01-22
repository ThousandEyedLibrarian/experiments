# ASM Outcome Prediction Experiments

Multimodal fusion experiments for predicting anti-seizure medication (ASM) treatment outcomes by combining embeddings from clinical text reports, EEG signals, and drug molecular structures (SMILES).

See `OBJECTIVES.md` for detailed experiment goals and methodology.

## Prerequisites

- Python 3.10
- [uv](https://github.com/astral-sh/uv) package manager
- CUDA-capable GPU (recommended, 8GB+ VRAM for full experiments)
- ~10GB disk space for dependencies

## Quick Start

```bash
# Clone the repository
git clone https://github.com/ThousandEyedLibrarian/experiments
cd experiments

# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Environment Setup

This project uses **two separate virtual environments** due to TensorFlow/PyTorch dependency conflicts.

### Main Environment (PyTorch-based)

Used for most experiments (Exp 1, Exp 2):

```bash
uv venv --python 3.10 .venv-others
source .venv-others/bin/activate

# For GPU support (recommended) - check CUDA version with: nvidia-smi
# CUDA 11.8:
uv pip install torch --index-url https://download.pytorch.org/whl/cu118
# CUDA 12.1:
uv pip install torch --index-url https://download.pytorch.org/whl/cu121

# For CPU only:
# uv pip install torch

# Install remaining dependencies
uv pip install transformers scikit-learn numpy pandas mne braindecode
```

### MoLeR Environment (TensorFlow-based)

Only needed for MoLeR molecular embeddings:

```bash
uv venv --python 3.10 .venv-moler
source .venv-moler/bin/activate
uv pip install "rdkit" "tensorflow<2.10" numpy molecule-generation
```

## Data Setup

Data is not included in this repository for privacy reasons. The expected data structure:

```
../asm_data/
├── alfred_1st_regimen.csv      # Patient metadata with outcomes
└── Alfred/
    └── EEG/
        └── *.edf               # EEG recordings (157 files)
```

The CSV should contain columns: `pid`, `outcome` (1=failure, 2=success), `ASM`, `eeg_report`, etc.

## Running Experiments

### HPC Setup (GPU Nodes)

On HPC systems, experiments must run on GPU compute nodes, not login nodes.

**Using the submission script (recommended):**

A pre-configured SLURM script `submit_job.sh` is included for M3/Monash HPC:

```bash
# 1. Edit configuration in submit_job.sh:
#    - EXPERIMENT="exp1" or "exp2"
#    - EXTRA_ARGS for additional flags
#    - SBATCH directives (time, memory, GPU type)

# 2. Test without running (dry-run mode):
DRY_RUN=true bash submit_job.sh

# 3. Submit the job:
sbatch submit_job.sh

# 4. Monitor:
squeue -u $USER

# 5. Check logs:
cat logs/asm_<jobid>.out
```

**Script configuration options:**
- Default: 4 hours, 32GB RAM, 1 GPU (any type)
- For specific GPU: edit `#SBATCH --gres=gpu:A100:1` or `--gres=gpu:A40:1`
- For email alerts: uncomment `--mail-user` and `--mail-type` lines

**Interactive session (alternative):**
```bash
srun --gres=gpu:1 --partition=gpu --mem=32G --time=4:00:00 --pty bash
source .venv-others/bin/activate
python -m exp1_fusion.run_experiments
```

### Experiment 1: LLM + SMILES Fusion

Combines clinical text embeddings with drug molecular embeddings.

```bash
source .venv-others/bin/activate

# Preview what experiments will run
python -m exp1_fusion.run_experiments --dry-run

# Run all Experiment 1 combinations (8 experiments, 5-fold CV each)
python -m exp1_fusion.run_experiments
```

### Experiment 2: EEG + SMILES Fusion

Combines EEG signal embeddings with drug molecular embeddings.

```bash
source .venv-others/bin/activate

# Preview experiments
python -m exp2_fusion.run_experiments --dry-run

# Run with SimpleCNN encoder (works on 8GB GPU)
python -m exp2_fusion.run_experiments --eeg-encoder simplecnn

# Run with specific SMILES model
python -m exp2_fusion.run_experiments --eeg-encoder simplecnn --smiles-model chemberta
```

### Embedding Generation (Optional)

Generate embeddings separately using scripts in `exp1_misc/`:

```bash
source .venv-others/bin/activate

# Generate ChemBERTa SMILES embeddings
python exp1_misc/e1_LLM+SMILES_ChemBERTa.py

# Generate SMILES Transformer embeddings
python exp1_misc/e1_LLM+SMILES_SMILESTransformer.py

# Generate text embeddings with ClinicalBERT/PubMedBERT
python exp1_misc/e1_LLM+SMILES_Bert.py
python exp1_misc/e1_LLM+SMILES_PubMedBert.py
```

## Output Structure

```
outputs/
├── chemberta_asm_embeddings.npy    # SMILES embeddings (ChemBERTa)
├── smilestrf_asm_embeddings.npy    # SMILES embeddings (SMILES Transformer)
├── exp1_results/                    # Experiment 1 results
│   ├── summary.json                # All experiments summary
│   └── exp1a_*.json               # Individual experiment results
├── exp2_results/                    # Experiment 2 results
│   ├── summary.json
│   └── exp2_*.json
└── eeg_cache/                       # Cached preprocessed EEG data
    └── processed_eeg.pkl
```

### Results Format

Each experiment JSON contains:
```json
{
  "experiment": "exp1b_clinicalbert_chemberta",
  "accuracy": {"mean": 0.61, "std": 0.08, "per_fold": [...]},
  "auc": {"mean": 0.65, "std": 0.08, "per_fold": [...]},
  "f1": {"mean": 0.62, "std": 0.13, "per_fold": [...]}
}
```

## Results

High-level findings and analysis are in the `findings/` folder.

## Troubleshooting

### PyTorch Not Using GPU

If experiments show "Device: cpu" instead of "cuda":

**1. On HPC systems: You're likely on a login node**

Login nodes don't have GPUs. Request a GPU compute node:

```bash
# Interactive GPU session (adjust for your HPC system)
# For SLURM-based systems:
srun --gres=gpu:1 --partition=gpu --time=2:00:00 --pty bash

# Or using smux (M3/Monash):
smux new-session --gres=gpu:1 --partition=gpu --mem=32G --time=2:00:00

# Verify GPU is accessible:
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

**2. PyTorch installed without CUDA support**

```bash
# Check if CUDA is available
python -c "import torch; print(torch.cuda.is_available())"

# If False on a GPU node, reinstall PyTorch with CUDA:
source .venv-others/bin/activate
uv pip uninstall torch
uv pip install torch --index-url https://download.pytorch.org/whl/cu118  # or cu121
```

### CUDA Out of Memory

Reduce batch size in the config files:
- `exp1_fusion/config.py` - `TRAIN_CONFIG["batch_size"]`
- `exp2_fusion/config.py` - `BATCH_SIZE_BY_ENCODER`

### EEG Loading Errors

The pipeline automatically handles encoding issues in EDF files by trying multiple encodings (UTF-8, Latin-1).

### Missing Dependencies

If you encounter import errors, ensure you've installed all packages:

```bash
source .venv-others/bin/activate
uv pip install torch transformers scikit-learn numpy pandas mne braindecode
```

## Project Structure

```
experiments/
├── exp1_fusion/          # Experiment 1: LLM + SMILES fusion
│   ├── run_experiments.py
│   ├── training.py
│   └── models/
├── exp2_fusion/          # Experiment 2: EEG + SMILES fusion
│   ├── run_experiments.py
│   ├── eeg_pipeline.py
│   └── models/
├── exp1_misc/            # Embedding generation scripts
├── outputs/              # Generated embeddings and results
├── findings/             # Analysis and notes
├── smiles-transformer/   # External SMILES transformer code
└── MoLeR_checkpoint/     # Pre-trained MoLeR model
```
