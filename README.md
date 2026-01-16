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
uv pip install torch transformers scikit-learn numpy pandas mne braindecode
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
