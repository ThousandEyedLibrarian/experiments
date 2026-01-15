## Setup with uv

MoLeR Specific: `uv venv --python 3.10 .venv-moler && source .venv-moler/bin/activate && uv pip install --python .venv-moler "rdkit" "tensorflow<2.10" numpy molecule-generation`

All Others: `uv venv --python 3.10 .venv-others && source .venv-others/bin/activate && uv pip install torch transformers scikit-learn numpy`

## Results

Results can be examined at a high level in the `findings/` folder as Markdown notes.

## Data

For privacy reasons I have not included the data used directly in the repository.