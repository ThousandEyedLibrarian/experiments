## Setup with uv

MoLeR Specific: `uv venv --python 3.10 .venv-moler && source .venv-moler/bin/activate && uv pip install --python .venv-moler "rdkit" "tensorflow<2.10" numpy molecule-generation`

All Others: `uv venv --python 3.10 .venv-others && source .venv-others/bin/activate && uv pip install torch transformers scikit-learn numpy`