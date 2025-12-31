## Setup with uv

MoLeR Specific: `uv venv --python 3.10 .venv-moler && source .venv-moler/bin/activate && uv pip install molecule-generation scikit-learn numpy`

All Others: `uv venv .venv-others && source .venv-others/bin/activate && uv pip install torch transformers scikit-learn numpy`