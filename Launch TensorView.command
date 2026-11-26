#!/bin/zsh
set -e
cd -- "$(dirname -- "$0")"

if [[ -d dist/TensorView.app ]]; then
    exec open dist/TensorView.app --args "$@"
fi

if [[ ! -x .venv/bin/python ]]; then
    python3 -m venv .venv
fi

if ! .venv/bin/python -c 'import h5py, ml_dtypes, onnx, pyqtgraph, safetensors, scipy, sklearn, torch, umap' >/dev/null 2>&1; then
    .venv/bin/python -m pip install -e '.[torch,scientific,embeddings]'
fi

exec .venv/bin/python -m tensorview "$@"
