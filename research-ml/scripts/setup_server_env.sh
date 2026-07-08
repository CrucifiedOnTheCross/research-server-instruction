#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="${VENV_DIR:-.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

"$PYTHON_BIN" -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip wheel setuptools

# Current PyTorch Linux wheels include CUDA runtime dependencies when installed
# from the proper PyTorch/PyPI channel. Override TORCH_INDEX_URL when pinning a
# specific CUDA build, for example:
#   TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 ./scripts/setup_server_env.sh
if [[ -n "${TORCH_INDEX_URL:-}" ]]; then
  python -m pip install torch torchvision --index-url "$TORCH_INDEX_URL"
else
  python -m pip install torch torchvision
fi

python -m pip install -r requirements.txt

python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda available", torch.cuda.is_available())
print("cuda runtime", torch.version.cuda)
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
    x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    y = x @ x.T
    torch.cuda.synchronize()
    print("bf16 matmul ok", tuple(y.shape))
PY

