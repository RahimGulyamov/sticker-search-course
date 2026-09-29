#!/usr/bin/env bash
# Run from the project root in the affected Python environment.
# Restore the CUDA 12.8 / PyTorch 2.9.1 profile originally present in this pod.
set -euo pipefail
python - <<'PY'
from pathlib import Path
p = Path('pyproject.toml')
text = p.read_text()
if '"torch==2.7.1"' in text:
    p.write_text(text.replace('"torch==2.7.1"', '"torch>=2.7.1,<2.10"'))
elif '"torch>=2.7.1,<2.10"' not in text:
    raise SystemExit('Unexpected torch requirement; inspect pyproject.toml first')
PY
# Refresh editable metadata without re-resolving unrelated packages.
python -m pip install --no-deps -e .
# XPU and CUDA Triton distributions share the triton/ package directory.
# Remove both so torch reinstalls an intact matching CUDA dependency.
python -m pip uninstall -y pytorch-triton-xpu triton
python -m pip install --upgrade \
  'torch==2.9.1+cu128' 'torchvision==0.24.1+cu128' 'torchaudio==2.9.1+cu128' \
  --index-url https://download.pytorch.org/whl/cu128
python - <<'PY'
import torch
import torchvision
print('torch:', torch.__version__, 'torchvision:', torchvision.__version__)
print('CUDA build:', torch.version.cuda)
print('CUDA available:', torch.cuda.is_available(), 'GPU count:', torch.cuda.device_count())
assert torch.version.cuda is not None, 'A CUDA torch wheel is required'
assert torch.cuda.is_available(), 'CUDA wheel installed; check nvidia-smi and GPU allocation'
for i in range(torch.cuda.device_count()):
    torch.ones(1, device=f'cuda:{i}').sum().item()
    print(i, torch.cuda.get_device_name(i))
PY
mkdir -p runs
python -m pip freeze > runs/environment_pod.txt
