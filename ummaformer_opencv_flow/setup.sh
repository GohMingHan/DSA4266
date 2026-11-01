#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
reuse=""
if [[ $# -gt 0 ]]; then
  if [[ $# -ne 2 || "$1" != "--reuse-upstream" ]]; then
    echo 'Usage: bash setup.sh [--reuse-upstream /path/to/existing/upstream]' >&2
    exit 2
  fi
  reuse="$(realpath "$2")"
fi
python -c 'import torch; print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda)'
python -m pip install -r requirements.txt
if ! python -c 'import cv2; assert hasattr(cv2, "calcOpticalFlowFarneback")' ; then
  python -m pip install 'opencv-python-headless>=4.8,<4.12'
fi
if ! python -c 'import torch, torchvision; from torchvision.models import resnet18, ResNet18_Weights; print("torchvision:", torchvision.__version__)'; then
  cat >&2 <<'MSG'
A matching torchvision installation is required for RGB extraction.
For your recorded torch 2.7.1+cu128 environment ONLY, install:
  python -m pip install --no-deps torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
Then rerun this setup. Do not change a working PyTorch/CUDA installation.
For another torch version, use the matching official torchvision release instead.
MSG
  exit 2
fi
if [[ -n "$reuse" ]]; then
  [[ -f "$reuse/libs/modeling/av_recoverynonorm_meta_arch.py" ]] || { echo 'Invalid upstream path.' >&2; exit 2; }
  if [[ -e upstream || -L upstream ]]; then
    [[ "$(realpath upstream)" == "$reuse" ]] || { echo 'upstream points elsewhere; not overwriting.' >&2; exit 2; }
  else
    ln -s -- "$reuse" upstream
  fi
elif [[ ! -d upstream ]]; then
  git clone https://github.com/ymhzyj/UMMAFormer.git upstream
  git -C upstream rev-parse HEAD > upstream_revision.txt
fi
# Reuse previously compiled dependencies. Build CPU temporal NMS only when absent.
if ! python -c 'import sys; sys.path.insert(0,"upstream/libs/utils"); import nms_1d_cpu'; then
  MAX_JOBS="${MAX_JOBS:-1}" python -m pip install --no-build-isolation --no-deps ./upstream/libs/utils
fi
if ! python -c 'import torchsort'; then
  cat >&2 <<'MSG'
torchsort is still missing. This is an upstream dependency, not a Farneback dependency.
Use your previously configured matching CUDA toolkit, then:
  MAX_JOBS=1 python -m pip install --no-deps --no-build-isolation torchsort==0.1.10
Do not let pip build isolation pull a different PyTorch version.
MSG
  exit 2
fi
python - <<'PY'
from pathlib import Path
from upstream_adapter import load_official
_, _, _, identity = load_official(Path('upstream'))
print('Official import OK:', identity['git_commit'])
PY
python -m pytest -q tests
printf '\nNext: run the real-model synthetic smoke test, then a small extraction pilot.\n'
