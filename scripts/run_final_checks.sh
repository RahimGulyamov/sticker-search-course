#!/usr/bin/env bash
set -euo pipefail
source configs/pod.env
source configs/offline.env
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
python -m sticker_search.finalize_run "$@"
