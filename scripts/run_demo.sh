#!/usr/bin/env bash
set -euo pipefail
source configs/pod.env
source configs/offline.env
export STICKER_INDEX='runs/full_v2/index'
export STICKER_CHECKPOINT='runs/full_v2/adapter/best.pt'
export STICKER_ENABLE_GENERATION=1
python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
