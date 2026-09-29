#!/usr/bin/env bash
# Run from the project root with annotation workers stopped.
set -euo pipefail
source configs/pod.env
source configs/offline.env
mkdir -p runs/logs
python -m sticker_search.repair_translations \
  --queries "${STICKER_CATALOG:-data/catalog}/labels/queries_en.jsonl" \
  --output "${STICKER_RUN_ROOT:-runs/full_v2}/translations" \
  2>&1 | tee runs/logs/translation_fix_v1.log
bash scripts/run_gpu.sh --from index 2>&1 | tee runs/logs/full_v2_from_index.log
