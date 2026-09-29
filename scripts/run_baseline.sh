#!/usr/bin/env bash
# Full-catalog frozen baseline. Development split only; no test-set tuning.
set -euo pipefail
CATALOG="${STICKER_CATALOG:-data/catalog}"
DEVICE="${STICKER_DEVICE:-cuda}"
OUT="${STICKER_BASELINE_RUN:-runs/full_baseline_v1}"
python -m sticker_search.features index --catalog "$CATALOG" \
  --output "$OUT/index" --device "$DEVICE" --batch-size 128
python -m sticker_search.features queries --queries "$CATALOG/labels/queries_en.jsonl" \
  --output "$OUT/queries" --device "$DEVICE" --batch-size 128
python -m sticker_search.evaluate --index "$OUT/index" --queries "$OUT/queries" \
  --methods image --split dev --output "$OUT/dev"
