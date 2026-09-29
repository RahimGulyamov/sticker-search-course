#!/usr/bin/env bash
set -euo pipefail
: "${YT_PROXY:?Set YT_PROXY to the target cluster}"
: "${STICKER_YT_DIR:?Set a new versioned YT directory}"

python -m sticker_search.prepare --nyu-count 30000 --synth-cards --output data/catalog
python -m sticker_search.audit --catalog data/catalog
python -m sticker_search.pack pack --input data/catalog --output data/bundle
python -m sticker_search.yt_transfer upload --local data/bundle --remote "$STICKER_YT_DIR"
