#!/usr/bin/env bash
set -euo pipefail
: "${YT_PROXY:?Set YT_PROXY to the target cluster}"
: "${STICKER_YT_DIR:?Set the prepared YT directory}"
python -m sticker_search.yt_transfer download --local data/bundle --remote "$STICKER_YT_DIR"
python -m sticker_search.pack unpack --input data/bundle --output data/catalog
