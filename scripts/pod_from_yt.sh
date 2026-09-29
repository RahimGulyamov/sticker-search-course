#!/usr/bin/env bash
# Uses YT connectivity only. Model downloads from the internet are never invoked.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source configs/pod.env
KIND="${1:?Usage: bash scripts/pod_from_yt.sh dataset|retrieval|annotation|generation}"
case "$KIND" in
  dataset) bash scripts/restore_from_yt.sh ;;
  retrieval|annotation|generation)
    python -m sticker_search.yt_transfer download --local "data/model_bundle_$KIND" \
      --remote "$STICKER_YT_ROOT/models/$KIND/v1"
    python -m sticker_search.file_bundle unpack --input "data/model_bundle_$KIND" \
      --output data/offline_models
    ;;
  *) echo "Unknown kind: $KIND" >&2; exit 2 ;;
esac
echo 'For model inference, source configs/pod.env and then configs/offline.env in your terminal.'
