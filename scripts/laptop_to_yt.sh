#!/usr/bin/env bash
# Uses the active lightweight laptop environment. No torch import or installation.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source configs/pod.env
KIND="${1:?Usage: bash scripts/laptop_to_yt.sh dataset|retrieval|annotation|generation}"
case "$KIND" in
  dataset)
    python -m sticker_search.restore_catalog --output data/catalog
    python -m sticker_search.pack pack --input data/catalog --output data/bundle
    python -m sticker_search.yt_transfer upload --local data/bundle --remote "$STICKER_YT_DIR"
    ;;
  retrieval|annotation|generation)
    python -m sticker_search.offline_models download --preset "$KIND"
    python -m sticker_search.offline_models pack --preset "$KIND" --output "data/model_bundle_$KIND"
    python -m sticker_search.yt_transfer upload --local "data/model_bundle_$KIND" \
      --remote "$STICKER_YT_ROOT/models/$KIND/v1"
    ;;
  *) echo "Unknown kind: $KIND" >&2; exit 2 ;;
esac
