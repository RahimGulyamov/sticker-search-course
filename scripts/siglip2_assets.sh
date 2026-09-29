#!/usr/bin/env bash
# Download on an online laptop; restore on the offline pod. Never installs torch.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source configs/pod.env
ACTION="${1:?Usage: bash scripts/siglip2_assets.sh laptop|pod}"
REMOTE="${STICKER_SIGLIP_YT:-$STICKER_YT_ROOT/models/siglip2/v1}"
case "$ACTION" in
  laptop)
    # Offline variables from a previous terminal must not disable laptop downloads.
    unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
    python -m sticker_search.offline_models download --preset siglip2 --root data/offline_siglip2
    python -m sticker_search.offline_models pack --preset siglip2 --root data/offline_siglip2 \
      --output data/model_bundle_siglip2
    python -m sticker_search.yt_transfer upload --local data/model_bundle_siglip2 --remote "$REMOTE"
    ;;
  pod)
    ASSETS="${STICKER_SIGLIP_ASSETS:-/tmp/cv_project/siglip2_assets}"
    BUNDLE="${STICKER_SIGLIP_BUNDLE:-/tmp/cv_project/model_bundle_siglip2}"
    mkdir -p "$ASSETS" "$BUNDLE"
    python - "$ASSETS" "$BUNDLE" <<'PY'
import shutil,sys
for path in sys.argv[1:]:
    free=shutil.disk_usage(path).free/1024**3
    if free < 12:
        raise SystemExit(f'Need at least 12 GiB free for transfer at {path}; available {free:.1f}')
PY
    python -m sticker_search.yt_transfer download --local "$BUNDLE" --remote "$REMOTE"
    python -m sticker_search.file_bundle unpack --input "$BUNDLE" --output "$ASSETS"
    echo "SigLIP 2 restored under $ASSETS"
    ;;
  *) echo "Use laptop or pod" >&2; exit 2 ;;
esac
