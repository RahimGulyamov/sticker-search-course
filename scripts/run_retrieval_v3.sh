#!/usr/bin/env bash
set -euo pipefail
source configs/pod.env
source configs/offline.env
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
mkdir -p runs/logs
exec > >(tee runs/logs/retrieval_v3.log) 2>&1
python -m sticker_search.calibrate_retrieval
for lang in en ru; do
  echo "Evaluating calibrated retrieval on dev: $lang"
  python -m sticker_search.evaluate --index runs/full_v2/index \
    --queries "runs/full_v2/queries_$lang" --calibration runs/retrieval_v3/calibration \
    --methods rrf rrf_calibrated rrf_calibrated_ocr rrf_calibrated_all image_calibrated caption_calibrated \
    --split dev --output "runs/retrieval_v3/dev_$lang"
done
echo "Checking native CLIP on original English queries"
python -m sticker_search.native_clip_queries
python -m sticker_search.evaluate --index runs/full_v2/index \
  --queries runs/retrieval_v3/native_en_queries --methods image \
  --split dev --output runs/retrieval_v3/native_en_dev
python - <<'PY'
from pathlib import Path
import zipfile
root = Path('runs/retrieval_v3')
target = root / 'results.zip'
with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(root.rglob('*')):
        if path.is_file() and path.suffix in {'.json', '.jsonl'}:
            archive.write(path, path.relative_to(root))
print(f'Attach this file: {target}')
PY
