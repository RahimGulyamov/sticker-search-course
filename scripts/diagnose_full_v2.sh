#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
mkdir -p runs/logs
python -m sticker_search.diagnose_retrieval \
  --index runs/full_v2/index --queries runs/full_v2/queries \
  --output runs/full_v2/diagnostics_v1 \
  2>&1 | tee runs/logs/diagnostics_v1.log
python - <<'PY'
from pathlib import Path
import zipfile
root = Path('runs/full_v2')
files = ['diagnostics_v1/summary.json', 'diagnostics_v1/summary.md', 'diagnostics_v1/cases.jsonl',
         'dev_en/metrics.json', 'dev_en/cases.jsonl', 'dev_ru/metrics.json', 'dev_ru/cases.jsonl',
         'adapter/summary.json', 'adapter/history.jsonl', 'adapter/config.json',
         'translations/translation_corrections_audit.json']
target = root / 'retrieval_diagnostics.zip'
with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as archive:
    for relative in files:
        path = root / relative
        if path.exists():
            archive.write(path, relative)
print(f'Attach this file: {target} ({target.stat().st_size / 1024**2:.1f} MiB)')
PY
