#!/usr/bin/env bash
# Full train run, automatic checkpoint resume; no internet and no pip install.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source configs/pod.env
source configs/offline.env
CONFIG="${STICKER_SIGLIP_CONFIG:-configs/siglip2_full.json}"
WORKERS="${SIGLIP_GPU_WORKERS:-4}"
CATALOG="${STICKER_CATALOG:-data/catalog}"
ASSETS="${STICKER_SIGLIP_ASSETS:-/tmp/cv_project/siglip2_assets}"
MODEL="${STICKER_SIGLIP_MODEL:-$ASSETS/hub/models--google--siglip2-so400m-patch14-384/snapshots/e8e487298228002f3d8a82e0cd5c8ea9c567f57f}"
RUN="${STICKER_SIGLIP_RUN:-/tmp/cv_project/siglip2_v1}"
ANNOTATIONS="${STICKER_SIGLIP_ANNOTATIONS:-runs/full_v2/captions/annotations.jsonl}"
TRANSLATIONS="${STICKER_SIGLIP_TRANSLATIONS:-runs/full_v2/translations/queries_ru.jsonl}"
mkdir -p "$RUN"

python - "$WORKERS" "$MODEL" "$RUN" <<'PY'
import json,shutil,sys
from pathlib import Path
import torch,transformers
workers,model,run=int(sys.argv[1]),Path(sys.argv[2]),Path(sys.argv[3])
assert torch.cuda.is_available(), 'CUDA is unavailable'
assert 1 <= workers <= torch.cuda.device_count(), 'Invalid GPU worker count'
assert (model/'model.safetensors').is_file(), f'Restore SigLIP 2 assets first: {model}'
architecture=json.loads((model/'config.json').read_text())
assert architecture['model_type']=='siglip' and architecture['vision_config']['image_size']==384
assert architecture['vision_config']['hidden_size']==1152 and architecture['text_config']['num_hidden_layers']==27
assert transformers.__version__ == '4.57.1', 'Use the existing project transformers==4.57.1'
free=shutil.disk_usage(run).free/1024**3
assert free >= 60, f'Need 60 GiB free for model/optimizer checkpoints; available {free:.1f} at {run}'
print(f'Preflight: torch={torch.__version__}, GPUs={workers}, free={free:.1f} GiB',flush=True)
for i in range(workers):
    print(i,torch.cuda.get_device_name(i),flush=True)
    torch.ones(1,device=f'cuda:{i}').sum().item()
PY
python -m sticker_search.siglip.data --catalog "$CATALOG" --annotations "$ANNOTATIONS" \
  --translated "$TRANSLATIONS" --output "$RUN/prepared" --config "$CONFIG"

python -m torch.distributed.run --standalone --nproc_per_node="$WORKERS" \
  -m sticker_search.siglip.train --model "$MODEL" --config "$CONFIG" --catalog "$CATALOG" \
  --prepared "$RUN/prepared" --output "$RUN/smoke" --smoke --resume

python -m torch.distributed.run --standalone --nproc_per_node="$WORKERS" \
  -m sticker_search.siglip.train --model "$MODEL" --config "$CONFIG" --catalog "$CATALOG" \
  --prepared "$RUN/prepared" --output "$RUN" --resume
python scripts/collect_siglip2_results.py --run "$RUN" --catalog "$CATALOG" \
  --output "${STICKER_SIGLIP_RESULTS:-runs/siglip2_v1_results.zip}"
echo "Finished: $RUN/selection.json, $RUN/history.jsonl, $RUN/complete.json"
echo 'Test was not evaluated. Use the dev results before the final test commands in docs/SIGLIP2.md.'
