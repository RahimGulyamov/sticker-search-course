#!/usr/bin/env bash
# Run from the repository root after sourcing pod.env and offline.env.
# v2 audit: captions/OCR are weak features; VLM usage queries are NOT training labels.
set -euo pipefail
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM=false
CATALOG="${STICKER_CATALOG:-data/catalog}"
WORKERS="${STICKER_GPU_WORKERS:-4}"
RUN_ROOT="${STICKER_RUN_ROOT:-runs/full_v2}"
# Resume a completed earlier stage explicitly; never mutate its saved artifacts.
START_FROM="caption"
if [[ $# -gt 0 ]]; then
  if [[ $# -ne 2 || "$1" != "--from" ]]; then
    echo "Usage: bash scripts/run_gpu.sh [--from caption|translate|index|train|evaluate]" >&2
    exit 2
  fi
  START_FROM="$2"
fi
case "$START_FROM" in
  caption) START_STEP=0 ;;
  translate) START_STEP=1 ;;
  index) START_STEP=2 ;;
  train) START_STEP=3 ;;
  evaluate) START_STEP=4 ;;
  *) echo "Unknown start stage: $START_FROM" >&2; exit 2 ;;
esac
mkdir -p "$RUN_ROOT"

# Catch a stale annotation module before the long GPU run.
python - "$CATALOG" "$WORKERS" <<'PY'
import json
import sys
from pathlib import Path
import torch
from sticker_search.annotate import CAPTION_PROMPT, PARSER_VERSION, recover_errors
from sticker_search.common import fingerprint, sha256
catalog = Path(sys.argv[1])
assert torch.cuda.is_available(), "CUDA is unavailable"
assert 0 < int(sys.argv[2]) <= torch.cuda.device_count(), "Invalid GPU worker count"
assert "STICKER" in CAPTION_PROMPT and "SEARCH box" in CAPTION_PROMPT, "Update annotate.py to prompt v2"
assert PARSER_VERSION >= 3, "Update annotate.py to parser v3"
pilot = Path("runs/captions_500_v2/config-rank00.json")
if pilot.exists():
    config = json.loads(pilot.read_text())
    assert config["prompt_sha256"] == fingerprint(CAPTION_PROMPT), "Pilot and current caption prompts differ"
    assert config["input_sha256"] == sha256(catalog / "manifest.jsonl"), "Pilot and current catalog differ"
print("CUDA and annotation preflight passed", flush=True)
PY

if (( START_STEP <= 0 )); then
# An existing run already has the first pass parameters recorded. Recover
# formatting errors on CPU, then give ONLY missing rows a larger output budget.
if [[ ! -f "$RUN_ROOT/captions/config-rank00.json" ]]; then
  python -m torch.distributed.run --standalone --nproc_per_node="$WORKERS" \
    -m sticker_search.annotate --task caption --catalog "$CATALOG" \
    --output "$RUN_ROOT/captions" --batch-size 8
fi
python -m sticker_search.annotate --task caption --catalog "$CATALOG" \
  --output "$RUN_ROOT/captions" --recover-errors
python -m torch.distributed.run --standalone --nproc_per_node="$WORKERS" \
  -m sticker_search.annotate --task caption --catalog "$CATALOG" \
  --output "$RUN_ROOT/captions" --retry-missing --batch-size 4 --max-new-tokens 1024
python -m sticker_search.annotate --task caption --catalog "$CATALOG" \
  --output "$RUN_ROOT/captions" --recover-errors
# Never invent missing annotations or remove their images. At most 0.5% may
# fall back to the existing image branch; coverage.json lists exact missing IDs.
python -m sticker_search.annotate --task caption --catalog "$CATALOG" \
  --output "$RUN_ROOT/captions" --merge --max-missing-fraction 0.005

# The random 100-image review contained no postcards. Review this slice separately.
python - "$CATALOG" "$RUN_ROOT" <<'PY'
import subprocess
import sys
from pathlib import Path
from sticker_search.common import read_jsonl, write_jsonl
catalog, root = map(Path, sys.argv[1:])
ids = {r["item_id"] for r in read_jsonl(catalog / "manifest.jsonl") if r["kind"] == "postcard"}
rows = [r for r in read_jsonl(root / "captions/annotations.jsonl") if r["item_id"] in ids]
if rows:
    path = root / "postcard_annotations.jsonl"
    write_jsonl(path, rows)
    subprocess.run([sys.executable, "-m", "sticker_search.review_annotations",
                    "--catalog", str(catalog), "--annotations", str(path),
                    "--count", "100", "--output", str(root / "review_postcards.html")], check=True)
PY

fi

if (( START_STEP <= 1 )); then
if [[ ! -f "$RUN_ROOT/captions/annotations.jsonl" ]]; then
  echo "Missing $RUN_ROOT/captions/annotations.jsonl; complete the caption stage first" >&2
  exit 1
fi
# Keep the original 96-token pass/config. Only missing translations use 512 tokens.
if [[ ! -f "$RUN_ROOT/translations/config-rank00.json" ]]; then
  python -m torch.distributed.run --standalone --nproc_per_node="$WORKERS" \
    -m sticker_search.annotate --task translate \
    --queries "$CATALOG/labels/queries_en.jsonl" --output "$RUN_ROOT/translations" \
    --batch-size 16 --max-new-tokens 96
fi
python -m sticker_search.annotate --task translate \
  --queries "$CATALOG/labels/queries_en.jsonl" --output "$RUN_ROOT/translations" --recover-errors
python -m torch.distributed.run --standalone --nproc_per_node="$WORKERS" \
  -m sticker_search.annotate --task translate \
  --queries "$CATALOG/labels/queries_en.jsonl" --output "$RUN_ROOT/translations" \
  --retry-missing --batch-size 4 --max-new-tokens 512
python -m sticker_search.annotate --task translate \
  --queries "$CATALOG/labels/queries_en.jsonl" --output "$RUN_ROOT/translations" --recover-errors
if ! python -m sticker_search.annotate --task translate \
  --queries "$CATALOG/labels/queries_en.jsonl" --output "$RUN_ROOT/translations" --merge; then
  # Save the exact latest responses and original English queries for diagnosis.
  # Never label an English fallback or a guessed answer as a Russian translation.
  python - "$CATALOG" "$RUN_ROOT" <<'PY_REPORT'
import sys
from pathlib import Path
from types import SimpleNamespace
from sticker_search.annotate import load_saved_run
from sticker_search.common import read_jsonl, write_jsonl
catalog, root = map(Path, sys.argv[1:])
output = root / "translations"
args = SimpleNamespace(task="translate", queries=str(catalog / "labels/queries_en.jsonl"), output=str(output))
config, done = load_saved_run(args)
source = {r["query_id"]: r for r in read_jsonl(args.queries)}
latest = {}
for path in sorted(output.glob("errors-rank*.jsonl")):
    for row in read_jsonl(path):
        latest[row["key"]] = row
missing = [key for key in config["selected_keys"] if key not in done]
rows = [dict(key=key, source=source[key], latest_error=latest.get(key)) for key in missing]
target = output / "failures_latest.jsonl"
write_jsonl(target, rows)
print(f"Incomplete translations: {len(missing)}. Exact responses saved to {target}", flush=True)
for row in rows[:3]:
    print(row["key"], repr((row["latest_error"] or {}).get("response", "")[:500]), flush=True)
PY_REPORT
  exit 1
fi
fi

if (( START_STEP <= 2 )); then
# Omit --annotations: only source labels and their translations train adapters.
# Captions and OCR still feed the index, so their value can be measured on dev.
python -m sticker_search.build_queries --catalog "$CATALOG" \
  --translated "$RUN_ROOT/translations/queries_ru.jsonl" --output "$RUN_ROOT/queries_source.jsonl"
python -m sticker_search.features index --catalog "$CATALOG" \
  --annotations "$RUN_ROOT/captions/annotations.jsonl" --output "$RUN_ROOT/index" \
  --device cuda --batch-size 128
python -m sticker_search.features queries --queries "$RUN_ROOT/queries_source.jsonl" \
  --output "$RUN_ROOT/queries" --device cuda --batch-size 128

# Split already-computed vectors by language, retaining order and source hashes.
# Translation pairs are not independent human judgements of Russian relevance.
python - "$RUN_ROOT" <<'PY'
import sys
from pathlib import Path
import numpy as np
from sticker_search.common import read_jsonl, sha256, write_json, write_jsonl
root = Path(sys.argv[1])
rows = read_jsonl(root / "queries/queries.jsonl")
vectors = np.load(root / "queries/queries.npy", allow_pickle=False)
assert len(rows) == len(vectors), "Query vector alignment error"
assert not any(r["label_source"] == "vlm_weak" for r in rows), "Unexpected weak training labels"
for lang in ["en", "ru"]:
    positions = [i for i, row in enumerate(rows) if row["language"] == lang]
    assert positions, f"Missing {lang} queries"
    out = root / f"queries_{lang}"
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "queries.jsonl", [rows[i] for i in positions])
    np.save(out / "queries.npy", vectors[positions])
    write_json(out / "metadata.json", dict(language=lang, parent=str(root / "queries"),
        parent_queries_sha256=sha256(root / "queries/queries.jsonl"),
        parent_vectors_sha256=sha256(root / "queries/queries.npy"), positions=positions))
print("Training query counts:", {lang: sum(r["split"] == "train" and r["language"] == lang for r in rows)
                                 for lang in ["en", "ru"]}, flush=True)
PY

fi

if (( START_STEP <= 3 )); then
python -m sticker_search.train --index "$RUN_ROOT/index" --queries "$RUN_ROOT/queries" \
  --output "$RUN_ROOT/adapter" --rank 64 --lr 0.001 --epochs 10

fi

# Identical candidates/queries for every method. Defer test until choices are fixed.
for QUERY_LANGUAGE in en ru; do
  python -m sticker_search.evaluate --index "$RUN_ROOT/index" --queries "$RUN_ROOT/queries_$QUERY_LANGUAGE" \
    --checkpoint "$RUN_ROOT/adapter/best.pt" --methods image hybrid lexical learned \
    --split dev --output "$RUN_ROOT/dev_$QUERY_LANGUAGE"
done

echo "Done: $RUN_ROOT/adapter/summary.json, dev_en/metrics.json, dev_ru/metrics.json, review_postcards.html"
