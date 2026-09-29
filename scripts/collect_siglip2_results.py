"""Small result bundle plus a transparent before/after DEV contact sheet."""
import argparse
import base64
import html
import io
import json
import sys
import zipfile
from pathlib import Path

from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sticker_search.common import read_jsonl, rgb_image


def build_review(run,catalog):
    selection = json.loads((run/"selection.json").read_text())
    baseline = read_jsonl(run/"evaluations/epoch_000/cases.jsonl")
    selected = read_jsonl(run/f"evaluations/epoch_{selection['epoch']:03d}/cases.jsonl")
    before = {r["query_id"]:r for r in baseline}
    items = {r["item_id"]:r for r in read_jsonl(run/"baseline/items.jsonl")}
    chosen = []
    for lang in ("en","ru"):
        rows = [r for r in selected if r["language"]==lang]
        rows.sort(key=lambda r:(r["metrics"]["NDCG@5"]-before[r["query_id"]]["metrics"]["NDCG@5"],r["query_id"]))
        seen = set()
        for row in rows[:4]+rows[-4:]:
            if row["query_id"] not in seen:
                chosen.append(row);seen.add(row["query_id"])
    images = {}
    def thumb(pid):
        if pid not in images:
            image = rgb_image(catalog/items[pid]["image_path"])
            image.thumbnail((180,180))
            stream = io.BytesIO();image.save(stream,format="JPEG",quality=80)
            images[pid] = base64.b64encode(stream.getvalue()).decode()
        return images[pid]
    parts = ['<!doctype html><meta charset="utf-8"><title>SigLIP 2: dev review</title>',
        '<style>body{font:16px system-ui;max-width:1100px;margin:30px auto}article{border-top:1px solid #aaa;padding:15px 0}.row{display:flex;gap:15px}.item{width:180px;font-size:12px;overflow-wrap:anywhere}img{height:160px;max-width:180px;object-fit:contain}.positive{outline:3px solid green}h2{margin-bottom:8px}</style>',
        '<h1>SigLIP 2: исходная и выбранная модель</h1>',
        '<p>DEV: по 4 наибольших снижения и роста NDCG@5 на язык. Это целевой разбор крайних случаев, а не случайная выборка. Зелёная рамка — известная положительная картинка; отсутствие рамки не доказывает нерелевантность. RU — переводы.</p>']
    for row in chosen:
        old = before[row["query_id"]]
        parts.append(f'<article><h2>{html.escape(row["text"])}</h2><p>{html.escape(row["query_id"])} · NDCG@5: {old["metrics"]["NDCG@5"]:.4f} → {row["metrics"]["NDCG@5"]:.4f}</p>')
        for label,case in [("Исходная",old),(f"Выбранная, эпоха {selection['epoch']}",row)]:
            parts.append(f'<h3>{label}</h3><div class="row">')
            for pid in case["ranked_ids"][:5]:
                klass="positive" if pid in case["positive_item_ids"] else ""
                parts.append(f'<div class="item"><img class="{klass}" src="data:image/jpeg;base64,{thumb(pid)}"><p>{html.escape(pid)}</p><p>{html.escape(items[pid].get("caption_text",""))}</p></div>')
            parts.append('</div>')
        parts.append('</article>')
    path=run/"review_dev.html"
    path.write_text("\n".join(parts),encoding="utf-8")
    return path


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--run",default="/tmp/cv_project/siglip2_v1")
    p.add_argument("--catalog",default="data/catalog")
    p.add_argument("--output",default="runs/siglip2_v1_results.zip")
    a=p.parse_args()
    run=Path(a.run).resolve()
    if not (run/"complete.json").is_file():
        raise ValueError("Wait for the full training run to complete")
    build_review(run,Path(a.catalog))
    files = [run/name for name in ("config.json","training_data.json","history.jsonl","selection.json",
        "complete.json","review_dev.html","queries.jsonl","prepared/coverage.json","prepared/provenance.json",
        "prepared/rejected_captions.jsonl","smoke/complete.json","baseline/metadata.json")]
    for pattern in ("evaluations/*/*.json","evaluations/*/*.jsonl","baseline_dev/*.json*",
                    "selected_dev*/*.json*","baseline_test/*.json*","selected_test*/*.json*"):
        files.extend(run.glob(pattern))
    target=Path(a.output)
    target.parent.mkdir(parents=True,exist_ok=True)
    temporary=target.with_suffix(".tmp.zip")
    with zipfile.ZipFile(temporary,"w",compression=zipfile.ZIP_DEFLATED) as archive:
        for f in sorted(set(files)):
            if f.is_file():
                archive.write(f,arcname=f.relative_to(run).as_posix())
    temporary.replace(target)
    print(f"Saved {target}: {target.stat().st_size/1024**2:.1f} MiB; no model weights included")


if __name__=="__main__":
    main()
