"""Self-contained visual examples for fixed Russian user scenarios, without gold labels."""
import argparse
import base64
import html
import io
from pathlib import Path
import time

from .common import rgb_image, write_json, write_jsonl

PROBES = [
    ("привет, котик машет лапой", "sticker"),
    ("спасибо за помощь", "sticker"),
    ("я очень устал, хочу спать", "sticker"),
    ("смеюсь до слёз", "sticker"),
    ("обнимаю тебя", "sticker"),
    ("я злюсь", "sticker"),
    ("ничего не понимаю", "sticker"),
    ("прости меня", "sticker"),
    ("поддержать друга перед экзаменом", "sticker"),
    ("кот с чашкой кофе", "sticker"),
    ("ура, у меня получилось", "sticker"),
    ("нет, я не согласен", "sticker"),
    ("с днём рождения", "postcard"),
    ("поздравить с Новым годом", "postcard"),
    ("пожелать доброго утра", "postcard"),
    ("пожелать удачи", "postcard"),
    ("грустный кот", None),
    ("робот в космосе играет на скрипке", None),
]


def build(catalog, index_path, checkpoint, output):
    from .features import text_encoder
    from .retrieval import SearchIndex
    import torch
    torch.set_num_threads(8)
    catalog, output = Path(catalog), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    index = SearchIndex(index_path, checkpoint)
    encoder = text_encoder("cpu")
    encoder.encode(["проверка"], normalize_embeddings=True)  # Exclude cold start from per-query timings.
    cache, sections, records = {}, [], []
    labels = {"image": "По изображению", "rrf": "RRF: изображение + описание", "learned": "Обученный адаптер"}
    for query, kind in PROBES:
        start = time.perf_counter()
        vector = encoder.encode([query], normalize_embeddings=True)
        encode_ms = (time.perf_counter() - start) * 1000
        groups = []
        for method, label in labels.items():
            start = time.perf_counter()
            scores = index.scores(vector, method, [query])[0]
            top = index.topk(scores, 3, kind=kind)
            search_ms = (time.perf_counter() - start) * 1000
            cards = []
            for item, score in top:
                key = item["item_id"]
                if key not in cache:
                    image = rgb_image(catalog / item["image_path"])
                    image.thumbnail((220, 220))
                    stream = io.BytesIO()
                    image.save(stream, format="PNG")
                    cache[key] = "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode()
                cards.append(f'<article><img src="{cache[key]}" alt="{html.escape(key)}"><small>{html.escape(key)}</small>'
                             f'<p>{html.escape(item.get("caption_text", ""))}</p>'
                             f'<p>Надпись: {html.escape(item.get("ocr_text", ""))}</p></article>')
            groups.append(f'<div class="method"><h3>{html.escape(label)}</h3><div class="cards">{"".join(cards)}</div></div>')
            records.append(dict(query=query, kind=kind, method=method, encoder_ms=encode_ms, search_ms=search_ms,
                                warm_end_to_end_ms=encode_ms + search_ms,
                                top3=[dict(item_id=item["item_id"], score=score) for item, score in top]))
        sections.append(f'<section><h2>{html.escape(query)}</h2><p>Тип: {kind or "всё"}</p>{"".join(groups)}</section>')
        print(f"Visual review: {query}", flush=True)
    page = '''<!doctype html><html lang="ru"><meta charset="utf-8"><title>Примеры русского поиска</title>
<style>body{font:15px system-ui;background:#f3f5fa;color:#1a2438;max-width:1250px;margin:28px auto;padding:0 20px}
section{background:white;padding:22px;margin:24px 0;border-radius:14px}.cards{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:18px}
article{border:1px solid #e0e5ee;padding:12px;border-radius:10px}img{width:100%;height:180px;object-fit:contain}small{overflow-wrap:anywhere}p{overflow-wrap:anywhere}
h3{color:#354da6}.method{margin:20px 0}@media(max-width:700px){.cards{grid-template-columns:1fr}}</style>
<h1>Русские запросы: сравнение выдачи</h1><p>Фиксированные пользовательские сценарии, top-3, одинаковый фильтр типа для всех методов.
Это качественные примеры без заданных релевантных ответов, а не независимая человеческая оценка. Описания/OCR получены VLM и могут ошибаться.
Проверять содержание самих картинок, соответствие эмоции, надписи и запросу. Сложный последний запрос также проверяет сценарий генерации при неудовлетворительном поиске.</p>'''
    (output / "search_review.html").write_text(page + "".join(sections) + "</html>", encoding="utf-8")
    write_jsonl(output / "search_examples.jsonl", records)
    write_json(output / "review_protocol.json", dict(probes=PROBES, methods=list(labels), top_k=3,
        gold_labels=False, human_reviewed=False, cold_start_excluded=True,
        timing_excludes="image loading, HTML rendering and network", device="cpu", threads=8))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--catalog", default="data/catalog")
    p.add_argument("--index", default="runs/full_v2/index")
    p.add_argument("--checkpoint", default="runs/full_v2/adapter/best.pt")
    p.add_argument("--output", default="runs/final_v1/review")
    args = p.parse_args()
    build(args.catalog, args.index, args.checkpoint, args.output)


if __name__ == "__main__":
    main()
