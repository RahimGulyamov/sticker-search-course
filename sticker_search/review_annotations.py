"""Build a self-contained HTML sample for reviewing generated descriptions/OCR."""
import argparse
import base64
import html
import json
import random
from pathlib import Path

from .common import read_jsonl


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--catalog',default='data/catalog')
    p.add_argument('--annotations',required=True)
    p.add_argument('--output',default='runs/annotation_review.html')
    p.add_argument('--count',type=int,default=100)
    args=p.parse_args()
    root=Path(args.catalog);items={r['item_id']:r for r in read_jsonl(root/'manifest.jsonl')}
    annotations=read_jsonl(args.annotations)
    selected=random.Random(42).sample(annotations,min(args.count,len(annotations)))
    cards=[]
    from .common import rgb_image
    import io
    for a in selected:
        row=items[a['item_id']]
        image=rgb_image(root/row['image_path']);image.thumbnail((256,256))
        stream=io.BytesIO();image.save(stream,format='PNG')
        uri='data:image/png;base64,'+base64.b64encode(stream.getvalue()).decode()
        fields={k:a.get(k) for k in ['caption_en','caption_ru','emotion_ru','style_ru','ocr_text','ocr_language','usage_queries_ru']}
        cards.append(f'<article><img src="{uri}"><div><h3>{html.escape(a["item_id"])}</h3><pre>{html.escape(json.dumps(fields,ensure_ascii=False,indent=2))}</pre></div></article>')
    page='''<!doctype html><html lang="ru"><meta charset="utf-8"><title>Проверка разметки</title>
    <style>body{font:16px system-ui;max-width:1100px;margin:32px auto;background:#f6f7fb;color:#192237}article{display:flex;gap:24px;padding:24px;margin:18px 0;background:white;border-radius:14px}img{width:220px;object-fit:contain;align-self:start}pre{white-space:pre-wrap;font:14px system-ui}h1{font-size:30px}</style>
    <h1>Проверка автоматической разметки</h1><p>Это выход модели, а не подтверждённые человеком факты. Проверить предметы, эмоцию, надпись, перевод и ситуации использования. Зафиксировать число просмотренных изображений и ошибок каждого типа.</p>'''+''.join(cards)+'</html>'
    Path(args.output).parent.mkdir(parents=True,exist_ok=True);Path(args.output).write_text(page,encoding='utf8')
    print(f'Saved {len(cards)} examples to {args.output}')


if __name__=='__main__':main()
