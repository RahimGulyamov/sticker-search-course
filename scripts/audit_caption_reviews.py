"""Compare the same embedded image sample; counts are proxies, not relevance scores.

python scripts/audit_caption_reviews.py --v1 review_500.html --v2 review_500_v2.html
"""
import argparse
import base64
import hashlib
import html
import json
import re
from collections import Counter
from pathlib import Path


QUESTION_PREFIX = r"^(как|что|кто|где|почему|како[йея]|какие)\b"


def extract(path):
    rows = []
    for article in re.findall(r"<article>(.*?)</article>", path.read_text(), re.S):
        key = html.unescape(re.search(r"<h3>(.*?)</h3>", article, re.S).group(1))
        fields = json.loads(html.unescape(re.search(r"<pre>(.*?)</pre>", article, re.S).group(1)))
        image_bytes = base64.b64decode(re.search(r'base64,([^\"]+)', article).group(1))
        rows.append(dict(item_id=key, image_sha256=hashlib.sha256(image_bytes).hexdigest(), **fields))
    if not rows or len({r['item_id'] for r in rows}) != len(rows):
        raise ValueError('Expected nonempty review with unique IDs')
    return rows


def stats(rows):
    queries = [q for r in rows for q in r['usage_queries_ru']]
    return dict(images=len(rows), queries=len(queries),
                question_mark_queries=sum('?' in q for q in queries),
                question_prefix_queries=sum(bool(re.match(QUESTION_PREFIX, q, re.I)) for q in queries),
                items_with_question_prefix=sum(any(re.match(QUESTION_PREFIX, q, re.I) for q in r['usage_queries_ru']) for r in rows),
                queries_without_cyrillic=sum(not re.search('[А-Яа-яЁё]', q) for q in queries),
                nonempty_ocr_items=sum(bool(r['ocr_text'].strip()) for r in rows),
                source_prefixes=dict(Counter(r['item_id'].split('_')[0] for r in rows)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--v1', type=Path, required=True)
    p.add_argument('--v2', type=Path, required=True)
    p.add_argument('--output', type=Path, default=Path('evidence/caption_review'))
    args = p.parse_args()
    versions = [extract(args.v1), extract(args.v2)]
    same = [(r['item_id'], r['image_sha256']) for r in versions[0]] == [(r['item_id'], r['image_sha256']) for r in versions[1]]
    if not same:
        raise ValueError('Reviews differ in image IDs/order/bytes; not a paired comparison')
    result = dict(same_image_sample_and_order=same, question_prefix_regex=QUESTION_PREFIX,
                  v1=stats(versions[0]), v2=stats(versions[1]),
                  files=[dict(name=p.name, sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in [args.v1,args.v2]],
                  caveat='Mechanical counts on 100 displayed examples, not 500 annotations or measured retrieval quality; no human OCR ground truth.')
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    (args.output/'paired_records.json').write_text(json.dumps(versions, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
