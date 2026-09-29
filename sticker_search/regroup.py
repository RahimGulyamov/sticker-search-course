"""Apply the current grouping rule without redownloading pixels; invalidate old indices."""
import argparse
import json
from collections import Counter
from pathlib import Path
from .common import read_jsonl,write_json,write_jsonl
from .prepare import group_and_split


def main():
    p=argparse.ArgumentParser();p.add_argument('--catalog',default='data/catalog');args=p.parse_args()
    root=Path(args.catalog);rows=read_jsonl(root/'manifest.jsonl')
    # Early pilot's source_family for SQ was an item identifier, not a pack ID.
    for r in rows:
        if r['source']=='StickerQueries':r['source_family']=None
    near=group_and_split(rows,42,image_root=root);items={r['item_id']:r for r in rows}
    labels=read_jsonl(root/'labels/queries_en.jsonl')
    for q in labels:
        item=items[q['positive_item_ids'][0]];q.update(group_id=item['group_id'],split=item['split'])
    unique={}
    for q in labels:
        unique.setdefault((q['text'].strip().lower(),q['language'],tuple(q['positive_item_ids'])),q)
    labels=list(unique.values())
    write_jsonl(root/'manifest.jsonl',rows);write_jsonl(root/'labels/queries_en.jsonl',labels)
    write_jsonl(root/'audit/near_duplicates.jsonl',near)
    write_json(root/'audit/grouping.json',dict(rule_version=3,seed=42,near_edges=len(near),
        groups=len({r['group_id'] for r in rows}),largest=Counter(r['group_id'] for r in rows).most_common(10),
        splits=dict(Counter(r['split'] for r in rows)),
        note='Known different artwork families kept separate. dHash candidates verified using foreground-cropped 64px thumbnails: mask IoU>=0.85 and RGB MAE<=0.03. Rebuild dependent indices.'))
    summary_path=root/'audit/summary.json'
    if summary_path.exists():
        summary=json.loads(summary_path.read_text())
        summary.update(near_edges=len(near),group_count=len({r['group_id'] for r in rows}),
            largest_groups=Counter(r['group_id'] for r in rows).most_common(10),
            splits=dict(Counter(r['split'] for r in rows)),source_query_count=len(labels),grouping_rule_version=3)
        write_json(summary_path,summary)
    print('Grouping updated; rebuild dependent indices and translated queries.')


if __name__=='__main__':main()
