"""Audit and create broad TRAIN-only supervision; no torch required."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from ..common import fingerprint, read_jsonl, rgb_image, sha256, write_json, write_jsonl


def normalize_text(text):
    return " ".join(text.lower().split())


def text_problem(value):
    if not isinstance(value, str) or len(value.strip()) < 3:
        return "empty_or_too_short"
    if len(value) > 1000 or re.search(r"(.)\1{11,}", value):
        return "long_or_repetitive"
    words = normalize_text(value).split()
    if len(words) > 16 and len(set(words)) / len(words) < .25:
        return "repetitive_words"
    if any(s in normalize_text(value) for s in
           ("cannot describe", "can't describe", "не могу описать", "as an ai", "как языковая модель")):
        return "refusal"
    return None


def validate_catalog(items, queries):
    by_id = {r["item_id"]: r for r in items}
    if len(by_id) != len(items):
        raise ValueError("Duplicate catalog item_id")
    groups = defaultdict(set)
    for row in items:
        if row["split"] not in {"train", "dev", "test"}:
            raise ValueError("Unknown split")
        groups[row["group_id"]].add(row["split"])
    if any(len(s) > 1 for s in groups.values()):
        raise ValueError("Image group crosses train/dev/test")
    if len({q["query_id"] for q in queries}) != len(queries):
        raise ValueError("Duplicate query_id")
    for q in queries:
        if not q["positive_item_ids"] or not q["text"].strip():
            raise ValueError(f"Empty query/positives: {q['query_id']}")
        if any(p not in by_id or by_id[p]["split"] != q["split"] for p in q["positive_item_ids"]):
            raise ValueError(f"Cross-split or missing positive: {q['query_id']}")
        if "qrels" in q and {k for k,v in q["qrels"].items() if v > 0} != set(q["positive_item_ids"]):
            raise ValueError("qrels disagree with positive_item_ids")
    return by_id


def prepare(catalog, annotations, translated, output, config):
    catalog, output = Path(catalog), Path(output)
    items = sorted(read_jsonl(catalog / "manifest.jsonl"), key=lambda r:r["item_id"])
    queries = read_jsonl(catalog / "labels/queries_en.jsonl") + read_jsonl(translated)
    by_id = validate_catalog(items, queries)
    annotation_rows = read_jsonl(annotations)
    ann = {r["item_id"]:r for r in annotation_rows}
    if len(ann) != len(annotation_rows) or not set(ann) <= set(by_id):
        raise ValueError("Duplicate or unknown annotation ID")
    provenance = dict(manifest_sha256=sha256(catalog / "manifest.jsonl"),
                      annotations_sha256=sha256(annotations),
                      queries_en_sha256=sha256(catalog / "labels/queries_en.jsonl"),
                      translations_sha256=sha256(translated), preparation_version=1,
                      config={k:config[k] for k in ("min_train_coverage", "caption_weight",
                              "source_query_weight", "translated_query_weight")})
    if (output / "provenance.json").exists():
        if json.loads((output / "provenance.json").read_text()) != provenance:
            raise ValueError("Prepared inputs changed; use a new output directory")
    pairs, rejected = [], []
    for item in items:
        if item["split"] != "train":
            continue
        a = ann.get(item["item_id"], {})
        for lang in ("en", "ru"):
            field = f"caption_{lang}"
            text = a.get(field, "")
            reason = text_problem(text)
            if reason:
                rejected.append(dict(item_id=item["item_id"], field=field, reason=reason, text=text))
                continue
            # Keep factual visual captions. Do not reuse hallucinated usage_queries_ru.
            pairs.append(dict(pair_id=f"caption:{lang}:{item['item_id']}",
                item_id=item["item_id"], text=text.strip(), language=lang,
                positive_item_ids=[item["item_id"]], origin="vlm_caption",
                weight=config["caption_weight"]))
    for q in queries:
        if q["split"] != "train":
            continue
        if q["label_source"] not in {"human_source", "human_source_machine_translation"}:
            raise ValueError(f"Unexpected query supervision: {q['label_source']}")
        translated_query = q["label_source"] == "human_source_machine_translation"
        for pid in q["positive_item_ids"]:
            pairs.append(dict(pair_id=f"query:{q['query_id']}:{pid}", item_id=pid,
                text=q["text"], language=q["language"], positive_item_ids=q["positive_item_ids"],
                origin=q["label_source"], weight=config["translated_query_weight" if translated_query
                                                    else "source_query_weight"]))
    covered = {p["item_id"] for p in pairs}
    train_ids = {i["item_id"] for i in items if i["split"] == "train"}
    missing = sorted(train_ids - covered)
    coverage = len(covered) / len(train_ids) if train_ids else 0
    # Decode every candidate once before expensive model loading, including dev/test.
    for item in items:
        path = catalog / item["image_path"]
        if not path.resolve().is_relative_to(catalog.resolve()):
            raise ValueError("Image path escapes catalog")
        try:
            rgb_image(path)
        except Exception as e:
            raise ValueError(f"Cannot decode {item['item_id']}: {path}") from e
    summary = dict(catalog_images=len(items), splits=dict(Counter(i["split"] for i in items)),
        train_unique_images=len(covered), train_coverage=coverage, missing_train_item_ids=missing,
        train_pairs=len(pairs), pairs_by_origin=dict(Counter(p["origin"] for p in pairs)),
        pairs_by_language=dict(Counter(p["language"] for p in pairs)),
        rejected_caption_fields=len(rejected), all_images_decoded=True,
        note="Automatic syntax checks do not establish semantic correctness of VLM captions or translations.",
        dev_test_used_for_training=False)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "coverage.json", summary)
    write_jsonl(output / "rejected_captions.jsonl", rejected)
    print(json.dumps(summary | {"missing_train_item_ids":len(missing)}, ensure_ascii=False, indent=2))
    if coverage < config["min_train_coverage"]:
        raise ValueError("Insufficient train coverage; inspect rejected_captions.jsonl before training")
    write_jsonl(output / "pairs.jsonl", pairs)
    write_jsonl(output / "queries.jsonl", queries)
    write_jsonl(output / "items.jsonl", [i | {"caption_text":ann.get(i["item_id"],{}).get("caption_ru", ""),
        "ocr_text":ann.get(i["item_id"],{}).get("ocr_text", "")} for i in items])
    write_json(output / "provenance.json", provenance)


def positive_map(pairs, tokenizer=None, max_length=64):
    """Merge known positives for identical encoder inputs, including truncation collisions."""
    keys = [normalize_text(p["text"]) for p in pairs]
    if tokenizer is not None:
        ids = tokenizer(keys, padding="max_length", truncation=True,
                        max_length=max_length)["input_ids"]
        keys = [fingerprint(row) for row in ids]
    positives = defaultdict(set)
    for key, pair in zip(keys, pairs):
        positives[key].update(pair["positive_item_ids"])
    return [positives[key] for key in keys]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--catalog", default="data/catalog")
    p.add_argument("--annotations", default="runs/full_v2/captions/annotations.jsonl")
    p.add_argument("--translated", default="runs/full_v2/translations/queries_ru.jsonl")
    p.add_argument("--output", default="runs/siglip2_v1/prepared")
    p.add_argument("--config", default="configs/siglip2_full.json")
    a = p.parse_args()
    prepare(a.catalog, a.annotations, a.translated, a.output, json.loads(Path(a.config).read_text()))


if __name__ == "__main__":
    main()
