"""Dev-only retrieval audit using saved embeddings; no model loading or training.

Compare branch scores, missing-branch weighting and a small declared RRF grid.
RRF ranks each available branch independently before combining its ranks.
All methods use exactly the same full catalog and held-out dev queries.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from .common import read_jsonl, sha256, write_json, write_jsonl
from .metrics import aggregate, query_metrics

BRANCHES = ("image", "caption", "ocr")
RRF_WEIGHTS = {
    "rrf_95_05_00": (.95, .05, 0.),
    "rrf_80_20_00": (.80, .20, 0.),
    "rrf_80_15_05": (.80, .15, .05),
}
METHODS = (*BRANCHES, "hybrid", "linear_fixed", *RRF_WEIGHTS)


def reciprocal_ranks(scores, present, constant=60):
    """Absent features contribute zero; ties retain catalog order."""
    result = np.zeros_like(scores, dtype=np.float32)
    eligible = np.flatnonzero(present)
    if len(eligible):
        order = np.argsort(-scores[:, eligible], axis=1, kind="stable")
        positions = eligible[order]
        result[np.arange(len(scores))[:, None], positions] = 1 / (constant + np.arange(1, len(eligible) + 1))
    return result


def score_methods(branch_scores, present):
    base_weights = np.array([.65, .25, .10], dtype=np.float32)
    item_weights = present * base_weights
    item_weights /= np.maximum(item_weights.sum(1, keepdims=True), 1e-9)
    ranked = [reciprocal_ranks(score, present[:, i]) for i, score in enumerate(branch_scores)]
    for i, name in enumerate(BRANCHES):
        yield name, np.where(present[None, :, i], branch_scores[i], -np.inf)
    yield "hybrid", sum(score * item_weights[None, :, i] for i, score in enumerate(branch_scores))
    yield "linear_fixed", sum(weight * score for weight, score in zip(base_weights, branch_scores))
    for name, weights in RRF_WEIGHTS.items():
        yield name, sum(weight * score for weight, score in zip(weights, ranked))


def top_positions(score, k=10):
    eligible = np.flatnonzero(np.isfinite(score))
    return eligible[np.argsort(-score[eligible], kind="stable")[:k]]


def distribution(values):
    if not values:
        return dict(count=0)
    array = np.asarray(values, dtype=float)
    return dict(count=len(values), mean=float(array.mean()),
                p05=float(np.quantile(array, .05)), p50=float(np.median(array)),
                p95=float(np.quantile(array, .95)))


def describe_item(item):
    return {field: item.get(field) for field in
            ("item_id", "source", "kind", "caption_text", "ocr_text")}


def run(args):
    index, query_dir, output = map(Path, (args.index, args.queries, args.output))
    if output.exists():
        raise ValueError("Diagnostic output exists; choose a new --output")
    items = read_jsonl(index / "items.jsonl")
    ids = [row["item_id"] for row in items]
    meta = json.loads((index / "metadata.json").read_text())
    if ids != meta["item_ids"] or len(set(ids)) != len(ids):
        raise ValueError("Catalog item order or uniqueness mismatch")
    id_to_pos = {key: i for i, key in enumerate(ids)}
    with np.load(index / "vectors.npz", allow_pickle=False) as saved:
        features = [saved[name].astype(np.float32) for name in BRANCHES]
    queries = read_jsonl(query_dir / "queries.jsonl")
    vectors = np.load(query_dir / "queries.npy", allow_pickle=False)
    query_meta = json.loads((query_dir / "metadata.json").read_text())
    if query_meta.get("source_sha256") != sha256(query_dir / "queries.jsonl"):
        raise ValueError("Query metadata/source hash mismatch; use the original combined queries directory")
    if query_meta.get("model") != meta.get("text_model") or query_meta.get("revision") != meta.get("text_revision"):
        raise ValueError("Query and catalog text encoder/revision mismatch")
    if len({q["query_id"] for q in queries}) != len(queries):
        raise ValueError("Duplicate query IDs")
    if vectors.shape != (len(queries), features[0].shape[1]) or not np.isfinite(vectors).all():
        raise ValueError("Invalid query vectors")
    selected = [i for i, q in enumerate(queries) if q["split"] == "dev" and q["label_source"] != "vlm_weak"]
    if not selected:
        raise ValueError("No dev queries")
    for i in selected:
        q = queries[i]
        if not q["positive_item_ids"] or any(p not in id_to_pos for p in q["positive_item_ids"]):
            raise ValueError(f"Missing dev positives: {q['query_id']}")
        if any(items[id_to_pos[p]]["split"] != "dev" for p in q["positive_item_ids"]):
            raise ValueError("Positive split mismatch")
        if "qrels" in q and {k for k, v in q["qrels"].items() if v > 0} != set(q["positive_item_ids"]):
            raise ValueError("qrels / positive IDs mismatch")
    norm_report = {}
    for name, matrix in zip(BRANCHES, features):
        if matrix.shape != (len(ids), vectors.shape[1]) or not np.isfinite(matrix).all():
            raise ValueError(f"Invalid {name} matrix")
        norms = np.linalg.norm(matrix, axis=1)
        present_mask = norms > 0
        if not np.allclose(norms[present_mask], 1, atol=2e-3):
            raise ValueError(f"Unnormalized {name} vectors")
        norm_report[name] = dict(present=int(present_mask.sum()), missing=int((~present_mask).sum()))
    if norm_report["image"]["missing"] or not np.allclose(np.linalg.norm(vectors[selected], axis=1), 1, atol=2e-3):
        raise ValueError("Missing images or unnormalized dev queries")
    present = np.stack([np.linalg.norm(x, axis=1) > 0 for x in features], axis=1)
    records = {name: [] for name in METHODS}
    branch_stats = {}
    examples = {}
    for start in range(0, len(selected), args.batch_size):
        batch = selected[start:start + args.batch_size]
        scores = [vectors[batch] @ matrix.T for matrix in features]
        for local_i, qi in enumerate(batch):
            q = queries[qi]
            example = dict(query_id=q["query_id"], query=q["text"], language=q["language"],
                           original_text=q.get("original_text"),
                           translation_provenance=q.get("translation_provenance"),
                           positives=[describe_item(items[id_to_pos[p]]) for p in q["positive_item_ids"]],
                           branch_scores={}, rankings={})
            for j, name in enumerate(BRANCHES):
                valid = present[:, j]
                positive_positions = [id_to_pos[p] for p in q["positive_item_ids"] if valid[id_to_pos[p]]]
                stats = branch_stats.setdefault(q["language"], {}).setdefault(name,
                    dict(catalog_mean=[], catalog_std=[], positive_score=[], positive_percentile=[]))
                catalog_scores = scores[j][local_i, valid]
                info = dict(positive_available=bool(positive_positions))
                if len(catalog_scores):
                    info.update(catalog_mean=float(catalog_scores.mean()), catalog_std=float(catalog_scores.std()))
                    stats["catalog_mean"].append(info["catalog_mean"])
                    stats["catalog_std"].append(info["catalog_std"])
                if positive_positions:
                    best = float(scores[j][local_i, positive_positions].max())
                    percentile = float((catalog_scores <= best).mean())
                    info.update(positive_score=best, positive_percentile=percentile)
                    stats["positive_score"].append(best)
                    stats["positive_percentile"].append(percentile)
                example["branch_scores"][name] = info
            examples[q["query_id"]] = example
        for method, combined in score_methods(scores, present):
            for local_i, qi in enumerate(batch):
                q = queries[qi]
                top = top_positions(combined[local_i])
                m = query_metrics([ids[p] for p in top], q.get("qrels", {p: 1 for p in q["positive_item_ids"]}))
                records[method].append(dict(query_id=q["query_id"], language=q["language"],
                    group_id=q["group_id"], metrics=m, top10=[ids[p] for p in top]))
                examples[q["query_id"]]["rankings"][method] = dict(metrics=m,
                    top3=[describe_item(items[p]) | dict(score=float(combined[local_i, p])) for p in top[:3]])
        print(f"Audited dev queries: {min(start + args.batch_size, len(selected))}/{len(selected)}", flush=True)

    languages = sorted({queries[i]["language"] for i in selected})
    reports = {}
    for lang in [*languages, "all"]:
        base = [r for r in records["image"] if lang == "all" or r["language"] == lang]
        report = {}
        for method in METHODS:
            rows = [r for r in records[method] if lang == "all" or r["language"] == lang]
            top1 = Counter(r["top10"][0] for r in rows if r["top10"])
            top10 = Counter(key for r in rows for key in r["top10"])
            source_counts = Counter(items[id_to_pos[key]].get("source", "unknown") for r in rows for key in r["top10"])
            deltas = [dict(group_id=r["group_id"], metrics={"NDCG@5_delta": r["metrics"]["NDCG@5"] - b["metrics"]["NDCG@5"]})
                      for r, b in zip(rows, base)]
            report[method] = dict(query_count=len(rows), metrics=aggregate(rows),
                paired_delta_vs_image=aggregate(deltas), unique_top10_items=len(top10),
                top1_source_counts=dict(Counter(items[id_to_pos[key]].get("source", "unknown") for key, n in top1.items() for _ in range(n))),
                top10_source_counts=dict(source_counts),
                frequent_top1=[describe_item(items[id_to_pos[key]]) | dict(count=n, fraction=n/len(rows)) for key, n in top1.most_common(5)])
        reports[lang] = report

    by_qid = {q["query_id"]: i for i, q in enumerate(queries)}
    pairs = []
    for i in selected:
        q = queries[i]
        original_id = q.get("original_query_id")
        if not original_id:
            continue
        j = by_qid[original_id]
        original = queries[j]
        if original["split"] != "dev" or q["original_text"] != original["text"] or q["positive_item_ids"] != original["positive_item_ids"]:
            raise ValueError("EN/RU pair metadata mismatch")
        pairs.append(dict(query_id=q["query_id"], original_text=original["text"], text_ru=q["text"],
                          cosine=float(vectors[i] @ vectors[j])))
    result = dict(split="dev", catalog_size=len(items), query_count=len(selected), norms=norm_report,
        index_sha256=sha256(index / "vectors.npz"), queries_sha256=sha256(query_dir / "queries.jsonl"),
        query_vectors_sha256=sha256(query_dir / "queries.npy"),
        score_distributions={lang: {name: {field: distribution(values) for field, values in stats.items()}
            for name, stats in branches.items()} for lang, branches in branch_stats.items()},
        translation_pair_cosine=distribution([p["cosine"] for p in pairs]),
        lowest_translation_pair_cosine=sorted(pairs, key=lambda p: p["cosine"])[:30],
        methods=reports, rrf_weights=RRF_WEIGHTS, rrf_constant=60,
        protocol="Dev-only diagnostic comparison; full catalog, known-positive labels, unjudged alternatives=0. Paired cluster bootstrap by group_id. CIs are descriptive and not adjusted for choosing among methods. No test evaluation or automatic model replacement.")
    write_json(output / "summary.json", result)
    write_jsonl(output / "cases.jsonl", examples.values())
    table = ["# Диагностика поиска на dev", "", "| Язык | Метод | NDCG@5 | Hit@10 | Уникальных картинок в top-10 |", "|---|---|---:|---:|---:|"]
    for lang, methods in reports.items():
        for name, row in methods.items():
            table.append(f"| {lang} | {name} | {row['metrics']['NDCG@5']['mean']:.6f} | {row['metrics']['Hit@10']['mean']:.6f} | {row['unique_top10_items']} |")
    table += ["", "Методы сравниваются на dev; выбор лучшего по этой таблице не является независимой test-оценкой.",
              "RRF объединяет ранги; отсутствие текста даёт нулевой вклад этой ветки. linear_fixed сохраняет веса 0.65/0.25/0.10 без перенормировки по картинке."]
    (output / "summary.md").write_text("\n".join(table) + "\n", encoding="utf-8")
    print("\n".join(table), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", default="runs/full_v2/index")
    parser.add_argument("--queries", default="runs/full_v2/queries")
    parser.add_argument("--output", default="runs/full_v2/diagnostics_v1")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    run(args)


if __name__ == "__main__":
    main()
