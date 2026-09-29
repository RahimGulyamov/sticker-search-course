"""Ranking metrics with explicit known-positive / graded-relevance semantics."""
import numpy as np


def query_metrics(ranked_ids, qrels, ks=(1, 5, 10)):
    relevant = {key for key, grade in qrels.items() if grade > 0}
    if not relevant:
        raise ValueError("No positive judgments; evaluate no-match queries separately")
    grades = np.array([qrels.get(key, 0) for key in ranked_ids], dtype=float)
    ideal = sorted([g for g in qrels.values() if g > 0], reverse=True)
    result = {}
    positive_positions = np.flatnonzero(grades > 0)
    result["MRR@10"] = float(1/(positive_positions[0]+1)) if len(positive_positions) and positive_positions[0] < 10 else 0.0
    for k in ks:
        top = grades[:k]
        result[f"Hit@{k}"] = float((top > 0).any())
        result[f"Recall@{k}"] = float((top > 0).sum()/len(relevant))
        discount = 1 / np.log2(np.arange(len(top)) + 2)
        dcg = ((2**top - 1) * discount).sum()
        best = np.array(ideal[:k], dtype=float)
        idcg = ((2**best-1) / np.log2(np.arange(len(best))+2)).sum()
        result[f"NDCG@{k}"] = float(dcg/idcg)
    return result


def aggregate(records, seed=42, bootstrap=1000):
    """Cluster bootstrap keeps all queries/translations for one image group together."""
    if not records:
        raise ValueError("No evaluated queries")
    names = list(records[0]["metrics"])
    groups = sorted({r["group_id"] for r in records})
    by_group = {g: [r for r in records if r["group_id"] == g] for g in groups}
    group_sum = np.array([[sum(r["metrics"][name] for r in by_group[g]) for name in names] for g in groups])
    group_n = np.array([len(by_group[g]) for g in groups])
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(bootstrap):
        sampled = rng.integers(0, len(groups), size=len(groups))
        boot.append(group_sum[sampled].sum(0)/group_n[sampled].sum())
    ci = np.quantile(boot, [0.025, 0.975], axis=0)
    means = group_sum.sum(0)/group_n.sum()
    return {name: dict(mean=float(means[i]), ci95=[float(ci[0, i]), float(ci[1, i])]) for i, name in enumerate(names)}
