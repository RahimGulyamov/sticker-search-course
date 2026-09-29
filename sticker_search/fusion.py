"""Rank fusion and a query-distribution correction fitted on TRAIN only."""
import numpy as np


def rank_fusion(branch_scores, present, weights=(.8, .2, 0.), constant=60):
    result = np.zeros_like(branch_scores[0], dtype=np.float32)
    for i, (scores, weight) in enumerate(zip(branch_scores, weights)):
        if not weight:
            continue
        eligible = np.flatnonzero(present[:, i])
        if not len(eligible):
            continue
        order = np.argsort(-scores[:, eligible], axis=1, kind="stable")
        positions = eligible[order]
        values = (1 / (constant + np.arange(1, len(eligible) + 1))).astype(np.float32)
        result[np.arange(len(scores))[:, None], positions] += np.float32(weight) * values
    return result


def fit_query_mean(queries, vectors):
    """No dev/test labels or vectors contribute to the fitted mean."""
    selected = [i for i, q in enumerate(queries) if q["split"] == "train" and q["label_source"] != "vlm_weak"]
    if not selected:
        raise ValueError("No source train queries for calibration")
    if vectors.ndim != 2 or len(vectors) != len(queries) or not np.isfinite(vectors[selected]).all():
        raise ValueError("Invalid calibration vectors")
    # Float64 accumulation avoids making this sensitive to summation order.
    return vectors[selected].mean(axis=0, dtype=np.float64).astype(np.float32), selected
