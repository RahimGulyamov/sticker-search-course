"""Exact vector search: 30k x 512 needs no approximate index or GPU service."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .common import read_jsonl, sha256
from .fusion import rank_fusion


class SearchIndex:
    def __init__(self, directory, checkpoint=None, calibration=None):
        self.directory = Path(directory)
        self.items = read_jsonl(self.directory / "items.jsonl")
        self.ids = [r["item_id"] for r in self.items]
        self.id_to_pos = {key: i for i, key in enumerate(self.ids)}
        self.meta = json.loads((self.directory / "metadata.json").read_text())
        if self.meta["item_ids"] != self.ids:
            raise ValueError("Item order disagrees with embedding metadata")
        with np.load(self.directory / "vectors.npz", allow_pickle=False) as data:
            self.image, self.caption, self.ocr = [data[k].astype(np.float32) for k in ["image", "caption", "ocr"]]
        for x in [self.image, self.caption, self.ocr]:
            if x.shape != (len(self.items), 512) or not np.isfinite(x).all():
                raise ValueError("Invalid vector matrix")
        present = np.stack([np.linalg.norm(x, axis=1) > 0 for x in [self.image,self.caption,self.ocr]], 1)
        self.present = present
        weights = present * np.array([.65,.25,.10], dtype=np.float32)
        weights /= np.maximum(weights.sum(1, keepdims=True), 1e-9)
        self.hybrid = self.image*weights[:, :1] + self.caption*weights[:, 1:2] + self.ocr*weights[:, 2:3]
        self.query_mean = None
        if calibration:
            calibration = Path(calibration)
            config = json.loads((calibration / "metadata.json").read_text())
            if config["index_sha256"] != sha256(self.directory / "vectors.npz"):
                raise ValueError("Calibration belongs to a different index")
            self.query_mean = np.load(calibration / "query_mean.npy", allow_pickle=False)
            if self.query_mean.shape != (512,) or not np.isfinite(self.query_mean).all():
                raise ValueError("Invalid query calibration mean")
            if config["text_model"] != self.meta["text_model"] or config["text_revision"] != self.meta["text_revision"]:
                raise ValueError("Calibration text encoder mismatch")
            self.branch_bias = [self.query_mean @ matrix.T for matrix in [self.image, self.caption, self.ocr]]
        self.vectorizer = None
        texts = [r.get("ocr_text", "") for r in self.items]
        if any(t.strip() for t in texts):
            self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2,5), max_features=150000, dtype=np.float32)
            self.lexical = self.vectorizer.fit_transform(texts)
        self.model = None
        if checkpoint:
            import torch
            from .model import load_model
            self.model, config = load_model(checkpoint)
            if config["index_sha256"] != sha256(self.directory / "vectors.npz"):
                raise ValueError("Checkpoint was trained against a different index")
            with torch.inference_mode():
                self.learned = self.model.encode_items(*[torch.from_numpy(x) for x in [self.image,self.caption,self.ocr]]).numpy()

    def scores(self, query_vectors, method="image", query_texts=None):
        if method in {"rrf", "rrf_calibrated", "rrf_calibrated_ocr", "rrf_calibrated_all", "image_calibrated", "caption_calibrated"}:
            branches = [query_vectors @ matrix.T for matrix in [self.image, self.caption, self.ocr]]
            if method != "rrf":
                if self.query_mean is None:
                    raise ValueError("Provide TRAIN-fitted --calibration")
                # s'(q,item) = s(q,item) - E_train[s(query,item)].
                # Keep the correction unnormalized: it is a score offset,
                # not a claim that the centered vector is a cosine embedding.
                for i in [1, 2]:
                    branches[i] = branches[i] - self.branch_bias[i]
                if method in {"rrf_calibrated_all", "image_calibrated"}:
                    branches[0] = branches[0] - self.branch_bias[0]
            if method == "image_calibrated":
                return branches[0]
            if method == "caption_calibrated":
                return np.where(self.present[None, :, 1], branches[1], -np.inf)
            weights = (.8, .15, .05) if method == "rrf_calibrated_ocr" else (.8, .2, 0.)
            return rank_fusion(branches, self.present, weights)
        if method == "lexical":
            if query_texts is None:
                raise ValueError("Lexical search needs query text")
            if self.vectorizer is None:
                return np.zeros((len(query_texts), len(self.items)), dtype=np.float32)
            return (self.vectorizer.transform(query_texts) @ self.lexical.T).toarray()
        if method == "learned":
            if self.model is None:
                raise ValueError("Provide a trained checkpoint")
            import torch
            with torch.inference_mode():
                q = self.model.encode_query(torch.from_numpy(query_vectors.astype(np.float32))).numpy()
            return q @ self.learned.T
        matrices = {"image":self.image, "caption":self.caption, "ocr":self.ocr, "hybrid":self.hybrid}
        return query_vectors @ matrices[method].T

    def topk(self, scores, k=10, kind=None):
        eligible = np.array([i for i,r in enumerate(self.items) if np.isfinite(scores[i]) and (kind is None or r["kind"] == kind)], dtype=int)
        if not len(eligible):
            return []
        k = min(k, len(eligible))
        # Stable tie handling for deterministic metrics; full sorting is cheap at 30k.
        order = eligible[np.argsort(-scores[eligible], kind="stable")[:k]]
        return [(self.items[i],float(scores[i])) for i in order]
