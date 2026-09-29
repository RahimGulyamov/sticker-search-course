"""Interactive exact vector search and an optional explicit OCR branch."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ..common import read_jsonl, sha256
from .io import encode_texts, load_local, model_identity


class OcrFusion:
    """Fixed RRF k=60 over image ranks and nonzero lexical OCR matches.

    This is a separate ablation; it is NOT used to select encoder checkpoints.
    """
    def __init__(self,items):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer(analyzer="char_wb",ngram_range=(3,5),lowercase=True)
        texts = [r.get("ocr_text","") for r in items]
        try:
            self.matrix = self.vectorizer.fit_transform(texts)
        except ValueError as e:
            if "empty vocabulary" not in str(e):
                raise
            self.matrix = None

    def scores(self,query,image_scores):
        if self.matrix is None:
            return image_scores
        lexical = (self.matrix @ self.vectorizer.transform([query]).T).toarray().ravel()
        if not np.any(lexical > 0):
            return image_scores
        order = np.argsort(-image_scores,kind="stable")
        score = np.zeros_like(image_scores)
        score[order] = 1/(60+np.arange(1,len(order)+1))
        lex_order = np.argsort(-lexical,kind="stable")
        lex_order = lex_order[lexical[lex_order] > 0]
        score[lex_order] += 1/(60+np.arange(1,len(lex_order)+1))
        return score


class SiglipSearch:
    def __init__(self,run,catalog,device="cuda",which="selected"):
        run = Path(run)
        self.catalog = Path(catalog)
        index = run/"baseline" if which == "baseline" else Path(json.loads((run/"selection.json").read_text())["index_path"])
        self.meta = json.loads((index/"metadata.json").read_text())
        if sha256(self.catalog/"manifest.jsonl") != self.meta["manifest_sha256"]:
            raise ValueError("Catalog differs from the index")
        if model_identity(self.meta["model_path"]) != self.meta["model_sha256"]:
            raise ValueError("Model differs from indexed weights")
        self.items = read_jsonl(index/"items.jsonl")
        vectors = np.load(index/"images.npy",allow_pickle=False)
        if vectors.shape != (len(self.items),self.meta["dimension"]):
            raise ValueError("Vector shape mismatch")
        if [r["item_id"] for r in self.items] != self.meta["item_ids"]:
            raise ValueError("Item order mismatch")
        self.device = device
        self.vectors = torch.from_numpy(vectors).to(device)
        self.model,self.processor = load_local(self.meta["model_path"],device)
        self.ocr = None

    @torch.inference_mode()
    def search(self,query,k=12,kind=None,method="image"):
        if not query.strip():
            return []
        text = encode_texts(self.model,self.processor,[query],self.device,1,
            self.meta["precision"],self.meta["max_text_length"])
        score = (torch.from_numpy(text).to(self.device) @ self.vectors.T)[0].cpu().numpy()
        if method == "image_ocr_rrf":
            if self.ocr is None:
                self.ocr = OcrFusion(self.items)
            score = self.ocr.scores(query,score)
        elif method != "image":
            raise ValueError("Unknown search method")
        order = np.argsort(-score,kind="stable")
        if kind:
            order = [i for i in order if self.items[i]["kind"] == kind]
        return [(self.items[int(i)],float(score[i])) for i in order[:k]]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run",default="/tmp/cv_project/siglip2_v1")
    p.add_argument("--catalog",default="data/catalog")
    p.add_argument("--query",required=True)
    p.add_argument("--device",default="cuda")
    p.add_argument("--which",choices=["baseline","selected"],default="selected")
    p.add_argument("--method",choices=["image","image_ocr_rrf"],default="image")
    a = p.parse_args()
    index = SiglipSearch(a.run,a.catalog,a.device,a.which)
    for row,score in index.search(a.query,method=a.method):
        print(json.dumps(dict(item_id=row["item_id"],score=score,image_path=row["image_path"],
            caption=row.get("caption_text",""),ocr=row.get("ocr_text","")),ensure_ascii=False))


if __name__ == "__main__":
    main()
