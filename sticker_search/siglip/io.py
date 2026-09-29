"""Offline model loading, deterministic inputs and checkpoint identity."""
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset
from transformers import AutoModel, AutoProcessor

from ..common import fingerprint, rgb_image, sha256
from .data import normalize_text


def model_identity(path):
    path = Path(path)
    names = sorted(p for p in path.glob("*") if p.is_file() and
                   (p.suffix == ".safetensors" or p.suffix == ".json" or p.name == "tokenizer.model"))
    if not any(p.suffix == ".safetensors" for p in names):
        raise ValueError(f"Missing local safetensors weights: {path}")
    return fingerprint({p.name:sha256(p) for p in names})


def load_local(path, device):
    path = Path(path).resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Download/restore the SigLIP 2 snapshot first: {path}")
    processor = AutoProcessor.from_pretrained(str(path), local_files_only=True, use_fast=False)
    model = AutoModel.from_pretrained(str(path), local_files_only=True,
                                      attn_implementation="sdpa", dtype=torch.float32)
    if model.config.model_type != "siglip":
        raise ValueError("This experiment expects the fixed-resolution SiglipModel checkpoint")
    return model.to(device), processor


def autocast(device, precision):
    return torch.autocast("cuda", dtype=torch.bfloat16) if str(device).startswith("cuda") and precision == "bf16" else nullcontext()


def tokenize(processor, texts, max_length):
    return processor.tokenizer([normalize_text(t) for t in texts], padding="max_length",
        truncation=True, max_length=max_length, return_tensors="pt")


class PairDataset(Dataset):
    def __init__(self, pairs, items, catalog, processor, max_length):
        self.pairs, self.catalog = pairs, Path(catalog)
        self.items = {r["item_id"]:r for r in items}
        self.image_processor = processor.image_processor
        encoded = tokenize(processor, [p["text"] for p in pairs], max_length)
        self.input_ids = encoded["input_ids"]
        self.attention_mask = encoded.get("attention_mask", torch.ones_like(self.input_ids))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        row = self.items[self.pairs[i]["item_id"]]
        image = rgb_image(self.catalog / row["image_path"])
        pixels = self.image_processor(images=image, return_tensors="pt")["pixel_values"][0]
        return dict(pixel_values=pixels, input_ids=self.input_ids[i],
                    attention_mask=self.attention_mask[i], row_index=i,
                    weight=float(self.pairs[i]["weight"]))


@torch.inference_mode()
def encode_images(model, processor, items, catalog, device, batch_size, precision):
    model.eval()
    parts = []
    for start in range(0, len(items), batch_size):
        images = [rgb_image(Path(catalog) / r["image_path"]) for r in items[start:start+batch_size]]
        values = processor.image_processor(images=images, return_tensors="pt")["pixel_values"].to(device)
        with autocast(device, precision):
            vectors = model.get_image_features(pixel_values=values)
        parts.append(torch.nn.functional.normalize(vectors.float(), dim=-1).cpu().numpy())
        if (not dist.is_initialized() or dist.get_rank() == 0) and (start == 0 or (start//batch_size)%50 == 0 or start+batch_size >= len(items)):
            print(f"Image candidates on rank 0: {min(start+batch_size,len(items))}/{len(items)}",flush=True)
    dim = model.config.text_config.projection_size
    return np.concatenate(parts) if parts else np.empty((0,dim), dtype=np.float32)


@torch.inference_mode()
def encode_texts(model, processor, texts, device, batch_size, precision, max_length):
    model.eval()
    parts = []
    for start in range(0,len(texts),batch_size):
        batch = tokenize(processor, texts[start:start+batch_size], max_length)
        inputs = {k:v.to(device) for k,v in batch.items() if k in {"input_ids", "attention_mask"}}
        with autocast(device, precision):
            vectors = model.get_text_features(**inputs)
        parts.append(torch.nn.functional.normalize(vectors.float(), dim=-1).cpu().numpy())
    dim = model.config.text_config.projection_size
    return np.concatenate(parts) if parts else np.empty((0,dim), dtype=np.float32)
