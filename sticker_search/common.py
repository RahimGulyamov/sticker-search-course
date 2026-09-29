"""Small, dependency-light utilities shared by offline and online stages."""
from __future__ import annotations

import hashlib
import json
import random
import struct
import time
import urllib.request
from pathlib import Path

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def download(url, path, expected_sha256=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and (expected_sha256 is None or sha256(path) == expected_sha256):
        return path
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "sticker-course/0.1"})
            with urllib.request.urlopen(req, timeout=45) as response:
                temporary = path.with_suffix(path.suffix + ".part")
                with temporary.open("wb") as out:
                    for block in iter(lambda: response.read(1024 * 1024), b""):
                        out.write(block)
            if expected_sha256 and sha256(temporary) != expected_sha256:
                raise ValueError(f"Checksum mismatch: {path.name}")
            temporary.replace(path)
            return path
        except Exception:
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def rgb_image(path):
    """Composite alpha on white; converting RGBA directly to RGB makes false backgrounds."""
    with Image.open(path) as image:
        image.seek(0)
        image = ImageOps.exif_transpose(image).convert("RGBA")
        bg = Image.new("RGBA", image.size, "white")
        return Image.alpha_composite(bg, image).convert("RGB")


def image_properties(path):
    import numpy as np

    with Image.open(path) as image:
        frames = getattr(image, "n_frames", 1)
        image.seek(0)
        rgba = ImageOps.exif_transpose(image).convert("RGBA")
        if rgba.width * rgba.height > 16_000_000:
            raise ValueError("Image exceeds pixel limit")
        arr = np.asarray(rgba).copy()
        arr[arr[:, :, 3] == 0, :3] = 0  # Hidden RGB must not defeat exact deduplication.
        pixel_hash = hashlib.sha256(struct.pack("<II", *rgba.size) + arr.tobytes()).hexdigest()
        rgb = Image.alpha_composite(Image.new("RGBA", rgba.size, "white"), rgba).convert("RGB")
        small = np.asarray(rgb.resize((9, 8)).convert("L"))
        bits = small[:, 1:] > small[:, :-1]
        dhash = sum(int(bit) << i for i, bit in enumerate(bits.flat))
        mean_rgb = np.asarray(rgb.resize((1, 1)))[0, 0].tolist()
        return dict(width=rgba.width, height=rgba.height, n_frames=frames,
                    has_transparency=bool((arr[:, :, 3] < 255).any()),
                    alpha_fraction=float((arr[:, :, 3] < 255).mean()),
                    pixel_sha256=pixel_hash, file_sha256=sha256(path),
                    dhash=f"{dhash:016x}", mean_rgb=mean_rgb)


def set_seed(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def model_revision(model_id):
    revisions = json.loads((ROOT / "configs/model_revisions.json").read_text())
    revision = revisions[model_id]
    if len(revision) != 40:
        raise ValueError(f"Invalid pinned model revision for {model_id}")
    return revision
