"""Independent CPU background removal for generated or existing images."""
import argparse
import os
import time
from pathlib import Path

from PIL import Image, ImageOps

from .common import sha256, write_json


def remove_background(input_path, output_path):
    """Save an RGBA PNG without modifying the input or loading a generator."""
    source, target = Path(input_path), Path(output_path)
    if source.resolve() == target.resolve():
        raise ValueError("Use a separate output path to preserve the original image")
    if target.suffix.lower() != ".png":
        raise ValueError("Background removal output must be a .png file")
    with Image.open(source) as image:
        image = ImageOps.exif_transpose(image).convert("RGBA")

    from rembg import new_session, remove

    start = time.perf_counter()
    session = new_session("u2net", providers=["CPUExecutionProvider"])
    cutout = remove(image, session=session).convert("RGBA")
    target.parent.mkdir(parents=True, exist_ok=True)
    cutout.save(target, format="PNG")
    alpha_min, alpha_max = cutout.getchannel("A").getextrema()
    weights = Path(os.environ.get("U2NET_HOME", str(Path.home()/".u2net")))/"u2net.onnx"
    write_json(target.with_suffix(".background.json"), dict(
        stage="background_removal", input_sha256=sha256(source), output_sha256=sha256(target),
        background_remover="rembg/u2net",
        background_weights_sha256=sha256(weights) if weights.exists() else None,
        alpha_range=[alpha_min, alpha_max], seconds=time.perf_counter()-start,
        valid_mask=alpha_min < alpha_max,
        cutout_warning="Mask quality requires visual review; transparency alone is not segmentation accuracy.",
    ))
    if alpha_min == alpha_max:
        raise RuntimeError("Background remover returned a constant mask; the original is unchanged")
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Path to an existing image")
    parser.add_argument("--output", required=True, help="Path to the RGBA PNG")
    args = parser.parse_args()
    print(remove_background(args.input, args.output))


if __name__ == "__main__":
    main()
