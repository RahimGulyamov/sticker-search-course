"""Freeze selected artifacts, evaluate test, render review, and check generation."""
import argparse
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import subprocess
import sys
import zipfile

from .common import ROOT, model_revision, read_jsonl, sha256, write_json

GENERATION_PROMPTS = [
    "весёлый кот с чашкой кофе",
    "панда обнимает маленькое сердечко",
    "сонная сова в ночном колпаке",
    "маленький робот играет на скрипке в космосе",
    "радостное облачко с тонкой радугой",
]


def freeze(out):
    import torch
    selection_path = ROOT / "configs/final_selection_v1.json"
    choice = json.loads(selection_path.read_text())
    checkpoint = Path(choice["checkpoint"])
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved["epoch"] != choice["expected_epoch"]:
        raise ValueError("Selected checkpoint epoch differs from the declared dev choice")
    index = Path("runs/full_v2/index")
    index_hash = sha256(index / "vectors.npz")
    if saved["index_sha256"] != index_hash:
        raise ValueError("Checkpoint/index mismatch")
    files = [checkpoint, index / "vectors.npz", index / "items.jsonl", selection_path]
    for lang in ["en", "ru"]:
        directory = Path(f"runs/full_v2/queries_{lang}")
        rows = read_jsonl(directory / "queries.jsonl")
        if not any(q["split"] == "test" for q in rows):
            raise ValueError(f"No test queries for {lang}")
        files.extend([directory / "queries.jsonl", directory / "queries.npy"])
    code = [ROOT / "sticker_search" / f"{name}.py" for name in
            ["evaluate", "retrieval", "model", "metrics", "fusion"]]
    frozen = dict(selection=choice, file_hashes={str(p): sha256(p) for p in files},
                  evaluation_code_hashes={p.name: sha256(p) for p in code},
                  index_sha256=index_hash, selected_epoch=saved["epoch"],
                  note="Created before test evaluation. Existing test outputs are reused only with identical frozen artifacts/code.")
    path = out / "frozen_selection.json"
    if path.exists():
        if json.loads(path.read_text()) != frozen:
            raise ValueError("Frozen artifacts changed; do not overwrite the final evaluation")
    else:
        if any((out / f"test_{lang}/metrics.json").exists() for lang in ["en", "ru"]):
            raise ValueError("Test outputs exist without a frozen selection")
        write_json(path, frozen)
    packages = {}
    for name in ["torch", "transformers", "sentence-transformers", "numpy", "Pillow", "scikit-learn", "diffusers", "rembg", "onnxruntime", "streamlit"]:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    write_json(out / "environment.json", dict(python=sys.version, packages=packages,
        cuda=torch.version.cuda, gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]))
    return choice


def command(module, arguments, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen([sys.executable, "-u", "-m", module, *arguments],
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            print(line, end="", flush=True)
            stream.write(line)
            stream.flush()
        status = process.wait()
    if status:
        raise RuntimeError(f"{module} exited with {status}; see {log}")


def generation_success(path):
    metadata = path / "generation.json"
    if not metadata.exists() or not (path / "sticker.png").exists():
        return False
    record = json.loads(metadata.read_text())
    alpha = record["alpha_range"]
    return alpha[0] < alpha[1] and record["output_sha256"] == sha256(path / "sticker.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-generation", action="store_true", help="Explicitly record generation as not checked")
    args = parser.parse_args()
    out = Path("runs/final_v1")
    choice = freeze(out)
    status = dict(test="pending", review="pending", generation="pending", errors=[])
    current_stage = "test"
    try:
        for lang in ["en", "ru"]:
            target = out / f"test_{lang}"
            if (target / "metrics.json").exists() and (target / "cases.jsonl").exists():
                print(f"Reusing frozen test evaluation: {lang}", flush=True)
                continue
            command("sticker_search.evaluate", ["--index", "runs/full_v2/index", "--queries", f"runs/full_v2/queries_{lang}",
                "--checkpoint", choice["checkpoint"], "--methods", *choice["comparison_methods"],
                "--split", "test", "--output", str(target)], out / "logs" / f"test_{lang}.log")
        status["test"] = "complete"
        current_stage = "review"
        review = out / "review"
        if not all((review / name).exists() for name in ["search_review.html", "search_examples.jsonl", "review_protocol.json"]):
            command("sticker_search.review_search", ["--output", str(review)], out / "logs/review.log")
        status["review"] = "complete_unjudged"
        current_stage = "generation"
        if args.skip_generation:
            status["generation"] = "explicitly_skipped"
        else:
            for i, prompt in enumerate(GENERATION_PROMPTS):
                parent = out / "generation" / f"example_{i+1:02}"
                prior = sorted(parent.glob("attempt_*"))
                if any(generation_success(path) for path in prior):
                    print(f"Reusing successful generation {i+1}", flush=True)
                    continue
                attempt = parent / f"attempt_{len(prior):02}"
                command("sticker_search.generate", [prompt, "--output", str(attempt), "--seed", str(42+i)],
                        out / "logs" / f"generation_{i+1:02}_{len(prior):02}.log")
                if not generation_success(attempt):
                    raise ValueError(f"Generation/mask check failed: {attempt}")
            status["generation"] = "five_examples_created_visual_review_required"
    except Exception as error:
        status[current_stage] = "failed"
        status["errors"].append(str(error))
        print(f"Final checks stopped: {error}", flush=True)
    finally:
        write_json(out / "status.json", status)
        target = out / "final_results.zip"
        with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(out.rglob("*")):
                if path.is_file() and path != target and path.suffix in {".json", ".jsonl", ".html", ".png", ".log"}:
                    archive.write(path, path.relative_to(out))
        print(f"Attach this file: {target}", flush=True)
    if status["errors"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
