#!/usr/bin/env python3
"""Export source/docs/evidence without runtime data, environments or checkpoints.

Uses only the Python standard library and works on an offline pod.
The export does not modify the source tree and never overwrites an archive.
"""
import argparse
from pathlib import Path
import re
from zipfile import ZipFile, ZIP_DEFLATED

ROOT_FILES = {"README.md", "README_CLIP.md", "QUICKSTART.md", "LICENSE", "NOTICE",
              "pyproject.toml", "app.py", "app_siglip2.py", ".gitignore"}
ROOT_DIRS = {"sticker_search", "scripts", "tests", "configs", "requirements",
             "docs", "evidence", "output"}
SKIP_DIRS = {".git", ".venv", ".venv-transfer", "__pycache__", ".pytest_cache",
             ".cache", "models", "checkpoints", "downloads", "runs", "node_modules"}
SKIP_SUFFIXES = {".pyc", ".pyo", ".pt", ".pth", ".safetensors", ".onnx",
                 ".bin", ".npy", ".npz", ".log", ".pem", ".key"}
TEXT_SUFFIXES = {".py", ".sh", ".md", ".json", ".jsonl", ".txt", ".toml",
                 ".yaml", ".yml", ".env", ".example", ".html", ".csv"}
SECRET = re.compile(rb"(?<![A-Za-z0-9_])(?:hf_[A-Za-z0-9]{25,}|gh[pousr]_[A-Za-z0-9]{20,}"
                    rb"|github_pat_[A-Za-z0-9_]{30,})"
                    rb"|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")


def export(root, output):
    root = root.resolve()
    output = output.absolute()
    if output.exists():
        raise ValueError("Output already exists; choose a new filename")
    selected = []
    for source in sorted(root.rglob("*")):
        rel = source.relative_to(root)
        if source.is_symlink() or not source.is_file():
            continue
        if rel.parts[0] not in ROOT_DIRS and rel.as_posix() not in ROOT_FILES:
            continue
        if any(part in SKIP_DIRS or part.endswith(".egg-info") for part in rel.parts):
            continue
        if source.name == ".DS_Store" or source.name.startswith(".env"):
            continue
        if source.suffix.lower() in SKIP_SUFFIXES:
            continue
        if source.suffix == ".env" and rel.as_posix() != "configs/offline.env":
            continue
        if source.stat().st_size >= 50 * 1024**2:
            raise ValueError(f"File exceeds export limit: {rel}")
        if source.suffix in TEXT_SUFFIXES and SECRET.search(source.read_bytes()):
            raise ValueError(f"Possible credential in {rel}; values are not printed")
        selected.append((source, rel))
    if not any(rel.as_posix() == "README.md" for _, rel in selected):
        raise ValueError("Project README.md was not found")
    output.parent.mkdir(parents=True, exist_ok=True)
    archive = ZipFile(output, "x", ZIP_DEFLATED)
    try:
        with archive:
            for source, rel in selected:
                archive.write(source, "sticker_search/" + rel.as_posix())
    except Exception:
        # Remove only the incomplete new archive created by this invocation.
        if output.is_file():
            output.unlink()
        raise
    print(f"Saved {len(selected)} files to {output} ({output.stat().st_size / 1024**2:.1f} MiB)")
    print("Excluded runtime data, checkpoints and local .env files. Review before publishing.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export(args.root, args.output)
