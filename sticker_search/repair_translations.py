"""Apply explicit, source-checked assistant translations without model inference.

Run with stopped annotation workers. Append only missing keys to their original
rank shards; never overwrite existing annotations, configs, or error logs.
The JSON correction list is an auditable artifact, not human-validated gold.
"""
from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path

from .annotate import load_saved_run, merge, parse_answer
from .common import read_jsonl, sha256, write_json


def repair(args):
    if args.task != "translate":
        raise ValueError("Correction patches are only for translations")
    output = Path(args.output)
    config, records = load_saved_run(args)
    corrections_path = Path(args.corrections)
    corrections = json.loads(corrections_path.read_text(encoding="utf-8"))
    if corrections.get("schema_version") != 1 or corrections.get("author") != "OpenAI assistant":
        raise ValueError("Unsupported correction schema/author")
    entries = corrections["entries"]
    fixes = {entry["key"]: entry for entry in entries}
    if len(fixes) != len(entries):
        raise ValueError("Duplicate correction keys")
    selected = config["selected_keys"]
    if len(selected) != corrections["expected_total"]:
        raise ValueError("Correction patch targets a different query selection")
    owners = {key: i % config["world_size"] for i, key in enumerate(selected)}
    queries = read_jsonl(args.queries)
    original = {q["query_id"]: q for q in queries}
    if len(original) != len(queries):
        raise ValueError("Duplicate source query IDs")
    missing = set(selected) - records.keys()
    if missing - fixes.keys():
        raise ValueError(f"Missing keys outside this patch: {sorted(missing - fixes.keys())}")

    # Validate every correction and every input before the first write. Comparing
    # the full source row protects English text, split, positives and group ID.
    patch_sha = sha256(corrections_path)
    pending = {}
    for key, entry in fixes.items():
        if key not in owners or original.get(key) != entry["source"]:
            raise ValueError(f"Correction source mismatch: {key}")
        annotation = parse_answer(json.dumps({"text_ru": entry["text_ru"]}), "translate")
        if len(annotation["text_ru"]) > 200 or annotation["text_ru"] != entry["text_ru"]:
            raise ValueError(f"Invalid correction length/whitespace: {key}")
        if not isinstance(entry["ambiguous"], bool) or not isinstance(entry["note"], str):
            raise ValueError(f"Invalid correction metadata: {key}")
        provenance = dict(method="assistant_correction", author=corrections["author"],
            patch_id=corrections["patch_id"], patch_sha256=patch_sha,
            correction_file=corrections_path.name, original_text=entry["source"]["text"],
            strategy=entry["strategy"], ambiguous=entry["ambiguous"], note=entry["note"],
            human_reviewed=False, based_on="source_query_text_only")
        if key in records:
            saved_provenance = records[key].get("translation_provenance", {})
            if saved_provenance.get("method") == "assistant_correction":
                if records[key]["annotation"] != annotation or saved_provenance != provenance:
                    raise ValueError(f"Existing correction differs; refusing overwrite: {key}")
            continue
        row = dict(key=key, annotation=annotation, translation_provenance=provenance)
        pending.setdefault(owners[key], []).append(row)

    # Replacement is atomic per shard and retains the exact original prefix.
    # If interrupted between shards, a second invocation skips the applied rows.
    for rank, rows in sorted(pending.items()):
        part = output / f"part-rank{rank:02}.jsonl"
        previous = part.read_bytes() if part.exists() else b""
        separator = b"\n" if previous and not previous.endswith(b"\n") else b""
        addition = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode("utf-8")
        temporary = part.with_name(part.name + f".correction-{uuid.uuid4().hex}.tmp")
        temporary.write_bytes(previous + separator + addition)
        temporary.replace(part)

    _, final = load_saved_run(args)
    applied = sorted(key for key in fixes
                     if final[key].get("translation_provenance", {}).get("patch_sha256") == patch_sha)
    write_json(output / "translation_corrections_audit.json", dict(
        patch_id=corrections["patch_id"], patch_sha256=patch_sha,
        expected=len(selected), complete=len(final), assistant_corrected_keys=applied,
        assistant_corrected_count=len(applied), human_reviewed=False,
        ambiguous_keys=[key for key in applied if fixes[key]["ambiguous"]],
        preserved_existing_keys=sorted(set(final) - set(applied))))
    added = sum(map(len, pending.values()))
    print(f"Added {added} assistant corrections on CPU; complete {len(final)}/{len(selected)}.")
    merge(args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", default="data/catalog/labels/queries_en.jsonl")
    parser.add_argument("--output", default="runs/full_v2/translations")
    parser.add_argument("--corrections", default="configs/translation_corrections_v1.json")
    args = parser.parse_args()
    args.task = "translate"
    repair(args)


if __name__ == "__main__":
    main()
