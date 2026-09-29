"""CPU-only checks using real source queries and synthetic saved translations.

The synthetic successes verify repair mechanics, not translation quality.
"""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from sticker_search.annotate import merge
from sticker_search.common import read_jsonl, sha256, write_json, write_jsonl
from sticker_search.repair_translations import repair

ROOT = Path(__file__).resolve().parents[1]


class TranslationCorrectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / "translations"
        self.queries = self.root / "queries_en.jsonl"
        self.queries.write_bytes((ROOT / "evidence/full_catalog/labels/queries_en.jsonl").read_bytes())
        self.patch = self.root / "corrections.json"
        self.patch.write_bytes((ROOT / "configs/translation_corrections_v1.json").read_bytes())
        self.corrections = json.loads(self.patch.read_text())
        self.fixes = {row["key"]: row for row in self.corrections["entries"]}
        self.source = sorted(read_jsonl(self.queries), key=lambda row: row["query_id"])
        self.config = dict(task="translate", world_size=4,
            selected_keys=[row["query_id"] for row in self.source],
            input_sha256=sha256(self.queries), model="fixture/model", revision="fixture/revision",
            prompt_sha256="fixture/prompt")
        for rank in range(4):
            write_json(self.output / f"config-rank{rank:02}.json", self.config)
            write_jsonl(self.output / f"part-rank{rank:02}.jsonl", [
                dict(key=row["query_id"], annotation=dict(text_ru="Синтетическая успешная запись"))
                for i, row in enumerate(self.source) if i % 4 == rank and row["query_id"] not in self.fixes])
            write_jsonl(self.output / f"errors-rank{rank:02}.jsonl", [
                dict(key=row["query_id"], response='{"text_ru": "нууууу', error="No JSON object")
                for i, row in enumerate(self.source) if i % 4 == rank and row["query_id"] in self.fixes])
        self.args = SimpleNamespace(task="translate", output=str(self.output), queries=str(self.queries),
                                    corrections=str(self.patch))

    def invoke(self):
        with contextlib.redirect_stdout(io.StringIO()) as stream:
            repair(self.args)
        return stream.getvalue()

    def snapshot(self):
        return {p.name: p.read_bytes() for p in self.output.glob("*.json*")}

    def test_4124_to_4171_provenance_splits_and_idempotence(self):
        before = self.snapshot()
        self.assertIn("Added 47 assistant corrections on CPU; complete 4171/4171", self.invoke())
        for name, old in before.items():
            current = (self.output / name).read_bytes()
            if name.startswith("part-"):
                self.assertTrue(current.startswith(old))
            else:
                self.assertEqual(current, old)
        translated = read_jsonl(self.output / "queries_ru.jsonl")
        self.assertEqual(len(translated), 4171)
        self.assertEqual(len({r["query_id"] for r in translated}), 4171)
        for original, row in zip(self.source, translated):
            key = original["query_id"]
            for field in ["split", "group_id", "positive_item_ids"]:
                self.assertEqual(row[field], original[field])
            self.assertEqual(row["original_text"], original["text"])
            self.assertFalse(row["translation_reviewed"])
            provenance = row["translation_provenance"]
            self.assertFalse(provenance["human_reviewed"])
            if key in self.fixes:
                self.assertEqual(row["text"], self.fixes[key]["text_ru"])
                self.assertEqual(provenance["method"], "assistant_correction")
                self.assertNotIn("model", provenance)
            else:
                self.assertEqual(row["text"], "Синтетическая успешная запись")
                self.assertEqual(provenance["model"], "fixture/model")
        after = self.snapshot()
        self.assertIn("Added 0", self.invoke())
        self.assertEqual(after, self.snapshot())
        with contextlib.redirect_stdout(io.StringIO()):
            merge(self.args)
        self.assertEqual(after, self.snapshot())

    def test_mismatched_correction_source_aborts_before_any_write(self):
        for field, value in [("text", "changed"), ("split", "other"), ("positive_item_ids", ["wrong"])]:
            data = json.loads(self.patch.read_text())
            data["entries"][-1]["source"][field] = value
            path = self.root / f"bad_{field}.json"
            write_json(path, data)
            self.args.corrections = str(path)
            before = self.snapshot()
            with self.assertRaisesRegex(ValueError, "Correction source mismatch"):
                self.invoke()
            self.assertEqual(before, self.snapshot())

    def test_changed_query_file_aborts(self):
        self.queries.write_text(self.queries.read_text() + "\n")
        before = self.snapshot()
        with self.assertRaisesRegex(ValueError, "input differs"):
            self.invoke()
        self.assertEqual(before, self.snapshot())

    def test_uncovered_missing_query_aborts_before_any_write(self):
        path = self.output / "part-rank00.jsonl"
        write_jsonl(path, read_jsonl(path)[1:])
        before = self.snapshot()
        with self.assertRaisesRegex(ValueError, "Missing keys outside this patch"):
            self.invoke()
        self.assertEqual(before, self.snapshot())

    def test_partial_prior_application_and_existing_success_are_preserved(self):
        # Simulate interruption after one shard and an independent successful
        # model retry for a second key. Neither row may be overwritten.
        self.invoke()
        keep = next(iter(self.fixes))
        rows_for_key = {row["key"]: row for p in self.output.glob("part-rank*.jsonl")
                        for row in read_jsonl(p)}
        other = list(self.fixes)[1]
        for rank in range(4):
            path = self.output / f"part-rank{rank:02}.jsonl"
            write_jsonl(path, [
                dict(key=key, annotation=dict(text_ru="Уже переведено")) if key == other else rows_for_key[key]
                for i, key in enumerate(self.config["selected_keys"])
                if i % 4 == rank and (key not in self.fixes or key in {keep, other})])
        self.assertIn("Added 45", self.invoke())
        translated = {r["original_query_id"]: r for r in read_jsonl(self.output / "queries_ru.jsonl")}
        self.assertEqual(translated[other]["text"], "Уже переведено")
        self.assertEqual(translated[keep]["translation_provenance"]["method"], "assistant_correction")


if __name__ == "__main__":
    unittest.main()
