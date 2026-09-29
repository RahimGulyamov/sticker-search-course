"""CPU-only regression tests: python -m unittest discover -s tests -p test_annotation_recovery.py."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from sticker_search.annotate import merge, parse_answer, recover_errors, validate_resume
from sticker_search.common import fingerprint, read_jsonl, sha256, write_json, write_jsonl


def answer(language="ja"):
    return json.dumps(dict(caption_en="A waving cat", caption_ru="Кот машет лапой",
                           emotion_ru="радость", style_ru="мультяшный",
                           ocr_text="こんにちは", ocr_language=language,
                           usage_queries_ru=["привет от котика", "поздороваться"]), ensure_ascii=False)


class AnnotationRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.catalog = self.root / "catalog"
        self.output = self.root / "captions_500"
        self.args = SimpleNamespace(task="caption", catalog=str(self.catalog), output=str(self.output))

    def make_run(self):
        keys = [f"item_{i:03}" for i in range(500)]
        write_jsonl(self.catalog / "manifest.jsonl", [dict(item_id=k) for k in keys])
        # Match the 16 failures' language distribution from the submitted log.
        languages = ["ja"] * 6 + ["ko"] * 3 + ["pt", "it", "es", "es", "id", "ar", "ar"]
        config = dict(task="caption", world_size=4, selected_keys=keys,
                      input_sha256=sha256(self.catalog / "manifest.jsonl"),
                      model="saved/model", revision="a" * 40,
                      prompt_sha256=fingerprint("original prompt, before this fix"))
        for rank in range(4):
            write_json(self.output / f"config-rank{rank:02}.json", config)
            write_jsonl(self.output / f"part-rank{rank:02}.jsonl",
                        [dict(key=k, annotation={"caption_ru": "Existing annotation"})
                         for i, k in enumerate(keys) if i % 4 == rank and i >= 16])
            write_jsonl(self.output / f"errors-rank{rank:02}.jsonl",
                        [dict(key=keys[i], error="Invalid OCR language", response=answer(lang))
                         for i, lang in enumerate(languages) if i % 4 == rank])
        return config

    def test_language_normalization_keeps_raw_prediction(self):
        for lang in ["ja", "ko", "pt", "it", "es", "id", "ar"]:
            with self.subTest(lang=lang):
                result = parse_answer("```json\n" + answer(lang) + "\n```", "caption")
                self.assertEqual(result["ocr_language"], "other")
                self.assertEqual(result["ocr_language_raw"], lang)
        self.assertEqual(parse_answer(answer("en-US"), "caption")["ocr_language"], "en")
        self.assertEqual(parse_answer(answer("zh_Hant"), "caption")["ocr_language"], "zh")
        for lang in ["ru", "en", "zh", "mixed", "other", "none"]:
            self.assertEqual(parse_answer(answer(lang), "caption")["ocr_language"], lang)
        for lang in ["", "some explanation", "123"]:
            with self.subTest(lang=lang), self.assertRaises(ValueError):
                parse_answer(answer(lang), "caption")

    def test_484_to_500_idempotence_provenance_and_byte_preservation(self):
        config = self.make_run()
        before = {p: p.read_bytes() for p in self.output.glob("*.json*")}
        with contextlib.redirect_stdout(io.StringIO()) as log:
            recover_errors(self.args)
        self.assertIn("Recovered 16", log.getvalue())
        self.assertIn("complete 500/500", log.getvalue())
        for path, content in before.items():
            if path.name.startswith("part-"):
                self.assertTrue(path.read_bytes().startswith(content))
            else:
                self.assertEqual(path.read_bytes(), content)
        after = {p: p.read_bytes() for p in self.output.glob("*.json*")}
        with contextlib.redirect_stdout(io.StringIO()) as log:
            recover_errors(self.args)
        self.assertIn("Recovered 0", log.getvalue())
        self.assertEqual(after, {p: p.read_bytes() for p in self.output.glob("*.json*")})
        with contextlib.redirect_stdout(io.StringIO()):
            merge(self.args)
        merged = read_jsonl(self.output / "annotations.jsonl")
        self.assertEqual(len(merged), 500)
        self.assertEqual(len({r["item_id"] for r in merged}), 500)
        self.assertEqual([r["item_id"] for r in merged], config["selected_keys"])
        self.assertEqual(sum("recovery" in r for r in merged), 16)
        self.assertEqual({r["prompt_sha256"] for r in merged}, {config["prompt_sha256"]})
        self.assertEqual(merged[0]["ocr_language_raw"], "ja")

    def test_invalid_response_remains_missing_and_merge_rejects_it(self):
        self.make_run()
        path = self.output / "errors-rank00.jsonl"
        errors = read_jsonl(path)
        errors[0]["response"] = '{"caption_ru": "truncated'
        write_jsonl(path, errors)
        with contextlib.redirect_stdout(io.StringIO()):
            recover_errors(self.args)
        with self.assertRaisesRegex(ValueError, "Incomplete annotation: 499/500"):
            merge(self.args)
        self.assertFalse((self.output / "annotations.jsonl").exists())

    def test_duplicate_attempt_does_not_create_duplicate_annotation(self):
        self.make_run()
        path = self.output / "errors-rank00.jsonl"
        errors = read_jsonl(path)
        write_jsonl(path, [errors[0]] + errors)
        with contextlib.redirect_stdout(io.StringIO()):
            recover_errors(self.args)
            merge(self.args)
        self.assertEqual(len(read_jsonl(self.output / "annotations.jsonl")), 500)

    def test_foreign_key_aborts_before_any_write(self):
        self.make_run()
        path = self.output / "errors-rank03.jsonl"
        errors = read_jsonl(path)
        errors[-1]["key"] = "foreign_catalog_key"
        write_jsonl(path, errors)
        before = {p: p.read_bytes() for p in self.output.glob("*.json*")}
        with self.assertRaisesRegex(ValueError, "Unexpected error key/rank"):
            recover_errors(self.args)
        self.assertEqual(before, {p: p.read_bytes() for p in self.output.glob("*.json*")})

    def test_changed_input_is_rejected_before_recovery(self):
        self.make_run()
        write_jsonl(self.catalog / "manifest.jsonl", [dict(item_id="changed")])
        with self.assertRaisesRegex(ValueError, "input differs"):
            recover_errors(self.args)

    def test_literal_ocr_newlines_and_named_language_are_preserved(self):
        payload = json.loads(answer("punjabi"))
        payload["ocr_text"] = "first\nsecond\tword"
        response = json.dumps(payload).replace(r"\n", "\n").replace(r"\t", "\t")
        parsed = parse_answer(response, "caption")
        self.assertEqual(parsed["ocr_text"], payload["ocr_text"])
        self.assertEqual(parsed["ocr_language"], "other")
        self.assertEqual(parsed["ocr_language_raw"], "punjabi")
        with self.assertRaises(ValueError):
            parse_answer(response[:-4], "caption")

    def test_cpu_recovery_copies_generation_provenance(self):
        self.make_run()
        path = self.output / "errors-rank00.jsonl"
        errors = read_jsonl(path)
        errors[0]["generation"] = dict(max_new_tokens=1024, retry_missing=True)
        write_jsonl(path, errors)
        with contextlib.redirect_stdout(io.StringIO()):
            recover_errors(self.args)
            merge(self.args)
        first = read_jsonl(self.output / "annotations.jsonl")[0]
        self.assertEqual(first["generation"]["max_new_tokens"], 1024)
        self.assertEqual(first["recovery"]["parser_version"], 3)

    def test_retry_changes_only_token_budget_and_preserves_original_config(self):
        old = dict(task="caption", max_new_tokens=384, max_pixels=100,
                   model="original", revision="a" * 40, world_size=4,
                   selected_keys=["a", "b"], prompt_sha256="prompt", input_sha256="input")
        old_copy = dict(old)
        new = old | dict(max_new_tokens=1024)
        validate_resume(old, new, retry_missing=True)
        self.assertEqual(old, old_copy)
        with self.assertRaisesRegex(ValueError, "configuration changed"):
            validate_resume(old, new)
        for field, value in [("prompt_sha256", "new"), ("input_sha256", "new"),
                             ("world_size", 2), ("max_pixels", 101), ("selected_keys", ["a"])]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_resume(old, new | {field: value}, retry_missing=True)
        with self.assertRaisesRegex(ValueError, "must not decrease"):
            validate_resume(old, old | dict(max_new_tokens=128), retry_missing=True)

    def test_explicit_image_only_fallback_is_bounded_and_reported(self):
        self.make_run()
        path = self.output / "errors-rank00.jsonl"
        errors = read_jsonl(path)
        errors[0]["response"] = "no JSON"
        write_jsonl(path, errors)
        with contextlib.redirect_stdout(io.StringIO()):
            recover_errors(self.args)
        self.args.max_missing_fraction = 0.001  # 1/500 exceeds this limit.
        with self.assertRaisesRegex(ValueError, "Incomplete annotation"):
            merge(self.args)
        self.args.max_missing_fraction = 0.005
        with contextlib.redirect_stdout(io.StringIO()):
            merge(self.args)
        rows = read_jsonl(self.output / "annotations.jsonl")
        coverage = json.loads((self.output / "coverage.json").read_text())
        self.assertEqual(len(rows), 499)
        self.assertEqual(coverage["expected"], 500)
        self.assertEqual(coverage["missing_keys"], ["item_000"])
        self.assertEqual(coverage["fallback"], "image_only_features_for_missing_captions")
        self.assertEqual(len(read_jsonl(self.catalog / "manifest.jsonl")), 500)


if __name__ == "__main__":
    unittest.main()
