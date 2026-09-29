"""Exercise Bash stage routing with a recording Python stub; no models/GPU needed."""
import json
import contextlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from sticker_search.common import read_jsonl, sha256, write_json, write_jsonl


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_gpu.sh"


class PipelineResumeTests(unittest.TestCase):
    def invoke(self, stage, fail_merge=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "bin"
            binary.mkdir()
            stub = binary / "python"
            stub.write_text(f"#!{sys.executable}\n" + '''import json, os, sys
with open(os.environ["STICKER_TEST_TRACE"], "a") as stream:
    stream.write(json.dumps(sys.argv[1:]) + "\\n")
if os.environ.get("STICKER_TEST_FAIL_MERGE") == "1" and "translate" in sys.argv and "--merge" in sys.argv:
    sys.exit(1)
''')
            stub.chmod(0o700)
            run = root / "run"
            (run / "captions").mkdir(parents=True)
            (run / "captions/annotations.jsonl").write_text("saved captions\n")
            (run / "translations").mkdir()
            (run / "translations/config-rank00.json").write_text("{}")
            trace = root / "trace.jsonl"
            env = os.environ | dict(PATH=str(binary) + os.pathsep + os.environ["PATH"],
                STICKER_RUN_ROOT=str(run), STICKER_TEST_TRACE=str(trace),
                STICKER_TEST_FAIL_MERGE="1" if fail_merge else "0")
            result = subprocess.run(["bash", str(SCRIPT), "--from", stage], cwd=root,
                                    env=env, capture_output=True, text=True)
            calls = [json.loads(line) for line in trace.read_text().splitlines()] if trace.exists() else []
            return result, calls

    def test_translation_resume_skips_captions_and_retries_missing(self):
        result, calls = self.invoke("translate")
        self.assertEqual(result.returncode, 0, result.stderr)
        annotation_calls = [call for call in calls if "sticker_search.annotate" in call]
        self.assertFalse(any("caption" in call for call in annotation_calls))
        retries = [call for call in annotation_calls if "--retry-missing" in call]
        self.assertEqual(len(retries), 1)
        self.assertIn("512", retries[0])
        self.assertFalse(any("96" in call for call in annotation_calls))
        self.assertTrue(any("sticker_search.features" in call for call in calls))
        self.assertTrue(any("sticker_search.train" in call for call in calls))
        self.assertEqual(sum("sticker_search.evaluate" in call for call in calls), 2)

    def test_failed_translation_merge_stops_before_index_and_train(self):
        result, calls = self.invoke("translate", fail_merge=True)
        self.assertEqual(result.returncode, 1)
        self.assertFalse(any("sticker_search.features" in call for call in calls))
        self.assertFalse(any("sticker_search.train" in call for call in calls))
        self.assertEqual(calls[-1][0], "-")  # Failure report block, after preflight.
        self.assertEqual(sum(call[0] == "-" for call in calls), 2)

    def test_index_resume_skips_annotation(self):
        result, calls = self.invoke("index")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(any("sticker_search.annotate" in call for call in calls))
        self.assertTrue(any("sticker_search.features" in call for call in calls))

    def test_evaluation_resume_does_not_retrain(self):
        result, calls = self.invoke("evaluate")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(any("sticker_search.train" in call or "sticker_search.features" in call for call in calls))
        self.assertEqual(sum("sticker_search.evaluate" in call for call in calls), 2)

    def test_unknown_stage_launches_nothing(self):
        result, calls = self.invoke("typo")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(calls, [])

    def test_failure_report_contains_latest_response_for_missing_query_only(self):
        code = SCRIPT.read_text().split("<<'PY_REPORT'\n", 1)[1].split("\nPY_REPORT", 1)[0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog, run = root / "catalog", root / "run"
            source = [dict(query_id="q0", text="hello"), dict(query_id="q1", text="bye")]
            path = catalog / "labels/queries_en.jsonl"
            write_jsonl(path, source)
            output = run / "translations"
            write_json(output / "config-rank00.json", dict(task="translate", world_size=1,
                selected_keys=["q0", "q1"], input_sha256=sha256(path)))
            write_jsonl(output / "part-rank00.jsonl", [dict(key="q0", annotation=dict(text_ru="привет"))])
            write_jsonl(output / "errors-rank00.jsonl", [
                dict(key="q0", response="old, later recovered", error="invalid"),
                dict(key="q1", response="first attempt", error="invalid"),
                dict(key="q1", response="latest attempt", error="invalid")])
            with patch.object(sys, "argv", ["script", str(catalog), str(run)]), contextlib.redirect_stdout(io.StringIO()):
                exec(compile(code, "failure_report", "exec"), {})
            rows = read_jsonl(output / "failures_latest.jsonl")
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["key"], "q1")
            self.assertEqual(rows[0]["source"], source[1])
            self.assertEqual(rows[0]["latest_error"]["response"], "latest attempt")


if __name__ == "__main__":
    unittest.main()
