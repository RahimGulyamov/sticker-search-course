"""CPU-only diagnostic tests with synthetic embeddings, never quality claims."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from sticker_search.common import sha256, write_json, write_jsonl
from sticker_search.diagnose_retrieval import reciprocal_ranks, run, score_methods, top_positions


class RetrievalDiagnosticTests(unittest.TestCase):
    def test_rrf_excludes_missing_features_and_keeps_stable_ties(self):
        scores = np.array([[.5, 100, .5, .1]], dtype=np.float32)
        ranks = reciprocal_ranks(scores, np.array([True, False, True, True]))
        np.testing.assert_allclose(ranks, [[1/61, 0, 1/62, 1/63]])
        np.testing.assert_array_equal(reciprocal_ranks(scores, np.zeros(4, dtype=bool)), np.zeros_like(scores))

    def test_hybrid_matches_original_itemwise_weights(self):
        branches = [np.array([[.2, .4]]), np.array([[.7, .6]]), np.array([[0., .8]])]
        present = np.array([[1, 1, 0], [1, 1, 1]], dtype=bool)
        methods = dict(score_methods(branches, present))
        np.testing.assert_allclose(methods["hybrid"], [[(.65*.2+.25*.7)/.9, .65*.4+.25*.6+.1*.8]], rtol=1e-6)
        np.testing.assert_allclose(methods["linear_fixed"], [[.65*.2+.25*.7, .65*.4+.25*.6+.1*.8]], rtol=1e-6)
        self.assertTrue(np.isneginf(methods["ocr"][0, 0]))
        self.assertEqual(top_positions(methods["ocr"][0]).tolist(), [1])

    def make_fixture(self, root):
        items = [dict(item_id=f"i{i}", group_id=f"g{i}", split="dev", kind="sticker", source="fixture",
                      caption_text="", ocr_text="") for i in range(4)]
        index = root / "index"
        write_jsonl(index / "items.jsonl", items)
        write_json(index / "metadata.json", dict(item_ids=[r["item_id"] for r in items], text_model="fixture", text_revision="r1"))
        np.savez_compressed(index / "vectors.npz", image=np.eye(4, dtype=np.float32),
                            caption=np.zeros((4, 4), dtype=np.float32), ocr=np.zeros((4, 4), dtype=np.float32))
        queries = root / "queries"
        en = dict(query_id="q_en", text="hello", language="en", split="dev", group_id="g0",
                  positive_item_ids=["i0"], label_source="human_source")
        ru = en | dict(query_id="q_ru", text="привет", language="ru", original_query_id="q_en",
                       original_text="hello", label_source="human_source_machine_translation")
        test = en | dict(query_id="not_evaluated", split="test", positive_item_ids=["not_in_catalog"])
        write_jsonl(queries / "queries.jsonl", [en, ru, test])
        write_json(queries / "metadata.json", dict(model="fixture", revision="r1", source_sha256=sha256(queries / "queries.jsonl")))
        np.save(queries / "queries.npy", np.array([[1,0,0,0], [1,0,0,0], [0,1,0,0]], dtype=np.float32))
        return SimpleNamespace(index=str(index), queries=str(queries), output=str(root / "report"), batch_size=2)

    def test_end_to_end_uses_dev_and_missing_ocr_is_not_a_random_result(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.make_fixture(Path(directory))
            with contextlib.redirect_stdout(io.StringIO()):
                run(args)
            report = json.loads((Path(args.output) / "summary.json").read_text())
            self.assertEqual(report["query_count"], 2)
            self.assertEqual(report["translation_pair_cosine"]["mean"], 1)
            methods = report["methods"]["all"]
            self.assertEqual(methods["image"]["metrics"]["NDCG@5"]["mean"], 1)
            self.assertEqual(methods["ocr"]["metrics"]["NDCG@5"]["mean"], 0)
            self.assertEqual(methods["ocr"]["unique_top10_items"], 0)
            self.assertEqual(methods["rrf_80_15_05"]["metrics"]["NDCG@5"]["mean"], 1)
            self.assertEqual(methods["ocr"]["paired_delta_vs_image"]["NDCG@5_delta"]["mean"], -1)
            with self.assertRaisesRegex(ValueError, "output exists"):
                run(args)

    def test_vector_order_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.make_fixture(Path(directory))
            write_json(Path(args.index) / "metadata.json", dict(item_ids=["i3", "i2", "i1", "i0"]))
            with self.assertRaisesRegex(ValueError, "order"):
                run(args)

    def test_text_encoder_revision_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.make_fixture(Path(directory))
            path = Path(args.queries) / "metadata.json"
            meta = json.loads(path.read_text())
            write_json(path, meta | dict(revision="other"))
            with self.assertRaisesRegex(ValueError, "encoder/revision mismatch"):
                run(args)


if __name__ == "__main__":
    unittest.main()
