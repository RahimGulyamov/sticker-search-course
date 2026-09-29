"""Rank-fusion regression tests; synthetic values test mechanics, not quality."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from sticker_search.common import sha256, write_json, write_jsonl
from sticker_search.diagnose_retrieval import score_methods
from sticker_search.fusion import fit_query_mean, rank_fusion
from sticker_search.retrieval import SearchIndex


class RankFusionTests(unittest.TestCase):
    def test_fit_excludes_dev_test_and_weak_queries(self):
        rows = [dict(split=s, label_source=l) for s, l in [
            ('train', 'human_source'), ('train', 'human_source_machine_translation'),
            ('dev', 'human_source'), ('test', 'human_source'), ('train', 'vlm_weak')]]
        vectors = np.array([[1.,2.], [3.,4.], [999.,999.], [-999.,-999.], [777.,777.]])
        mean, positions = fit_query_mean(rows, vectors)
        np.testing.assert_array_equal(mean, [2,3])
        self.assertEqual(positions, [0,1])
        vectors[2:] *= 100
        np.testing.assert_array_equal(fit_query_mean(rows, vectors)[0], mean)

    def test_rank_fusion_matches_previously_evaluated_rrf(self):
        scores = [np.array([[.1,.4,.3],[.5,.2,.1]], dtype=np.float32),
                  np.array([[.8,0,.9],[.2,0,.3]], dtype=np.float32), np.zeros((2,3), dtype=np.float32)]
        present = np.array([[1,1,0], [1,0,0], [1,1,0]], dtype=bool)
        previous = dict(score_methods(scores, present))["rrf_80_20_00"]
        np.testing.assert_allclose(rank_fusion(scores, present), previous)

    def fixture(self, root):
        index = root / "index"
        items = [dict(item_id=f"i{i}", kind="sticker", ocr_text="", caption_text="caption") for i in range(2)]
        write_jsonl(index / "items.jsonl", items)
        write_json(index / "metadata.json", dict(item_ids=["i0", "i1"], text_model="fixture", text_revision="r"))
        image = np.zeros((2,512), dtype=np.float32)
        image[0,0] = 1
        image[1,1] = 1
        caption = np.zeros_like(image)
        caption[0,:2] = [1,0]
        caption[1,:2] = [.8,.6]
        np.savez_compressed(index / "vectors.npz", image=image, caption=caption, ocr=np.zeros_like(image))
        mean = np.zeros(512, dtype=np.float32)
        mean[0] = np.sqrt(.99)
        calibration = root / "calibration"
        write_json(calibration / "metadata.json", dict(index_sha256=sha256(index / "vectors.npz"), text_model="fixture", text_revision="r"))
        np.save(calibration / "query_mean.npy", mean)
        return index, calibration, mean

    def test_background_offset_removes_synthetic_caption_hub(self):
        with tempfile.TemporaryDirectory() as directory:
            index, calibration, mean = self.fixture(Path(directory))
            search = SearchIndex(index, calibration=calibration)
            q = mean[None,:].copy()
            q[0,1] = .1
            self.assertEqual(search.topk(search.scores(q, 'caption')[0], 1)[0][0]['item_id'], 'i0')
            corrected = search.scores(q, 'caption_calibrated')
            np.testing.assert_allclose(corrected, (q-mean) @ search.caption.T, atol=1e-7)
            self.assertEqual(search.topk(corrected[0], 1)[0][0]['item_id'], 'i1')
            self.assertTrue(np.isfinite(search.scores(q, 'rrf_calibrated_ocr')).all())

    def test_calibration_is_required_and_index_hash_is_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            index, calibration, mean = self.fixture(Path(directory))
            search = SearchIndex(index)
            with self.assertRaisesRegex(ValueError, 'TRAIN-fitted'):
                search.scores(mean[None,:], 'rrf_calibrated')
            path = calibration / 'metadata.json'
            meta = json.loads(path.read_text())
            write_json(path, meta | dict(index_sha256='different'))
            with self.assertRaisesRegex(ValueError, 'different index'):
                SearchIndex(index, calibration=calibration)

    def test_topk_does_not_return_absent_branch_items(self):
        with tempfile.TemporaryDirectory() as directory:
            index, calibration, mean = self.fixture(Path(directory))
            search = SearchIndex(index)
            self.assertEqual(search.topk(np.full(2, -np.inf)), [])


if __name__ == '__main__':
    unittest.main()
