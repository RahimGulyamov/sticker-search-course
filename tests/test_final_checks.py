"""Metadata and resume guards, tested without inference or GPU dependencies."""
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from sticker_search.common import sha256, write_json, write_jsonl
from sticker_search.finalize_run import freeze, generation_success


class FinalChecksTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.previous_cwd = Path.cwd()
        os.chdir(self.root)
        self.addCleanup(os.chdir, self.previous_cwd)
        self.output = self.root / 'runs/final_v1'
        self.checkpoint = Path('runs/full_v2/adapter/best.pt')
        self.checkpoint.parent.mkdir(parents=True)
        self.checkpoint.write_bytes(b'fixture checkpoint bytes')
        index = Path('runs/full_v2/index')
        index.mkdir()
        (index / 'vectors.npz').write_bytes(b'fixture index bytes')
        write_jsonl(index / 'items.jsonl', [dict(item_id='i0')])
        for lang in ['en', 'ru']:
            directory = Path(f'runs/full_v2/queries_{lang}')
            write_jsonl(directory / 'queries.jsonl', [dict(query_id='q', split='test')])
            (directory / 'queries.npy').write_bytes(b'fixture query bytes')
        write_json(self.root / 'configs/final_selection_v1.json', dict(checkpoint=str(self.checkpoint), expected_epoch=4))
        (self.root / 'sticker_search').mkdir()
        for name in ['evaluate', 'retrieval', 'model', 'metrics', 'fusion']:
            (self.root / 'sticker_search' / f'{name}.py').write_text('# fixture\n')
        self.saved = dict(epoch=4, index_sha256=sha256(index / 'vectors.npz'))
        self.torch = SimpleNamespace(load=lambda *a, **k:self.saved,
            cuda=SimpleNamespace(device_count=lambda:0), version=SimpleNamespace(cuda=None))

    def call_freeze(self):
        with patch('sticker_search.finalize_run.ROOT', self.root), patch.dict('sys.modules', {'torch': self.torch}):
            return freeze(self.output)

    def test_freeze_precedes_test_and_reuses_identical_selection(self):
        self.call_freeze()
        saved = (self.output / 'frozen_selection.json').read_bytes()
        self.assertFalse((self.output / 'test_en/metrics.json').exists())
        self.call_freeze()
        self.assertEqual(saved, (self.output / 'frozen_selection.json').read_bytes())
        self.checkpoint.write_bytes(b'changed fixture checkpoint')
        with self.assertRaisesRegex(ValueError, 'Frozen artifacts changed'):
            self.call_freeze()
        self.assertEqual(saved, (self.output / 'frozen_selection.json').read_bytes())

    def test_wrong_epoch_and_index_are_rejected(self):
        self.saved['epoch'] = 10
        with self.assertRaisesRegex(ValueError, 'checkpoint epoch'):
            self.call_freeze()
        self.saved['epoch'] = 4
        self.saved['index_sha256'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'Checkpoint/index mismatch'):
            self.call_freeze()

    def test_existing_test_without_selection_is_rejected(self):
        write_json(self.output / 'test_en/metrics.json', {})
        with self.assertRaisesRegex(ValueError, 'without a frozen selection'):
            self.call_freeze()

    def test_generation_needs_nonconstant_mask_and_matching_output_hash(self):
        directory = self.root / 'generation'
        self.assertFalse(generation_success(directory))
        directory.mkdir()
        (directory / 'sticker.png').write_bytes(b'fixture image bytes')
        metadata = dict(alpha_range=[0,255], output_sha256=sha256(directory / 'sticker.png'))
        write_json(directory / 'generation.json', metadata)
        self.assertTrue(generation_success(directory))
        write_json(directory / 'generation.json', metadata | dict(alpha_range=[255,255]))
        self.assertFalse(generation_success(directory))
        write_json(directory / 'generation.json', metadata | dict(output_sha256='wrong'))
        self.assertFalse(generation_success(directory))


if __name__ == '__main__':
    unittest.main()
