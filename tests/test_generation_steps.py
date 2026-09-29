"""Dependency-light contracts: stage separation, retry and legacy metadata.

Inference is mocked; these checks do not measure image or mask quality.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from PIL import Image

from sticker_search.common import sha256
from sticker_search.generate import generate, generate_image
from sticker_search.remove_background import remove_background


class GenerationStepsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.torch = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: True, empty_cache=Mock()),
            bfloat16="bf16", Generator=lambda **kwargs: SimpleNamespace(manual_seed=lambda seed: seed),
        )
        self.pipe = Mock(return_value=SimpleNamespace(images=[Image.new("RGB", (8, 8), "white")]))
        self.pipe.to.return_value = self.pipe
        self.diffusers = SimpleNamespace(FluxPipeline=SimpleNamespace(from_pretrained=Mock(return_value=self.pipe)))
        self.mask = Image.new("RGBA", (8, 8), (255, 0, 0, 0))
        self.mask.putpixel((4, 4), (255, 0, 0, 255))
        self.rembg = SimpleNamespace(new_session=Mock(return_value="cpu-session"),
                                     remove=Mock(return_value=self.mask))

    def test_generation_does_not_import_or_call_rembg(self):
        with patch.dict(sys.modules, torch=self.torch, diffusers=self.diffusers, rembg=None):
            original = generate_image("a cat", self.root/"generated", english=True)
        self.assertTrue(original.is_file())
        self.assertFalse(original.with_name("sticker.png").exists())
        self.assertFalse(original.with_name("generation.json").exists())
        meta = json.loads(original.with_name("image_generation.json").read_text())
        self.assertEqual(meta["output_sha256"], sha256(original))
        self.assertEqual(self.pipe.call_count, 1)

    def test_background_removal_needs_neither_torch_nor_diffusers(self):
        original = self.root/"source.png"
        Image.new("RGB", (8, 8), "white").save(original)
        before = original.read_bytes()
        with patch.dict(sys.modules, torch=None, diffusers=None, rembg=self.rembg):
            cutout = remove_background(original, self.root/"result.png")
        self.assertEqual(original.read_bytes(), before)
        with Image.open(cutout) as image:
            self.assertEqual(image.mode, "RGBA")
            self.assertEqual(image.getchannel("A").getextrema(), (0, 255))
        self.rembg.new_session.assert_called_once_with("u2net", providers=["CPUExecutionProvider"])

    def test_failed_cutout_preserves_original_and_can_retry(self):
        original = self.root/"source.png"
        Image.new("RGB", (8, 8), "white").save(original)
        before = original.read_bytes()
        self.rembg.remove.side_effect = [Image.new("RGBA", (8, 8), "white"), self.mask]
        with patch.dict(sys.modules, torch=None, diffusers=None, rembg=self.rembg):
            with self.assertRaisesRegex(RuntimeError, "constant mask"):
                remove_background(original, self.root/"result.png")
            meta = json.loads((self.root/"result.background.json").read_text())
            self.assertFalse(meta["valid_mask"])
            result = remove_background(original, self.root/"result.png")
        self.assertTrue(result.is_file())
        self.assertEqual(original.read_bytes(), before)
        self.assertEqual(self.rembg.remove.call_count, 2)

    def test_original_cannot_be_overwritten(self):
        with self.assertRaisesRegex(ValueError, "separate output"):
            remove_background(self.root/"x.png", self.root/"x.png")

    def test_combined_wrapper_preserves_final_checks_schema(self):
        with patch.dict(sys.modules, torch=self.torch, diffusers=self.diffusers, rembg=self.rembg):
            result = generate("a cat", self.root/"combined", english=True)
        meta = json.loads(result.with_name("generation.json").read_text())
        self.assertEqual(meta["alpha_range"], [0, 255])
        self.assertEqual(meta["output_sha256"], sha256(result))
        self.assertEqual(meta["background_remover"], "rembg/u2net")
        self.assertIn("background_weights_sha256", meta)
        self.assertEqual(self.pipe.call_count, 1)
        self.assertEqual(self.rembg.remove.call_count, 1)


if __name__ == "__main__":
    unittest.main()
