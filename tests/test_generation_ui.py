"""Real Streamlit widget lifecycle, with generation and segmentation mocked."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image
try:
    from streamlit.testing.v1 import AppTest
except ModuleNotFoundError:
    AppTest = None


@unittest.skipIf(AppTest is None, "Install the demo extra for Streamlit UI checks")
class GenerationUITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.original = self.root/"original.png"
        self.cutout = self.root/"sticker.png"
        Image.new("RGB", (8, 8), "white").save(self.original)
        image = Image.new("RGBA", (8, 8), (255, 0, 0, 0))
        image.putpixel((4, 4), (255, 0, 0, 255))
        image.save(self.cutout)
        self.env = patch.dict(os.environ, STICKER_ENABLE_GENERATION="1")
        self.env.start()
        self.addCleanup(self.env.stop)

    def app(self):
        return AppTest.from_string(
            "from sticker_search.generation_ui import render_generation_panel\n"
            "render_generation_panel()\n"
        ).run(timeout=15)

    def test_two_buttons_persist_outputs_without_regenerating(self):
        with patch("sticker_search.generate.generate_image", return_value=self.original) as gen, \
             patch("sticker_search.remove_background.remove_background", return_value=self.cutout) as remove:
            app = self.app()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.button), 1)
            app.button(key="generate_image").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(gen.call_count, 1)
            remove.assert_not_called()
            self.assertEqual(len(app.get("imgs")), 1)
            app.run()  # Equivalent full rerun must preserve the original.
            self.assertEqual(len(app.get("imgs")), 1)
            self.assertEqual(gen.call_count, 1)
            app.button(key="remove_background").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.get("imgs")), 2)
            self.assertEqual(gen.call_count, 1)
            self.assertEqual(remove.call_count, 1)
            app.run()
            self.assertEqual(len(app.get("imgs")), 2)
            self.assertEqual(gen.call_count, 1)
            self.assertEqual(remove.call_count, 1)
            app.button(key="generate_image").click().run()
            self.assertEqual(len(app.get("imgs")), 1)
            self.assertIsNone(app.session_state["generation_result"]["cutout"])
            self.assertEqual(gen.call_count, 2)
            self.assertEqual(remove.call_count, 1)

    def test_cutout_failure_can_retry_without_generation(self):
        with patch("sticker_search.generate.generate_image", return_value=self.original) as gen, \
             patch("sticker_search.remove_background.remove_background",
                   side_effect=[RuntimeError("bad mask"), self.cutout]) as remove:
            app = self.app()
            app.button(key="generate_image").click().run()
            app.button(key="remove_background").click().run()
            self.assertFalse(app.exception)
            self.assertIn("bad mask", app.error[0].value)
            self.assertEqual(len(app.get("imgs")), 1)
            app.button(key="remove_background").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.get("imgs")), 2)
            self.assertEqual(gen.call_count, 1)
            self.assertEqual(remove.call_count, 2)

    def test_empty_prompt_disables_generation(self):
        app = self.app()
        app.text_input(key="generation_prompt").set_value(" ").run()
        self.assertTrue(app.button(key="generate_image").disabled)


if __name__ == "__main__":
    unittest.main()
