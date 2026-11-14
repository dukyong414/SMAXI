"""Open the correction window offscreen, edit one polygon, and save it."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import seg_io  # noqa: E402


class GuiOffscreenTest(unittest.TestCase):
    def test_edit_save_does_not_touch_input(self):
        try:
            from PySide6.QtWidgets import QApplication
            from seg_correct_app import MainWindow
        except ImportError as exc:
            self.skipTest(f"PySide6 is not installed: {exc}")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "dataset"
            out = Path(tmp) / "corrected"
            split = "val"
            stem = "Test1055_T2_00243"
            image = __import__("numpy").zeros((48, 64, 3), dtype="uint8")
            image[:] = 30
            seg_io.write_png(root / "images" / split / f"{stem}.png", image)
            label = root / "labels" / split / f"{stem}.txt"
            ids = root / "ids" / split / f"{stem}.json"
            label.parent.mkdir(parents=True, exist_ok=True)
            ids.parent.mkdir(parents=True, exist_ok=True)
            label.write_text(
                "1 0.20 0.20 0.40 0.20 0.40 0.50 0.20 0.50\n",
                encoding="utf-8",
            )
            ids.write_text(json.dumps([{"cls": 1, "id": 3, "src": "blob"}]), encoding="utf-8")
            before_label = label.read_bytes()
            before_ids = ids.read_bytes()

            app = QApplication.instance() or QApplication([])
            window = MainWindow(root, out, True, None, None, None, "cpu")
            try:
                self.assertFalse(window.sam.available)
                self.assertEqual(len(window.instances), 1)
                self.assertEqual(window.instances[0].track_id, 3)
                window.set_mode("polygon")
                window.poly_pts = [(8, 8), (18, 8), (18, 20)]
                window._close_polygon()
                self.assertTrue(window.dirty)
                self.assertEqual(len(window.instances), 2)
                window.save_current()
                self.assertFalse(window.dirty)
                saved_ids = json.loads((out / "ids" / split / f"{stem}.json").read_text(encoding="utf-8"))
                self.assertEqual(len(saved_ids), 2)
                self.assertTrue(all(row["src"] == "manual" for row in saved_ids))
                self.assertEqual(label.read_bytes(), before_label)
                self.assertEqual(ids.read_bytes(), before_ids)
                self.assertIn("save", (out / "edit_log.csv").read_text(encoding="utf-8"))
                self.assertTrue((out / "reviewed.csv").is_file())
                window.do_undo()
                self.assertEqual(len(window.instances), 1)
            finally:
                window.dirty = False
                window.close()
                app.processEvents()


if __name__ == "__main__":
    unittest.main()
