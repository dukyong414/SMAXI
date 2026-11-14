"""Headless checks for the segmentation corrector (no display, no torch)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import seg_io  # noqa: E402


class SegIoTest(unittest.TestCase):
    def test_self_test(self):
        seg_io.run_self_test()

    def test_score_is_not_a_vertex(self):
        parsed = seg_io.parse_yolo_line("1 0.10 0.10 0.20 0.10 0.20 0.20 0.80", 100, 100)
        self.assertIsNotNone(parsed)
        cls, poly, score = parsed
        self.assertEqual(cls, 1)
        self.assertEqual(len(poly), 3)
        self.assertAlmostEqual(score, 0.80)
        line = seg_io.format_yolo_line(1, poly / [100, 100], score)
        again = seg_io.parse_yolo_line(line, 100, 100)
        self.assertAlmostEqual(again[2], 0.80)
        self.assertEqual(len(again[1]), 3)


if __name__ == "__main__":
    unittest.main()
