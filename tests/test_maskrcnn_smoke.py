"""CPU smoke test for tools/maskrcnn_train.py.

The subprocess is the hang check: it has to train one step, evaluate, predict,
and return. A stalled DataLoader or an unclosed pool fails the timeout.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from maskrcnn_train import MetricAccum, average_precision, default_workers  # noqa: E402


class MetricTests(unittest.TestCase):
    def test_perfect_pore_match(self):
        gt = np.zeros((16, 16), dtype=bool)
        gt[2:8, 2:8] = True
        accum = MetricAccum(2)
        accum.add({0: [], 1: [gt]}, {0: [], 1: [(gt.copy(), 0.9)]})
        metrics = accum.to_dict(["keyhole", "pore"])
        self.assertAlmostEqual(metrics["pore_AP50"], 1.0)
        self.assertAlmostEqual(metrics["pore_AP50_95"], 1.0)
        self.assertAlmostEqual(metrics["pore_recall50"], 1.0)
        self.assertEqual(metrics["pore_gt"], 1)

    def test_missed_pore_is_zero(self):
        gt = np.zeros((16, 16), dtype=bool)
        gt[2:8, 2:8] = True
        accum = MetricAccum(2)
        accum.add({1: [gt]}, {1: []})
        metrics = accum.to_dict(["keyhole", "pore"])
        self.assertEqual(metrics["pore_AP50"], 0.0)
        self.assertEqual(metrics["pore_AP50_95"], 0.0)
        self.assertEqual(metrics["pore_recall50"], 0.0)

    def test_false_positive_then_true_positive(self):
        # High-score miss, then a lower-score hit. Interpolated AP is 0.5.
        flags = [False, True]
        scores = [0.9, 0.4]
        self.assertAlmostEqual(average_precision(flags, scores, n_gt=1), 0.5, places=5)

    def test_windows_workers_default_to_zero(self):
        self.assertEqual(default_workers(None, "nt"), 0)
        self.assertEqual(default_workers(4, "nt"), 4)
        self.assertGreaterEqual(default_workers(None, "posix"), 0)


class WatchdogTest(unittest.TestCase):
    def test_stall_exits_instead_of_hanging(self):
        code = textwrap.dedent(
            """
            import time
            from maskrcnn_train import StallWatchdog
            StallWatchdog(0.4)
            time.sleep(5)
            print("NO_STALL")
            """
        )
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT / "tools")
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(proc.returncode, 3, proc.stdout + proc.stderr)
        self.assertNotIn("NO_STALL", proc.stdout)


class SmokeCliTest(unittest.TestCase):
    def test_cpu_smoke_exits(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc = subprocess.run(
                [sys.executable, str(ROOT / "tools" / "maskrcnn_train.py"), "smoke", "--out", tmp],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=600,
            )
            sys.stdout.write(proc.stdout[-4000:])
            sys.stderr.write(proc.stderr[-4000:])
            self.assertEqual(proc.returncode, 0, proc.stderr[-4000:] + proc.stdout[-4000:])
            self.assertIn("SMOKE_OK", proc.stdout)
            self.assertTrue((Path(tmp) / "run" / "last.pt").is_file())
            metrics = json.loads((Path(tmp) / "eval" / "metrics.json").read_text(encoding="utf-8"))
            for key in ("pore_AP50", "pore_AP50_95", "pore_recall50"):
                self.assertIn(key, metrics)
            labels = list((Path(tmp) / "predictions").rglob("*.txt"))
            self.assertGreaterEqual(len(labels), 2)
            from seg_io import parse_yolo_line

            for path in labels:
                lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
                ids_path = path.parents[2] / "ids" / path.parent.name / f"{path.stem}.json"
                sidecar = json.loads(ids_path.read_text(encoding="utf-8"))
                self.assertEqual(len(sidecar), len(lines))
                for line, meta in zip(lines, sidecar):
                    parsed = parse_yolo_line(line, 64, 64)
                    self.assertIsNotNone(parsed)
                    self.assertIsNotNone(parsed[2])
                    self.assertEqual(meta["src"], "maskrcnn")


if __name__ == "__main__":
    unittest.main()
