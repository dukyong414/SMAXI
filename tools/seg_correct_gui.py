"""Segmentation correction GUI for keyhole / pore YOLO-seg labels.

Launch from the repository root (Windows or Linux)::

    python tools/seg_correct_gui.py --root <dataset_root> --out <output_root>

The input tree is only read. Corrected YOLO-seg polygons and ids sidecars
are written under ``--out``, mirroring ``labels/<split>`` and ``ids/<split>``,
with ``src`` set to ``manual``. A per-frame ``edit_log.csv`` and a
``reviewed.csv`` progress file live in that same output root.

SAM click-to-segment is optional. It turns on when ``segment_anything`` imports
and ``--sam-checkpoint`` points at a vit_b (or vit_l / vit_h) weight file.
Otherwise the control stays disabled and the rest of the editor still runs.

``--self-test`` runs the headless IO checks and does not open a window.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Correct keyhole and pore YOLO-seg labels without overwriting the input dataset."
    )
    parser.add_argument("--root", type=Path, help="Dataset root containing images/, labels/, and ids/.")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output root. Defaults to <root>_corrected beside the dataset. Never the input root.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load labels from --out when a corrected frame already exists (default: on).",
    )
    parser.add_argument("--audit", type=Path, default=None, help="Optional CSV of frames to jump to.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional manifest.csv (split,cine,track,name,frame,...). Defaults to <root>/manifest.csv when present.",
    )
    parser.add_argument("--sam-checkpoint", type=Path, default=None, help="Optional SAM vit_b .pth checkpoint.")
    parser.add_argument("--device", default=None, help="Device for SAM (cpu or cuda). Default: cuda when available.")
    parser.add_argument("--self-test", action="store_true", help="Run headless IO checks and exit.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test:
        from seg_io import run_self_test

        run_self_test()
        print("SELF_TEST_OK")
        return 0
    if args.root is None:
        build_parser().error("--root is required")
    try:
        from seg_correct_app import launch
    except ImportError as exc:
        print(
            "The correction GUI needs PySide6 and OpenCV.\n"
            "  pip install -r tools/requirements.txt\n"
            f"Import failed: {exc}",
            file=sys.stderr,
        )
        return 1
    return launch(args)


if __name__ == "__main__":
    raise SystemExit(main())
