"""Dataset IO and mask editing for the keyhole/pore segmentation corrector.

Label files are YOLO-seg polygons, one instance per line:

    cls x1 y1 x2 y2 ... xn yn

Coordinates are normalized by image width and height. A confidence may be
appended (Ultralytics ``save_conf`` style). An odd count of floats after the
class is read as that score; training labels omit it.

The ids sidecar is a JSON list aligned with the label lines:

    {"cls": 0, "id": null, "src": "sam_prop"}

``id`` is a persistent pore track id, or null for a keyhole. Corrections are
written with ``src`` set to ``manual`` and are never written back onto the
input paths.
"""

from __future__ import annotations

import csv
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

KEYHOLE = 0
PORE = 1
CLASS_NAMES = {KEYHOLE: "keyhole", PORE: "pore"}
CLASS_COLORS = {
    KEYHOLE: (0, 200, 255),
    PORE: (255, 70, 170),
}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
_STEM_RE = re.compile(r"^(?P<cine>.+)_T(?P<track>\d+)_(?P<frame>\d+)$")
_FALSE_FLAGS = {"0", "false", "no", "n", "ok", "clean", "pass"}
_HEADER_ALIASES = {
    "frame": "frame",
    "frame_idx": "frame",
    "frame_index": "frame",
    "idx": "frame",
    "name": "name",
    "stem": "name",
    "track_name": "name",
    "cine": "cine",
    "video": "cine",
    "track": "track",
    "split": "split",
    "path": "path",
    "file": "path",
    "image": "path",
    "image_path": "path",
    "reason": "reason",
    "note": "reason",
    "flag_reason": "reason",
    "flag": "flag",
    "flagged": "flag",
    "error": "flag",
    "bad": "flag",
    "laser_on": "laser_on",
    "kh": "kh",
    "n_pores": "n_pores",
    "amb": "amb",
    "neg": "neg",
}


@dataclass
class Instance:
    """One keyhole or pore polygon in pixel coordinates."""

    cls: int
    poly: np.ndarray
    track_id: int | None = None
    src: str = "yolo"
    score: float | None = None

    def cloned(self, src: str | None = None) -> "Instance":
        return Instance(
            cls=int(self.cls),
            poly=np.array(self.poly, dtype=np.float64, copy=True),
            track_id=None if self.track_id is None else int(self.track_id),
            src=self.src if src is None else src,
            score=self.score,
        )


@dataclass
class FrameRec:
    split: str
    cine: str
    track: str
    name: str
    frame: int
    stem: str
    image_path: Path
    meta: dict


def atomic_write_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _norm_header(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def read_csv_dicts(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return rows
        mapping = {
            header: _HEADER_ALIASES.get(_norm_header(header), _norm_header(header))
            for header in reader.fieldnames
            if header is not None
        }
        for raw in reader:
            row: dict[str, str] = {}
            for key, value in raw.items():
                if key is None:
                    continue
                row[mapping.get(key, key)] = "" if value is None else str(value).strip()
            if any(row.values()):
                rows.append(row)
    return rows


def parse_stem(stem: str) -> tuple[str, str, str, int]:
    match = _STEM_RE.match(stem)
    if match:
        cine = match.group("cine")
        track = match.group("track")
        frame = int(match.group("frame"))
        return cine, track, f"{cine}_T{track}", frame
    fallback = re.match(r"^(?P<name>.+?)_(?P<frame>\d+)$", stem)
    if fallback:
        name = fallback.group("name")
        frame = int(fallback.group("frame"))
        return name, "0", name, frame
    return stem, "0", stem, 0


def _track_sort_key(track: str):
    try:
        return (0, int(track))
    except ValueError:
        return (1, track)


def annotation_path(image_path: Path, kind: str, suffix: str) -> Path:
    """Map ``images/<split>/name.png`` to ``<kind>/<split>/name<suffix>``."""
    image_path = Path(image_path)
    parts = list(image_path.parts)
    if "images" in parts:
        index = len(parts) - 1 - parts[::-1].index("images")
        parts[index] = kind
        return Path(*parts).with_suffix(suffix)
    return image_path.parent.parent / kind / image_path.parent.name / f"{image_path.stem}{suffix}"


def input_label_path(image_path: Path) -> Path:
    return annotation_path(image_path, "labels", ".txt")


def input_ids_path(image_path: Path) -> Path:
    return annotation_path(image_path, "ids", ".json")


def assert_distinct_roots(in_root: Path, out_root: Path) -> None:
    if Path(in_root).resolve() == Path(out_root).resolve():
        raise SystemExit(
            f"Refusing to use {Path(in_root).resolve()} as both the input and the output. "
            "Pass a separate --out directory so the original labels stay untouched."
        )


def _same_file(left: Path, right: Path) -> bool:
    try:
        return Path(left).resolve() == Path(right).resolve()
    except OSError:
        return Path(left) == Path(right)


def discover_frames(root: Path, manifest_path: Path | None = None) -> list[FrameRec]:
    root = Path(root)
    images = root / "images"
    if not images.is_dir():
        raise FileNotFoundError(
            f"No images/ directory under {root}. Expected images/{{train,val}}/<Test####_T#>_<frame>.png."
        )
    manifest: dict[tuple[str, str, int], dict] = {}
    manifest_file = Path(manifest_path) if manifest_path else root / "manifest.csv"
    if manifest_file.is_file():
        for row in read_csv_dicts(manifest_file):
            frame_text = row.get("frame") or ""
            name = row.get("name") or ""
            if not name or not frame_text:
                continue
            try:
                frame_no = int(float(frame_text))
            except ValueError:
                continue
            split = row.get("split") or ""
            manifest[(split, name, frame_no)] = row
            manifest.setdefault(("", name, frame_no), row)

    frames: list[FrameRec] = []
    for split_dir in sorted(path for path in images.iterdir() if path.is_dir()):
        files = [path for path in split_dir.iterdir() if path.suffix.lower() in IMAGE_EXTS]
        for image_path in sorted(files, key=lambda path: path.name):
            cine, track, name, frame_no = parse_stem(image_path.stem)
            meta = manifest.get((split_dir.name, name, frame_no)) or manifest.get(("", name, frame_no)) or {}
            if meta.get("cine"):
                cine = meta["cine"]
            if meta.get("track"):
                track = str(meta["track"])
            if meta.get("name"):
                name = meta["name"]
            frames.append(
                FrameRec(
                    split=split_dir.name,
                    cine=str(cine),
                    track=str(track),
                    name=str(name),
                    frame=int(frame_no),
                    stem=image_path.stem,
                    image_path=image_path,
                    meta=dict(meta),
                )
            )
    frames.sort(key=lambda item: (item.split, item.cine, _track_sort_key(item.track), item.frame, item.stem))
    return frames


def load_audit(path: Path | None) -> list[dict]:
    if path is None or not Path(path).is_file():
        return []
    return read_csv_dicts(path)


def _row_is_flagged(row: dict) -> bool:
    flag = (row.get("flag") or "").strip().lower()
    if flag == "":
        return True
    return flag not in _FALSE_FLAGS


def audit_hit(frame: FrameRec, rows: list[dict]) -> dict | None:
    for row in rows:
        if not _row_is_flagged(row):
            continue
        path_text = row.get("path") or ""
        if path_text:
            candidate = Path(path_text)
            if candidate.stem == frame.stem or candidate.name == frame.image_path.name:
                return row
        frame_text = row.get("frame") or ""
        try:
            frame_no = int(float(frame_text)) if frame_text else None
        except ValueError:
            frame_no = None
        name = row.get("name") or ""
        if name and frame_no is not None and name == frame.name and frame_no == frame.frame:
            if not row.get("split") or row["split"] == frame.split:
                return row
        cine = row.get("cine") or ""
        track = row.get("track") or ""
        if cine and track and frame_no is not None and frame_no == frame.frame and cine == frame.cine:
            try:
                same_track = int(track) == int(frame.track)
            except ValueError:
                same_track = track == frame.track
            if same_track:
                return row
    return None


def parse_yolo_line(line: str, width: int, height: int):
    tokens = line.split()
    if len(tokens) < 7:
        return None
    try:
        cls = int(float(tokens[0]))
        values = [float(token) for token in tokens[1:]]
    except ValueError:
        return None
    score = None
    if len(values) % 2 == 1:
        score = values.pop()
    if len(values) < 6 or len(values) % 2:
        return None
    poly = np.asarray(values, dtype=np.float64).reshape(-1, 2)
    poly[:, 0] *= float(width)
    poly[:, 1] *= float(height)
    return cls, poly, score


def format_yolo_line(cls: int, poly_norm: np.ndarray, score: float | None = None) -> str:
    parts = [str(int(cls))]
    for x_coord, y_coord in np.asarray(poly_norm, dtype=np.float64).reshape(-1, 2):
        parts.append(f"{float(x_coord):.6f}")
        parts.append(f"{float(y_coord):.6f}")
    if score is not None:
        parts.append(f"{float(score):.6f}")
    return " ".join(parts)


def read_instances(label_path: Path | None, ids_path: Path | None, width: int, height: int) -> list[Instance]:
    if label_path is None or not Path(label_path).is_file():
        return []
    parsed = []
    for line in Path(label_path).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        item = parse_yolo_line(stripped, width, height)
        if item is not None:
            parsed.append(item)
    sidecar: list = []
    if ids_path is not None and Path(ids_path).is_file():
        try:
            raw = json.loads(Path(ids_path).read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            raw = []
        if isinstance(raw, list):
            sidecar = raw
    instances: list[Instance] = []
    for index, (cls, poly, score) in enumerate(parsed):
        meta = sidecar[index] if index < len(sidecar) and isinstance(sidecar[index], dict) else {}
        track_id = meta.get("id", None)
        if track_id is not None:
            try:
                track_id = int(track_id)
            except (TypeError, ValueError):
                track_id = None
        src = str(meta.get("src") or "yolo")
        if score is None and meta.get("score") is not None:
            try:
                score = float(meta["score"])
            except (TypeError, ValueError):
                score = None
        instances.append(Instance(cls=int(cls), poly=poly, track_id=track_id, src=src, score=score))
    return instances


def format_label_text(instances: list[Instance], width: int, height: int, with_score: bool = False):
    lines = []
    sidecar = []
    width = max(int(width), 1)
    height = max(int(height), 1)
    for inst in instances:
        if inst.poly is None or len(inst.poly) < 3:
            continue
        poly = np.asarray(inst.poly, dtype=np.float64).copy()
        poly[:, 0] = np.clip(poly[:, 0] / width, 0.0, 1.0)
        poly[:, 1] = np.clip(poly[:, 1] / height, 0.0, 1.0)
        score = float(inst.score) if with_score and inst.score is not None else None
        lines.append(format_yolo_line(inst.cls, poly, score))
        entry = {
            "cls": int(inst.cls),
            "id": None if inst.track_id is None else int(inst.track_id),
            "src": inst.src,
        }
        if with_score and inst.score is not None:
            entry["score"] = float(inst.score)
        sidecar.append(entry)
    text = ("\n".join(lines) + "\n") if lines else ""
    return text, sidecar


def write_label_pair(
    label_path: Path,
    ids_path: Path,
    instances: list[Instance],
    width: int,
    height: int,
    src: str | None = None,
    with_score: bool = False,
) -> list[Instance]:
    rows: list[Instance] = []
    for inst in instances:
        item = inst.cloned(src=src)
        if item.cls == KEYHOLE:
            item.track_id = None
        if not with_score:
            item.score = None
        if item.poly is None or len(item.poly) < 3:
            continue
        rows.append(item)
    text, sidecar = format_label_text(rows, width, height, with_score=with_score)
    atomic_write_text(Path(label_path), text)
    atomic_write_text(Path(ids_path), json.dumps(sidecar, indent=2) + "\n")
    return rows


def corrected_paths(out_root: Path, frame: FrameRec) -> tuple[Path, Path]:
    label_path = Path(out_root) / "labels" / frame.split / f"{frame.stem}.txt"
    ids_path = Path(out_root) / "ids" / frame.split / f"{frame.stem}.json"
    return label_path, ids_path


def load_frame_instances(frame: FrameRec, out_root: Path, resume: bool, width: int, height: int):
    label_path, ids_path = corrected_paths(out_root, frame)
    if resume and label_path.is_file():
        return read_instances(label_path, ids_path if ids_path.is_file() else None, width, height), True
    in_label = input_label_path(frame.image_path)
    in_ids = input_ids_path(frame.image_path)
    return read_instances(in_label, in_ids if in_ids.is_file() else None, width, height), False


def _refuse_if_input(dest: Path, frame: FrameRec) -> None:
    protected = (
        frame.image_path,
        input_label_path(frame.image_path),
        input_ids_path(frame.image_path),
    )
    for src in protected:
        if _same_file(dest, src):
            raise SystemExit(f"Refusing to overwrite input file {src}")


def append_edit_log(path: Path, frame: FrameRec, action: str, detail: str, n_instances: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists() or path.stat().st_size == 0
    with open(path, "a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        if new_file:
            writer.writerow(
                ["time", "split", "cine", "track", "name", "frame", "action", "n_instances", "detail"]
            )
        writer.writerow(
            [
                datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                frame.split,
                frame.cine,
                frame.track,
                frame.name,
                frame.frame,
                action,
                int(n_instances),
                detail,
            ]
        )
        handle.flush()
        os.fsync(handle.fileno())


def save_correction(
    in_root: Path,
    out_root: Path,
    frame: FrameRec,
    instances: list[Instance],
    width: int,
    height: int,
    action: str = "save",
    detail: str = "",
) -> tuple[Path, Path]:
    assert_distinct_roots(in_root, out_root)
    label_path, ids_path = corrected_paths(out_root, frame)
    _refuse_if_input(label_path, frame)
    _refuse_if_input(ids_path, frame)
    written = write_label_pair(label_path, ids_path, instances, width, height, src="manual", with_score=False)
    append_edit_log(Path(out_root) / "edit_log.csv", frame, action, detail, len(written))
    return label_path, ids_path


class ReviewStore:
    """Frame-level reviewed flags stored as ``reviewed.csv`` under the output root."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._rows: dict[tuple[str, str, int], bool] = {}
        self.load()

    def load(self) -> None:
        self._rows.clear()
        if not self.path.is_file():
            return
        for row in read_csv_dicts(self.path):
            try:
                frame_no = int(float(row.get("frame") or ""))
            except ValueError:
                continue
            key = (row.get("split") or "", row.get("name") or "", frame_no)
            self._rows[key] = (row.get("reviewed") or "").strip().lower() in {"1", "true", "yes", "y"}

    def get(self, frame: FrameRec) -> bool:
        return self._rows.get((frame.split, frame.name, frame.frame), False)

    def set(self, frame: FrameRec, reviewed: bool) -> None:
        self._rows[(frame.split, frame.name, frame.frame)] = bool(reviewed)
        self.save()

    def count(self, frames: list[FrameRec]) -> tuple[int, int]:
        done = sum(1 for frame in frames if self.get(frame))
        return done, len(frames)

    def save(self) -> None:
        lines = ["split,name,frame,reviewed\n"]
        for (split, name, frame_no), reviewed in sorted(self._rows.items(), key=lambda item: (item[0][0], item[0][1], item[0][2])):
            lines.append(f"{split},{name},{frame_no},{1 if reviewed else 0}\n")
        atomic_write_text(self.path, "".join(lines))


def read_image_rgb(path: Path) -> np.ndarray:
    path = Path(path)
    buffer = np.fromfile(str(path), dtype=np.uint8)
    if buffer.size == 0:
        raise FileNotFoundError(path)
    image = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not decode image {path}")
    if image.dtype != np.uint8:
        peak = float(np.max(image)) if image.size else 0.0
        scale = 255.0 / peak if peak > 0 else 1.0
        image = np.clip(image.astype(np.float32) * scale, 0, 255).astype(np.uint8)
    if image.ndim == 2:
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    else:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(rgb)


def write_png(path: Path, rgb: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
    ok, buffer = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError(f"Could not encode {path}")
    buffer.tofile(str(path))


def display_gray(rgb: np.ndarray, channel: str = "mean") -> np.ndarray:
    if channel == "R":
        gray = rgb[:, :, 0]
    elif channel == "G":
        gray = rgb[:, :, 1]
    elif channel == "B":
        gray = rgb[:, :, 2]
    else:
        gray = rgb.mean(axis=2)
    return np.clip(gray, 0, 255).astype(np.uint8)


def apply_contrast_gamma(gray: np.ndarray, contrast: float, gamma: float) -> np.ndarray:
    values = gray.astype(np.float32) / 255.0
    values = np.clip((values - 0.5) * float(contrast) + 0.5, 0.0, 1.0)
    values = np.clip(np.power(values, max(float(gamma), 1e-3)), 0.0, 1.0)
    return (values * 255.0).astype(np.uint8)


def rasterize(poly: np.ndarray, height: int, width: int) -> np.ndarray:
    mask = np.zeros((int(height), int(width)), np.uint8)
    if poly is None or len(poly) < 3:
        return mask
    points = np.round(np.asarray(poly, dtype=np.float64)).astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [points], 1)
    return mask


def polygon_from_mask(mask: np.ndarray, epsilon: float = 0.8, min_area: float = 6) -> np.ndarray | None:
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    if int(binary.sum()) < 1:
        return None
    contours, _hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < min_area:
        return None
    approx = cv2.approxPolyDP(contour, float(epsilon), True)
    chosen = approx if len(approx) >= 3 else contour
    poly = chosen.reshape(-1, 2).astype(np.float64)
    if len(poly) < 3:
        return None
    return poly


def paint_disk(mask: np.ndarray, x_coord: float, y_coord: float, radius: int, value: int) -> np.ndarray:
    cv2.circle(
        mask,
        (int(round(x_coord)), int(round(y_coord))),
        max(int(radius), 1),
        int(value),
        thickness=-1,
    )
    return mask


def centroid(poly: np.ndarray) -> tuple[float, float]:
    points = np.asarray(poly, dtype=np.float64)
    return float(points[:, 0].mean()), float(points[:, 1].mean())


def _contour(poly: np.ndarray) -> np.ndarray:
    return np.round(np.asarray(poly, dtype=np.float64)).astype(np.int32).reshape(-1, 1, 2)


def hit_test(instances: list[Instance], x_coord: float, y_coord: float) -> int | None:
    hits = []
    for index, inst in enumerate(instances):
        if inst.poly is None or len(inst.poly) < 3:
            continue
        inside = cv2.pointPolygonTest(_contour(inst.poly), (float(x_coord), float(y_coord)), False)
        if inside >= 0:
            hits.append((abs(cv2.contourArea(_contour(inst.poly))), index))
    if not hits:
        return None
    hits.sort()
    return hits[0][1]


def nearest_vertex(inst: Instance, x_coord: float, y_coord: float, tolerance: float) -> int | None:
    points = np.asarray(inst.poly, dtype=np.float64)
    if len(points) == 0:
        return None
    distance = np.hypot(points[:, 0] - x_coord, points[:, 1] - y_coord)
    index = int(np.argmin(distance))
    if float(distance[index]) <= tolerance:
        return index
    return None


def nearest_edge_point(inst: Instance, x_coord: float, y_coord: float):
    points = np.asarray(inst.poly, dtype=np.float64)
    count = len(points)
    best = None
    best_distance = 1e18
    query = np.array([x_coord, y_coord], dtype=np.float64)
    for index in range(count):
        start = points[index]
        end = points[(index + 1) % count]
        edge = end - start
        length2 = float(np.dot(edge, edge))
        if length2 < 1e-12:
            projection = start
        else:
            tee = float(np.clip(np.dot(query - start, edge) / length2, 0.0, 1.0))
            projection = start + tee * edge
        distance = float(np.hypot(*(projection - query)))
        if distance < best_distance:
            best_distance = distance
            best = (index, projection, distance)
    return best


def insert_vertex(inst: Instance, edge_index: int, point) -> None:
    points = np.asarray(inst.poly, dtype=np.float64)
    inst.poly = np.insert(points, int(edge_index) + 1, np.asarray(point, dtype=np.float64).reshape(2), axis=0)


def delete_vertex(inst: Instance, index: int) -> bool:
    if len(inst.poly) <= 3:
        return False
    inst.poly = np.delete(np.asarray(inst.poly, dtype=np.float64), int(index), axis=0)
    return True


def move_vertex(inst: Instance, index: int, x_coord: float, y_coord: float, width: int, height: int) -> None:
    inst.poly[int(index), 0] = float(np.clip(x_coord, 0, max(width - 1, 0)))
    inst.poly[int(index), 1] = float(np.clip(y_coord, 0, max(height - 1, 0)))


def merge_instances(instances: list[Instance], height: int, width: int) -> Instance | None:
    if len(instances) < 2:
        return None
    mask = np.zeros((int(height), int(width)), np.uint8)
    for inst in instances:
        mask = np.maximum(mask, rasterize(inst.poly, height, width))
    poly = polygon_from_mask(mask, min_area=1)
    if poly is None:
        return None
    classes = [int(inst.cls) for inst in instances]
    chosen = max(set(classes), key=lambda value: (classes.count(value), -classes.index(value)))
    ids = [inst.track_id for inst in instances if inst.cls == PORE and inst.track_id is not None]
    track_id = None
    if chosen == PORE and ids and len(set(ids)) == 1:
        track_id = int(ids[0])
    return Instance(cls=chosen, poly=poly, track_id=track_id, src="manual")


def split_with_cut(
    inst: Instance,
    cut: list[tuple[float, float]],
    height: int,
    width: int,
    thickness: int = 3,
    min_area: float = 6,
) -> list[Instance]:
    mask = rasterize(inst.poly, height, width)
    if len(cut) >= 2:
        points = np.round(np.asarray(cut, dtype=np.float64)).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(mask, [points], isClosed=False, color=0, thickness=max(int(thickness), 1))
    return _components_to_instances(inst, mask, min_area)


def split_components(inst: Instance, height: int, width: int, min_area: float = 6) -> list[Instance]:
    mask = rasterize(inst.poly, height, width)
    return _components_to_instances(inst, mask, min_area)


def _components_to_instances(inst: Instance, mask: np.ndarray, min_area: float) -> list[Instance]:
    count, labels = cv2.connectedComponents((mask > 0).astype(np.uint8))
    pieces: list[Instance] = []
    for label in range(1, count):
        component = (labels == label).astype(np.uint8)
        if float(component.sum()) < min_area:
            continue
        poly = polygon_from_mask(component, min_area=min_area)
        if poly is None:
            continue
        track_id = inst.track_id if inst.cls == PORE and len(pieces) == 0 else None
        if inst.cls == KEYHOLE:
            track_id = None
        pieces.append(Instance(cls=int(inst.cls), poly=poly, track_id=track_id, src="manual"))
    return pieces


def gap_fill(current: list[Instance], neighbor: list[Instance]) -> list[Instance]:
    """Copy pore ids (and a keyhole, if the frame has none) that this frame is missing."""
    present = {inst.track_id for inst in current if inst.cls == PORE and inst.track_id is not None}
    added: list[Instance] = []
    for inst in neighbor:
        if inst.cls == PORE and inst.track_id is not None and inst.track_id not in present:
            added.append(inst.cloned(src="manual"))
            present.add(inst.track_id)
    if not any(inst.cls == KEYHOLE for inst in current):
        for inst in neighbor:
            if inst.cls == KEYHOLE:
                added.append(inst.cloned(src="manual"))
                break
    return added


def render_overlay(
    gray: np.ndarray,
    instances: list[Instance],
    selected: set[int] | None = None,
    opacity: float = 0.45,
    preview_points=None,
    brush_mask: np.ndarray | None = None,
    sam_points=None,
    extra_mask: np.ndarray | None = None,
) -> np.ndarray:
    selected = selected or set()
    canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB).astype(np.float32)
    alpha = float(np.clip(opacity, 0.0, 1.0))
    height, width = gray.shape[:2]
    for index, inst in enumerate(instances):
        mask = rasterize(inst.poly, height, width).astype(bool)
        if not mask.any():
            continue
        color = np.array(CLASS_COLORS.get(int(inst.cls), (180, 180, 180)), dtype=np.float32)
        canvas[mask] = canvas[mask] * (1.0 - alpha) + color * alpha
    image = np.clip(canvas, 0, 255).astype(np.uint8)
    if extra_mask is not None and np.any(extra_mask):
        region = extra_mask > 0
        tint = np.array([80, 255, 120], dtype=np.float32)
        image[region] = np.clip(image[region].astype(np.float32) * 0.45 + tint * 0.55, 0, 255).astype(np.uint8)
    if brush_mask is not None and np.any(brush_mask):
        region = brush_mask > 0
        tint = np.array([255, 190, 40], dtype=np.float32)
        image[region] = np.clip(image[region].astype(np.float32) * 0.45 + tint * 0.55, 0, 255).astype(np.uint8)
    for index, inst in enumerate(instances):
        if inst.poly is None or len(inst.poly) < 2:
            continue
        points = _contour(inst.poly)
        color = (255, 220, 0) if index in selected else CLASS_COLORS.get(int(inst.cls), (180, 180, 180))
        cv2.polylines(image, [points], True, color, 2 if index in selected else 1, cv2.LINE_AA)
        if index in selected:
            for x_coord, y_coord in np.asarray(inst.poly):
                center = (int(round(x_coord)), int(round(y_coord)))
                cv2.rectangle(image, (center[0] - 3, center[1] - 3), (center[0] + 3, center[1] + 3), (255, 255, 255), -1)
        if int(inst.cls) == PORE:
            cx, cy = centroid(inst.poly)
            label = "?" if inst.track_id is None else str(int(inst.track_id))
            origin = (int(round(cx)) + 4, int(round(cy)))
            cv2.putText(image, label, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(image, label, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA)
    if preview_points is not None and len(preview_points) > 0:
        points = np.round(np.asarray(preview_points, dtype=np.float64)).astype(np.int32)
        if len(points) >= 2:
            cv2.polylines(image, [points.reshape(-1, 1, 2)], False, (255, 255, 0), 1, cv2.LINE_AA)
        for x_coord, y_coord in points:
            cv2.circle(image, (int(x_coord), int(y_coord)), 3, (255, 255, 0), -1)
    if sam_points:
        for x_coord, y_coord, label in sam_points:
            color = (0, 255, 0) if int(label) == 1 else (255, 60, 60)
            cv2.drawMarker(
                image,
                (int(round(x_coord)), int(round(y_coord))),
                color,
                markerType=cv2.MARKER_CROSS,
                markerSize=12,
                thickness=2,
            )
    return image


def default_out_root(in_root: Path) -> Path:
    in_root = Path(in_root)
    return in_root.parent / f"{in_root.name}_corrected"


def run_self_test() -> None:
    """Headless checks for label IO, edits, and the refuse-to-overwrite rule."""
    import tempfile

    sample = (
        "0 0.77656 0.48958 0.78125 0.57292 0.79063 0.63281 0.80000 0.70000\n"
        "1 0.91406 0.83333 0.90938 0.84896 0.92000 0.86000 0.93000 0.84000 0.91000\n"
        "1 0.50000 0.50000 0.56000 0.50000 0.56000 0.56000 0.50000 0.56000\n"
    )
    sidecar = [
        {"cls": 0, "id": None, "src": "sam_prop"},
        {"cls": 1, "id": 1, "src": "blob"},
        {"cls": 1, "id": 2, "src": "blob"},
    ]
    parsed = []
    for line in sample.splitlines():
        item = parse_yolo_line(line, 640, 384)
        assert item is not None
        parsed.append(item)
    assert [item[0] for item in parsed] == [0, 1, 1]
    scored = parse_yolo_line("1 0.1 0.1 0.2 0.1 0.2 0.2 0.90", 100, 100)
    assert scored is not None and abs(scored[2] - 0.90) < 1e-6

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "dataset"
        out = Path(tmp) / "corrected"
        _write_fixture(root, sample, sidecar)
        frames = discover_frames(root, root / "manifest.csv")
        assert len(frames) == 2
        frame = frames[0]
        assert frame.cine == "Test1055" and frame.track == "2" and frame.frame == 243
        original_label = input_label_path(frame.image_path).read_bytes()
        original_ids = input_ids_path(frame.image_path).read_bytes()
        rgb = read_image_rgb(frame.image_path)
        height, width = rgb.shape[:2]
        instances, from_out = load_frame_instances(frame, out, True, width, height)
        assert from_out is False
        assert [inst.track_id for inst in instances] == [None, 1, 2]
        assert [inst.src for inst in instances] == ["sam_prop", "blob", "blob"]
        assert_distinct_roots(root, out)
        try:
            assert_distinct_roots(root, root)
            raise AssertionError("same-root output was accepted")
        except SystemExit:
            pass
        save_correction(root, out, frame, instances, width, height, action="save", detail="self-test")
        assert input_label_path(frame.image_path).read_bytes() == original_label
        assert input_ids_path(frame.image_path).read_bytes() == original_ids
        reloaded, from_out = load_frame_instances(frame, out, True, width, height)
        assert from_out is True
        assert all(inst.src == "manual" for inst in reloaded)
        assert [inst.track_id for inst in reloaded] == [None, 1, 2]
        log_text = (out / "edit_log.csv").read_text(encoding="utf-8")
        assert "save" in log_text and "Test1055" in log_text
        reviews = ReviewStore(out / "reviewed.csv")
        assert reviews.get(frame) is False
        reviews.set(frame, True)
        assert reviews.get(frame) is True
        assert reviews.count(frames) == (1, 2)

        audit = load_audit(root / "audit.csv")
        assert audit_hit(frame, audit) is not None
        assert audit_hit(frames[1], audit) is None

        merged = merge_instances([reloaded[1], reloaded[2]], height, width)
        assert merged is not None and merged.cls == PORE and merged.track_id is None
        block = Instance(PORE, np.array([[10, 10], [50, 10], [50, 40], [10, 40]], dtype=np.float64), 7, "manual")
        pieces = split_with_cut(block, [(30, 5), (30, 45)], 64, 64, thickness=3)
        assert len(pieces) == 2
        assert pieces[0].track_id == 7 and pieces[1].track_id is None
        mask = np.zeros((32, 32), np.uint8)
        paint_disk(mask, 16, 16, 6, 1)
        assert polygon_from_mask(mask) is not None
        current = [reloaded[1]]
        neighbor = reloaded
        filled = gap_fill(current, neighbor)
        assert any(inst.track_id == 2 for inst in filled)
        assert any(inst.cls == KEYHOLE for inst in filled)
        overlay = render_overlay(display_gray(rgb), reloaded, {1}, opacity=0.5)
        assert overlay.shape == rgb.shape
    print("seg_io self-test passed")


def _write_fixture(root: Path, label_text: str, sidecar: list) -> None:
    split = "val"
    stem = "Test1055_T2_00243"
    image = np.zeros((48, 64, 3), np.uint8)
    image[:] = 40
    image[10:30, 40:48] = 210
    write_png(root / "images" / split / f"{stem}.png", image)
    other = "Test1055_T2_00244"
    write_png(root / "images" / split / f"{other}.png", image)
    label_dir = root / "labels" / split
    ids_dir = root / "ids" / split
    label_dir.mkdir(parents=True, exist_ok=True)
    ids_dir.mkdir(parents=True, exist_ok=True)
    (label_dir / f"{stem}.txt").write_text(label_text, encoding="utf-8")
    (ids_dir / f"{stem}.json").write_text(json.dumps(sidecar), encoding="utf-8")
    (label_dir / f"{other}.txt").write_text("", encoding="utf-8")
    (ids_dir / f"{other}.json").write_text("[]", encoding="utf-8")
    (root / "manifest.csv").write_text(
        "split,cine,track,name,frame,laser_on,kh,n_pores,amb,neg\n"
        "val,Test1055,2,Test1055_T2,243,1,1,2,0,0\n"
        "val,Test1055,2,Test1055_T2,244,1,0,0,1,0\n",
        encoding="utf-8",
    )
    (root / "audit.csv").write_text(
        "split,name,frame,reason\nval,Test1055_T2,243,missed pore\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    run_self_test()
