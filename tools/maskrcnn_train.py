"""Train, evaluate, and run torchvision Mask R-CNN on keyhole/pore YOLO-seg labels.

The model is ``maskrcnn_resnet50_fpn_v2`` (COCO initialization by default,
``min_size`` 576). Labels use the dataset layout

    images/{train,val}/<Test####_T#>_<frame>.png
    labels/{train,val}/<same>.txt
    ids/{train,val}/<same>.json

or the split lists named by a YOLO-style yaml (``train:`` / ``val:`` files of
image paths). Class 0 is the keyhole and class 1 is the pore. Inside the
detector those are labels 1 and 2; 0 is background.

Prediction labels are YOLO-seg polygons with the score appended, which is the
Ultralytics ``save_conf`` convention::

    cls x1 y1 x2 y2 ... xn yn score

A matching ids sidecar records ``src=maskrcnn`` and the score. An odd number of
floats after the class is the score, so a reader that only understands plain
YOLO-seg can drop the last field.

Why ``alt_v3.py`` hung after the last track
--------------------------------------------
That script logged the last held-out track and then stopped making progress, so
its 15-minute stall watchdog called ``os._exit(3)``.

1. After the last track it blocks on ``eval_v3.py``. The watchdog treats the
   job as alive only when ``eval_v3_alt/eval.log`` was modified in the last 10
   minutes. The child is started with block-buffered stdout redirected to a
   different file (``logs/alt_eval.log``). If ``eval.log`` is missing, written
   once, or flushed late, the parent never refreshes its heartbeat and the
   watchdog fires. In the log that looks like a hang immediately after the
   last track.
2. The training loader is a ``ThreadPoolExecutor`` that is never shut down,
   feeding a bounded queue. Leaving the iterator early (iteration cap or time
   budget) leaves the producer blocked on ``Queue.put``. The pool's worker
   threads are non-daemon, and the blocked producer keeps the pool alive, so
   ``sys.exit`` waits forever even after eval returns. The script also has no
   ``if __name__ == "__main__"`` guard, which is why it could not use a normal
   DataLoader: on Windows, spawn re-imports the module in every worker and
   would start training again. A CUDA context torn down while those threads
   are still running is a second shutdown hang. ``pickle.dump(open(...))``
   without a close is the same class of bug: on Windows an unclosed handle can
   lock the cache the eval child is trying to read.

This script defaults DataLoader workers to 0 on Windows, shuts loaders down in
``finally``, writes outputs with flush/fsync and atomic replace, synchronizes
CUDA before exiting, turns the OpenCV thread pool off, and keeps a stall
watchdog that hard-exits only when a step really stops calling ``beat``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

cv2.setNumThreads(0)

import torch

from seg_io import (
    IMAGE_EXTS,
    Instance,
    polygon_from_mask,
    read_image_rgb,
    read_instances,
    write_label_pair,
    write_png,
)

IOU_THRESHOLDS = tuple(round(0.50 + 0.05 * index, 2) for index in range(10))


class Logger:
    def __init__(self):
        self._handle = None

    def set_path(self, path: Path | None) -> None:
        self.close()
        if path is None:
            return
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = open(path, "a", encoding="utf-8", buffering=1)

    def __call__(self, message: str) -> None:
        line = time.strftime("%Y-%m-%d %H:%M:%S ") + str(message)
        print(line, flush=True)
        if self._handle is not None:
            self._handle.write(line + "\n")
            self._handle.flush()

    def close(self) -> None:
        if self._handle is not None:
            self._handle.flush()
            self._handle.close()
            self._handle = None


LOGGER = Logger()


class StallWatchdog:
    """Hard-exit if ``beat`` stops. ``seconds <= 0`` disables it.

    ``os._exit`` is deliberate: a deadlock on non-daemon threads would ignore
    ``sys.exit``. The normal path does not use it.
    """

    def __init__(self, seconds: float):
        self.seconds = float(seconds)
        self.last = time.time()
        self._stop = threading.Event()
        self._thread = None
        if self.seconds > 0:
            self._interval = 5.0 if self.seconds >= 30 else max(0.05, self.seconds / 4.0)
            self._thread = threading.Thread(target=self._loop, name="stall-watchdog", daemon=True)
            self._thread.start()

    def beat(self) -> None:
        self.last = time.time()

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        while not self._stop.wait(self._interval):
            idle = time.time() - self.last
            if idle > self.seconds:
                LOGGER(
                    f"WATCHDOG: no progress for {idle:.0f}s (limit {self.seconds:.0f}s); exiting 3"
                )
                LOGGER.close()
                os._exit(3)


def default_workers(requested: int | None, platform_name: str | None = None) -> int:
    """Windows defaults to 0. Spawn workers re-import the script and are a common hang."""
    name = platform_name if platform_name is not None else os.name
    if requested is None:
        if name == "nt":
            return 0
        return min(2, os.cpu_count() or 1)
    count = max(0, int(requested))
    if name == "nt" and count > 0:
        LOGGER(
            f"Windows DataLoader workers often deadlock under spawn; using the requested "
            f"--workers {count}. Prefer 0 if a run stalls after a step."
        )
    return count


def cuda_sync() -> None:
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass


def shutdown_loader(loader) -> None:
    if loader is None:
        return
    iterator = getattr(loader, "_iterator", None)
    if iterator is not None:
        shutdown = getattr(iterator, "_shutdown_workers", None)
        if callable(shutdown):
            try:
                shutdown()
            except Exception:
                pass
        try:
            loader._iterator = None
        except Exception:
            pass


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name.startswith("cuda") and not torch.cuda.is_available():
        LOGGER("CUDA was requested but is not available; using CPU")
        return torch.device("cpu")
    return torch.device(name)


def label_path_for(image_path: Path) -> Path:
    parts = list(Path(image_path).parts)
    if "images" in parts:
        index = len(parts) - 1 - parts[::-1].index("images")
        parts[index] = "labels"
        return Path(*parts).with_suffix(".txt")
    return Path(image_path).with_suffix(".txt")


def split_of(image_path: Path, fallback: str = "pred") -> str:
    parts = list(Path(image_path).parts)
    if "images" in parts:
        index = len(parts) - 1 - parts[::-1].index("images")
        if index + 1 < len(parts) - 1:
            return parts[index + 1]
    return fallback


@dataclass
class Sample:
    image: Path
    split: str


def class_names_from_yaml(data: dict | None) -> list[str]:
    names = (data or {}).get("names") or {0: "keyhole", 1: "pore"}
    if isinstance(names, dict):
        return [str(names[key]) for key in sorted(names, key=lambda item: int(item))]
    return [str(name) for name in names]


def _resolve_entry(entry, root: Path | None) -> list[Path]:
    if entry is None:
        return []
    path = Path(str(entry))
    if not path.is_absolute() and root is not None:
        path = root / path
    if path.is_dir():
        files = [item for item in path.rglob("*") if item.suffix.lower() in IMAGE_EXTS]
        return sorted(files)
    if path.is_file() and path.suffix.lower() in {".txt", ".list"}:
        found = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            image = Path(line)
            if not image.is_absolute() and root is not None:
                candidate = root / line
                image = candidate if candidate.exists() else image
            found.append(image)
        return found
    raise FileNotFoundError(f"Split entry does not exist: {entry}")


def load_yaml(path: Path) -> dict:
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit("Reading a dataset yaml needs pyyaml. pip install pyyaml") from exc
    with open(path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"Dataset yaml must be a mapping: {path}")
    return data


def resolve_samples(data_path: str | None, root: str | None, which: str, limit: int, seed: int) -> tuple[list[Sample], list[str]]:
    names = ["keyhole", "pore"]
    groups: list[tuple[str, list[Path]]] = []
    if data_path:
        cfg = load_yaml(Path(data_path))
        names = class_names_from_yaml(cfg)
        yaml_root = Path(cfg["path"]) if cfg.get("path") else (Path(root) if root else None)
        wanted = []
        if which in {"train", "all"}:
            wanted.append(("train", cfg.get("train")))
        if which in {"val", "all"}:
            wanted.append(("val", cfg.get("val")))
        for fallback, entry in wanted:
            groups.append((fallback, _resolve_entry(entry, yaml_root)))
    elif root:
        base = Path(root)
        if which in {"train", "all"}:
            groups.append(("train", _resolve_entry(base / "images" / "train", None)))
        if which in {"val", "all"}:
            groups.append(("val", _resolve_entry(base / "images" / "val", None)))
    else:
        raise SystemExit("Pass --data <yaml> or --root <dataset>")
    samples: list[Sample] = []
    for fallback, paths in groups:
        for image in paths:
            samples.append(Sample(Path(image), split_of(image, fallback)))
    if which == "train":
        random.Random(seed).shuffle(samples)
    if limit and limit > 0:
        samples = samples[: int(limit)]
    return samples, names


def _augment(image: np.ndarray, instances: list[Instance]) -> tuple[np.ndarray, list[Instance]]:
    contrast = random.uniform(0.85, 1.2)
    brightness = random.uniform(-0.07, 0.07)
    image = np.clip((image - 0.5) * contrast + 0.5 + brightness, 0.0, 1.0)
    if random.random() < 0.5:
        sigma = random.uniform(0.005, 0.025)
        image = np.clip(image + np.random.normal(0, sigma, image.shape).astype(np.float32), 0.0, 1.0)
    if random.random() < 0.5:
        image = np.ascontiguousarray(image[:, ::-1])
        width = image.shape[1]
        for inst in instances:
            inst.poly = np.array(inst.poly, dtype=np.float64, copy=True)
            inst.poly[:, 0] = (width - 1) - inst.poly[:, 0]
    return np.ascontiguousarray(image), instances


class YoloSegDataset(torch.utils.data.Dataset):
    def __init__(self, paths: list[str], augment: bool = False):
        self.paths = list(paths)
        self.augment = bool(augment)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        path = Path(self.paths[index])
        try:
            rgb = read_image_rgb(path)
        except Exception as exc:
            raise RuntimeError(f"Could not read {path}") from exc
        height, width = rgb.shape[:2]
        instances = read_instances(label_path_for(path), None, width, height)
        image = rgb.astype(np.float32) / 255.0
        if self.augment:
            image, instances = _augment(image, instances)
            height, width = image.shape[:2]
        masks = []
        boxes = []
        labels = []
        for inst in instances:
            mask = (cv2.fillPoly(np.zeros((height, width), np.uint8), [np.round(inst.poly).astype(np.int32).reshape(-1, 1, 2)], 1))
            if int(mask.sum()) < 6:
                continue
            ys, xs = np.where(mask > 0)
            x0, x1 = int(xs.min()), int(xs.max()) + 1
            y0, y1 = int(ys.min()), int(ys.max()) + 1
            if x1 <= x0 or y1 <= y0:
                continue
            masks.append(mask)
            boxes.append([x0, y0, x1, y1])
            labels.append(int(inst.cls) + 1)
        target = {
            "boxes": torch.as_tensor(np.array(boxes, dtype=np.float32), dtype=torch.float32).reshape(-1, 4),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "masks": (
                torch.as_tensor(np.stack(masks), dtype=torch.uint8)
                if masks
                else torch.zeros((0, height, width), dtype=torch.uint8)
            ),
        }
        tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))
        return tensor, target


def collate(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


def build_model(num_classes: int, min_size: int, max_size: int, pretrained: str, trainable_layers: int, score_thresh: float):
    from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

    kwargs = dict(
        weights_backbone=None,
        min_size=int(min_size),
        max_size=int(max_size),
        box_score_thresh=float(score_thresh),
        box_nms_thresh=0.5,
        box_detections_per_img=100,
    )
    if pretrained == "coco":
        from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights

        # Only a pretrained backbone honors this. Passing it with random init
        # makes torchvision warn and then ignore the value.
        kwargs["trainable_backbone_layers"] = int(trainable_layers)
        model = maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT, **kwargs)
    else:
        model = maskrcnn_resnet50_fpn_v2(weights=None, num_classes=int(num_classes), **kwargs)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, int(num_classes))
    mask_in = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(mask_in, 256, int(num_classes))
    return model


def load_checkpoint(path: Path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def save_checkpoint(path: Path, model, optimizer, step: int, epoch: int, class_names, min_size: int, max_size: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": None if optimizer is None else optimizer.state_dict(),
        "step": int(step),
        "epoch": int(epoch),
        "class_names": list(class_names),
        "num_classes": len(class_names) + 1,
        "min_size": int(min_size),
        "max_size": int(max_size),
    }
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _move_optimizer(optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _make_scaler(enabled: bool):
    if not enabled:
        return None
    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=True)


def make_loader(dataset, batch: int, workers: int, seed: int):
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=max(int(batch), 1),
        shuffle=True,
        num_workers=int(workers),
        collate_fn=collate,
        drop_last=False,
        generator=generator,
        pin_memory=False,
    )


def learning_rate(step: int, total: int, base: float, warmup: int = 100) -> float:
    progress = min(step / max(total, 1), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    warm = min(1.0, (step + 1) / max(warmup, 1))
    return base * cosine * warm + 1e-6


def mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    left_b = np.asarray(left, dtype=bool)
    right_b = np.asarray(right, dtype=bool)
    if left_b.shape != right_b.shape:
        return 0.0
    intersection = int(np.logical_and(left_b, right_b).sum())
    union = int(np.logical_or(left_b, right_b).sum())
    if union == 0:
        return 0.0
    return intersection / union


def greedy_match(gt_masks, pred_masks, scores, threshold: float):
    true_positive = [False] * len(pred_masks)
    if not pred_masks or not gt_masks:
        return true_positive, 0
    order = list(np.argsort(-np.asarray(scores, dtype=np.float64)))
    used: set[int] = set()
    matched = 0
    for pred_index in order:
        best_iou = float(threshold)
        best_gt = -1
        found = False
        for gt_index, gt_mask in enumerate(gt_masks):
            if gt_index in used:
                continue
            overlap = mask_iou(pred_masks[pred_index], gt_mask)
            if overlap >= threshold and (not found or overlap > best_iou):
                best_iou = overlap
                best_gt = gt_index
                found = True
        if best_gt >= 0:
            used.add(best_gt)
            true_positive[pred_index] = True
            matched += 1
    return true_positive, matched


def average_precision(true_positive, scores, n_gt: int):
    """COCO-style 101-point interpolated average precision."""
    if n_gt <= 0:
        return None
    if len(true_positive) == 0:
        return 0.0
    order = np.argsort(-np.asarray(scores, dtype=np.float64))
    tp = np.asarray(true_positive, dtype=np.float64)[order]
    fp = 1.0 - tp
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / float(n_gt)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
    curve = []
    for threshold in np.linspace(0.0, 1.0, 101):
        eligible = precision[recall >= float(threshold) - 1e-12]
        curve.append(float(eligible.max()) if eligible.size else 0.0)
    return float(np.mean(curve))


class MetricAccum:
    def __init__(self, n_classes: int):
        self.n_classes = n_classes
        self.buckets = []
        for _cls in range(n_classes):
            self.buckets.append(
                {
                    "n_gt": 0,
                    "n_pred": 0,
                    "matched50": 0,
                    "tp": {thr: [] for thr in IOU_THRESHOLDS},
                    "scores": {thr: [] for thr in IOU_THRESHOLDS},
                }
            )

    def add(self, gt_by_cls: dict, pred_by_cls: dict) -> None:
        for cls in range(self.n_classes):
            gt_masks = gt_by_cls.get(cls, [])
            preds = pred_by_cls.get(cls, [])
            bucket = self.buckets[cls]
            bucket["n_gt"] += len(gt_masks)
            bucket["n_pred"] += len(preds)
            scores = [float(score) for _mask, score in preds]
            masks = [mask for mask, _score in preds]
            for thr in IOU_THRESHOLDS:
                tp, matched = greedy_match(gt_masks, masks, scores, thr)
                bucket["tp"][thr].extend(tp)
                bucket["scores"][thr].extend(scores)
                if abs(thr - 0.5) < 1e-9:
                    bucket["matched50"] += matched

    def to_dict(self, names: list[str]) -> dict:
        per_class = {}
        for cls, name in enumerate(names):
            bucket = self.buckets[cls]
            aps = [
                average_precision(bucket["tp"][thr], bucket["scores"][thr], bucket["n_gt"])
                for thr in IOU_THRESHOLDS
            ]
            finite = [value for value in aps if value is not None]
            n_gt = bucket["n_gt"]
            per_class[name] = {
                "AP50": aps[0],
                "AP50_95": None if not finite else float(np.mean(finite)),
                "recall50": None if n_gt == 0 else bucket["matched50"] / n_gt,
                "gt": n_gt,
                "predictions": bucket["n_pred"],
            }
        pore_index = 0
        for index, name in enumerate(names):
            if name.lower() == "pore":
                pore_index = index
                break
        pore = per_class[names[pore_index]]
        return {
            "pore_AP50": pore["AP50"],
            "pore_AP50_95": pore["AP50_95"],
            "pore_recall50": pore["recall50"],
            "pore_gt": pore["gt"],
            "pore_predictions": pore["predictions"],
            "per_class": per_class,
            "backend": "numpy",
            "iou_thresholds": list(IOU_THRESHOLDS),
        }


def _instances_to_masks(instances: list[Instance], height: int, width: int, n_classes: int):
    by_class = {cls: [] for cls in range(n_classes)}
    for inst in instances:
        if inst.cls < 0 or inst.cls >= n_classes or inst.poly is None:
            continue
        mask = cv2.fillPoly(
            np.zeros((height, width), np.uint8),
            [np.round(inst.poly).astype(np.int32).reshape(-1, 1, 2)],
            1,
        )
        if int(mask.sum()) < 1:
            continue
        if inst.score is None:
            by_class[int(inst.cls)].append(mask.astype(bool))
        else:
            by_class[int(inst.cls)].append((mask.astype(bool), float(inst.score)))
    return by_class


def detections_from_output(output, height: int, width: int, score_thresh: float, min_area: float) -> list[Instance]:
    labels = output["labels"].detach().cpu().numpy()
    scores = output["scores"].detach().cpu().numpy()
    masks = output["masks"].detach().cpu().numpy()
    found = []
    for label, score, mask in zip(labels, scores, masks):
        if float(score) < score_thresh:
            continue
        yolo_cls = int(label) - 1
        if yolo_cls < 0:
            continue
        probability = mask[0] if mask.ndim == 3 else mask
        if probability.shape != (height, width):
            probability = cv2.resize(probability.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
        binary = (probability >= 0.5).astype(np.uint8)
        if float(binary.sum()) < min_area:
            continue
        poly = polygon_from_mask(binary, epsilon=0.8, min_area=min_area)
        if poly is None:
            continue
        found.append(Instance(yolo_cls, poly, None, "maskrcnn", float(score)))
    return found


@torch.inference_mode()
def infer_images(model, rgbs: list[np.ndarray], device: torch.device):
    tensors = []
    for rgb in rgbs:
        tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float().div_(255.0)
        tensors.append(tensor.to(device))
    outputs = model(tensors)
    if device.type == "cuda":
        torch.cuda.synchronize()
    moved = []
    for output in outputs:
        moved.append(
            {
                "boxes": output["boxes"].detach().cpu(),
                "labels": output["labels"].detach().cpu(),
                "scores": output["scores"].detach().cpu(),
                "masks": output["masks"].detach().cpu(),
            }
        )
    del outputs, tensors
    return moved


def _model_for_inference(args, device: torch.device):
    checkpoint = load_checkpoint(Path(args.weights), device)
    names = list(checkpoint.get("class_names") or ["keyhole", "pore"])
    min_size = int(args.min_size or checkpoint.get("min_size") or 576)
    max_size = int(args.max_size or checkpoint.get("max_size") or 960)
    model = build_model(
        num_classes=len(names) + 1,
        min_size=min_size,
        max_size=max_size,
        pretrained="none",
        trainable_layers=int(getattr(args, "backbone_layers", 3) or 3),
        score_thresh=float(args.score_thresh),
    )
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    model.roi_heads.score_thresh = float(args.score_thresh)
    return model, names, min_size, max_size


def train(args) -> None:
    seed_all(int(args.seed))
    device = pick_device(args.device)
    samples, names = resolve_samples(args.data, args.root, "train", int(args.limit or 0), int(args.seed))
    if not samples:
        raise SystemExit("No training images. Check --data or --root and the split lists.")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    LOGGER.set_path(out / "train.log")
    watchdog = StallWatchdog(float(args.stall_seconds))
    workers = default_workers(args.workers)
    use_amp = bool(args.amp) and device.type == "cuda"
    if args.amp and device.type != "cuda":
        LOGGER("AMP needs CUDA; training in full precision")
    LOGGER(
        f"train images={len(samples)} device={device.type} workers={workers} "
        f"batch={args.batch} iters={args.iters} amp={use_amp} min_size={args.min_size}"
    )
    watchdog.beat()
    model = build_model(
        num_classes=len(names) + 1,
        min_size=int(args.min_size),
        max_size=int(args.max_size),
        pretrained="none" if args.resume else args.weights,
        trainable_layers=int(args.backbone_layers),
        score_thresh=0.01,
    )
    model.to(device)
    params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=float(args.lr), weight_decay=float(args.weight_decay))
    scaler = _make_scaler(use_amp)
    step = 0
    epoch = 0
    if args.resume:
        checkpoint = load_checkpoint(Path(args.resume), device)
        model.load_state_dict(checkpoint["model"])
        if checkpoint.get("optimizer"):
            optimizer.load_state_dict(checkpoint["optimizer"])
            _move_optimizer(optimizer, device)
        step = int(checkpoint.get("step") or 0)
        epoch = int(checkpoint.get("epoch") or 0)
        names = list(checkpoint.get("class_names") or names)
        LOGGER(f"resumed at step {step}")
    dataset = YoloSegDataset([str(sample.image) for sample in samples], augment=not args.no_augment)
    total = int(args.iters)
    best_smooth = None
    smooth_loss = None
    started = time.time()
    loader = None
    try:
        if step >= total:
            LOGGER(f"checkpoint is already at step {step} (target {total}); writing weights and exiting")
        model.train()
        while step < total:
            if args.minutes and (time.time() - started) > float(args.minutes) * 60.0:
                LOGGER("time budget reached")
                break
            loader = make_loader(dataset, int(args.batch), workers, int(args.seed) + epoch)
            try:
                for images, targets in loader:
                    if step >= total:
                        break
                    if args.minutes and (time.time() - started) > float(args.minutes) * 60.0:
                        break
                    watchdog.beat()
                    lr = learning_rate(step, total, float(args.lr))
                    for group in optimizer.param_groups:
                        group["lr"] = lr
                    images = [image.to(device) for image in images]
                    targets = [{key: value.to(device) for key, value in target.items()} for target in targets]
                    optimizer.zero_grad(set_to_none=True)
                    if use_amp:
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            loss_dict = model(images, targets)
                            loss = sum(loss_dict.values())
                    else:
                        loss_dict = model(images, targets)
                        loss = sum(loss_dict.values())
                    loss_value = float(loss.detach().float().cpu())
                    if not math.isfinite(loss_value):
                        LOGGER(f"non-finite loss at step {step}; skipping the update")
                        optimizer.zero_grad(set_to_none=True)
                    elif use_amp:
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                        optimizer.step()
                    step += 1
                    if step == 1 or step % int(args.log_every) == 0:
                        LOGGER(f"step {step}/{total} loss {loss_value:.4f} lr {lr:.3e}")
                    if math.isfinite(loss_value):
                        smooth_loss = loss_value if smooth_loss is None else 0.9 * smooth_loss + 0.1 * loss_value
                        if best_smooth is None or smooth_loss < best_smooth:
                            best_smooth = smooth_loss
                            save_checkpoint(
                                out / "best.pt", model, optimizer, step, epoch, names, args.min_size, args.max_size
                            )
                    if step % int(args.ckpt_every) == 0:
                        save_checkpoint(
                            out / "last.pt", model, optimizer, step, epoch, names, args.min_size, args.max_size
                        )
                    del images, targets, loss, loss_dict
            finally:
                shutdown_loader(loader)
                loader = None
            epoch += 1
        save_checkpoint(out / "last.pt", model, optimizer, step, epoch, names, args.min_size, args.max_size)
        if not (out / "best.pt").exists():
            save_checkpoint(out / "best.pt", model, optimizer, step, epoch, names, args.min_size, args.max_size)
        LOGGER(f"training finished at step {step}")
    finally:
        shutdown_loader(loader)
        watchdog.stop()
        cuda_sync()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def evaluate(args) -> dict:
    device = pick_device(args.device)
    out = Path(args.out) if args.out else Path(args.weights).resolve().parent
    out.mkdir(parents=True, exist_ok=True)
    LOGGER.set_path(out / "eval.log")
    watchdog = StallWatchdog(float(args.stall_seconds))
    samples, names = resolve_samples(args.data, args.root, args.split, int(args.limit or 0), int(args.seed))
    if not samples:
        raise SystemExit(f"No images for split {args.split}")
    watchdog.beat()
    model, names, _min_size, _max_size = _model_for_inference(args, device)
    accum = MetricAccum(len(names))
    coco_records = [] if args.pycocotools else None
    LOGGER(f"evaluate images={len(samples)} device={device.type} score>={args.score_thresh}")
    batch = max(int(args.batch), 1)
    try:
        for start in range(0, len(samples), batch):
            watchdog.beat()
            chunk = samples[start : start + batch]
            rgbs = []
            metas = []
            for sample in chunk:
                rgb = read_image_rgb(sample.image)
                rgbs.append(rgb)
                metas.append((sample, rgb.shape[0], rgb.shape[1]))
            outputs = infer_images(model, rgbs, device)
            for (sample, height, width), output in zip(metas, outputs):
                gt = read_instances(label_path_for(sample.image), None, width, height)
                gt_masks = _instances_to_masks(gt, height, width, len(names))
                gt_by = {cls: masks for cls, masks in gt_masks.items()}
                preds = detections_from_output(output, height, width, float(args.score_thresh), min_area=6)
                pred_by = {cls: [] for cls in range(len(names))}
                for inst in preds:
                    if 0 <= inst.cls < len(names):
                        pred_by[inst.cls].append(
                            (
                                cv2.fillPoly(
                                    np.zeros((height, width), np.uint8),
                                    [np.round(inst.poly).astype(np.int32).reshape(-1, 1, 2)],
                                    1,
                                ).astype(bool),
                                float(inst.score),
                            )
                        )
                accum.add(gt_by, pred_by)
                if coco_records is not None:
                    pore_cls = next((i for i, name in enumerate(names) if str(name).lower() == "pore"), 0)
                    coco_records.append(
                        (
                            height,
                            width,
                            gt_by.get(pore_cls, []),
                            pred_by.get(pore_cls, []),
                        )
                    )
            done = min(start + batch, len(samples))
            if start == 0 or done == len(samples) or (start // batch) % 20 == 0:
                LOGGER(f"evaluated {done}/{len(samples)}")
        metrics = accum.to_dict(names)
        if coco_records is not None:
            metrics["pycocotools"] = coco_pore_ap(coco_records)
        path = out / "metrics.json"
        path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        LOGGER(
            "pore AP50={pore_AP50} AP50-95={pore_AP50_95} recall50={pore_recall50} "
            "gt={pore_gt} pred={pore_predictions} backend={backend}".format(**metrics)
        )
        return metrics
    finally:
        watchdog.stop()
        cuda_sync()


def _mask_bbox(mask: np.ndarray) -> list[int]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return [0, 0, 0, 0]
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return [x0, y0, int(x1 - x0 + 1), int(y1 - y0 + 1)]


def coco_pore_ap(records) -> dict:
    """Official COCO mask AP for pores, when pycocotools is installed.

    ``pore_AP*`` in the metrics file is always the numpy equivalent. This adds
    the official numbers beside it.
    """
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        from pycocotools import mask as mask_utils
    except ImportError:
        return {
            "available": False,
            "detail": "pycocotools is not installed; pore_AP50 and pore_AP50_95 are the numpy COCO equivalent",
        }
    images = []
    annotations = []
    detections = []
    ann_id = 1
    for image_id, (height, width, gt_masks, preds) in enumerate(records, start=1):
        images.append({"id": image_id, "height": int(height), "width": int(width)})
        for mask in gt_masks:
            encoded = mask_utils.encode(np.asfortranarray(np.asarray(mask, dtype=np.uint8)))
            encoded["counts"] = encoded["counts"].decode("ascii")
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "iscrowd": 0,
                    "area": int(np.asarray(mask).sum()),
                    "bbox": _mask_bbox(mask),
                    "segmentation": encoded,
                }
            )
            ann_id += 1
        for mask, score in preds:
            encoded = mask_utils.encode(np.asfortranarray(np.asarray(mask, dtype=np.uint8)))
            encoded["counts"] = encoded["counts"].decode("ascii")
            detections.append(
                {
                    "image_id": image_id,
                    "category_id": 1,
                    "score": float(score),
                    "bbox": _mask_bbox(mask),
                    "segmentation": encoded,
                }
            )
    if not annotations:
        return {"available": True, "pore_AP50": None, "pore_AP50_95": None, "detail": "no pore ground truth"}
    coco = COCO()
    coco.dataset = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "pore"}],
    }
    coco.createIndex()
    if not detections:
        return {"available": True, "pore_AP50": 0.0, "pore_AP50_95": 0.0}
    coco_dt = coco.loadRes(detections)
    evaluator = COCOeval(coco, coco_dt, iouType="segm")
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    return {
        "available": True,
        "pore_AP50": float(evaluator.stats[1]),
        "pore_AP50_95": float(evaluator.stats[0]),
    }


def predict(args) -> None:
    device = pick_device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    LOGGER.set_path(out / "predict.log")
    watchdog = StallWatchdog(float(args.stall_seconds))
    samples, names = resolve_samples(args.data, args.root, args.split, int(args.limit or 0), int(args.seed))
    if not samples:
        raise SystemExit(f"No images for split {args.split}")
    watchdog.beat()
    model, names, _min_size, _max_size = _model_for_inference(args, device)
    LOGGER(f"predict images={len(samples)} -> {out} score>={args.score_thresh}")
    summary_path = out / "summary.csv"
    batch = max(int(args.batch), 1)
    try:
        with open(summary_path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["split", "file", "n_keyhole", "n_pore", "max_score"])
            for start in range(0, len(samples), batch):
                watchdog.beat()
                chunk = samples[start : start + batch]
                rgbs = [read_image_rgb(sample.image) for sample in chunk]
                outputs = infer_images(model, rgbs, device)
                for sample, rgb, output in zip(chunk, rgbs, outputs):
                    height, width = rgb.shape[:2]
                    instances = detections_from_output(
                        output, height, width, float(args.score_thresh), float(args.min_area)
                    )
                    label_path = out / "labels" / sample.split / f"{sample.image.stem}.txt"
                    ids_path = out / "ids" / sample.split / f"{sample.image.stem}.json"
                    write_label_pair(
                        label_path,
                        ids_path,
                        instances,
                        width,
                        height,
                        src="maskrcnn",
                        with_score=True,
                    )
                    n_keyhole = sum(inst.cls == 0 for inst in instances)
                    n_pore = sum(inst.cls == 1 for inst in instances)
                    max_score = max((inst.score or 0.0) for inst in instances) if instances else 0.0
                    writer.writerow([sample.split, sample.image.name, n_keyhole, n_pore, f"{max_score:.6f}"])
                    handle.flush()
                done = min(start + batch, len(samples))
                if start == 0 or done == len(samples) or (start // batch) % 20 == 0:
                    LOGGER(f"predicted {done}/{len(samples)}")
        LOGGER(f"predictions written under {out}")
    finally:
        watchdog.stop()
        cuda_sync()


def _circle(center, radius, count=10) -> np.ndarray:
    points = []
    for index in range(count):
        angle = 2.0 * math.pi * index / count
        points.append((center[0] + radius * math.cos(angle), center[1] + radius * math.sin(angle)))
    return np.asarray(points, dtype=np.float64)


def write_synthetic_dataset(root: Path, size: int = 64, n_train: int = 2, n_val: int = 2) -> None:
    """Tiny keyhole-plus-pore frames so the CPU smoke test does not need real data."""
    root = Path(root)
    lists = {}
    for split, count in (("train", n_train), ("val", n_val)):
        paths = []
        for index in range(count):
            frame = index + 1
            stem = f"Test9001_T1_{frame:05d}"
            image = np.full((size, size, 3), 28, np.uint8)
            x0 = size // 2
            image[6 : size - 6, x0 : x0 + 5] = 220
            pore_center = (14, size - 14)
            cv2.circle(image, pore_center, 5, (235, 235, 235), -1)
            image_path = root / "images" / split / f"{stem}.png"
            write_png(image_path, image)
            keyhole = np.array(
                [[x0, 6], [x0 + 5, 6], [x0 + 5, size - 6], [x0, size - 6]],
                dtype=np.float64,
            )
            instances = [
                Instance(0, keyhole, None, "synthetic"),
                Instance(1, _circle(pore_center, 5), 1, "synthetic"),
            ]
            write_label_pair(
                root / "labels" / split / f"{stem}.txt",
                root / "ids" / split / f"{stem}.json",
                instances,
                size,
                size,
                src="synthetic",
                with_score=False,
            )
            paths.append(image_path.resolve())
        list_path = root / "lists" / f"{split}.txt"
        list_path.parent.mkdir(parents=True, exist_ok=True)
        list_path.write_text("".join(f"{path}\n" for path in paths), encoding="utf-8")
        lists[split] = list_path.resolve()
    yaml_path = root / "data.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {root.resolve().as_posix()}",
                f"train: {lists['train'].as_posix()}",
                f"val: {lists['val'].as_posix()}",
                "names:",
                "  0: keyhole",
                "  1: pore",
                "",
            ]
        ),
        encoding="utf-8",
    )


def run_smoke(out: Path) -> None:
    """One CPU iteration, then evaluate and predict, on a synthetic dataset."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    root = out / "synthetic"
    write_synthetic_dataset(root, size=64)
    run = out / "run"
    LOGGER("smoke: training 1 iteration on CPU")
    train(
        argparse.Namespace(
            data=str(root / "data.yaml"),
            root=None,
            out=str(run),
            device="cpu",
            workers=0,
            min_size=64,
            max_size=64,
            stall_seconds=0,
            seed=0,
            iters=1,
            batch=1,
            lr=1e-4,
            weight_decay=1e-4,
            amp=False,
            resume=None,
            ckpt_every=1,
            weights="none",
            backbone_layers=3,
            no_augment=True,
            limit=0,
            log_every=1,
            minutes=0,
            score_thresh=0.01,
        )
    )
    LOGGER("smoke: evaluate")
    metrics = evaluate(
        argparse.Namespace(
            data=str(root / "data.yaml"),
            root=None,
            weights=str(run / "last.pt"),
            out=str(out / "eval"),
            device="cpu",
            workers=0,
            min_size=64,
            max_size=64,
            stall_seconds=0,
            seed=0,
            split="val",
            score_thresh=0.01,
            batch=1,
            backbone_layers=3,
            limit=0,
            pycocotools=False,
        )
    )
    for key in ("pore_AP50", "pore_AP50_95", "pore_recall50"):
        if key not in metrics:
            raise SystemExit(f"smoke metrics missing {key}")
    LOGGER("smoke: predict")
    predict(
        argparse.Namespace(
            data=str(root / "data.yaml"),
            root=None,
            weights=str(run / "last.pt"),
            out=str(out / "predictions"),
            device="cpu",
            workers=0,
            min_size=64,
            max_size=64,
            stall_seconds=0,
            seed=0,
            split="val",
            score_thresh=0.01,
            batch=1,
            backbone_layers=3,
            limit=0,
            min_area=6,
        )
    )
    written = list((out / "predictions").rglob("*.txt"))
    if len(written) < 2:
        raise SystemExit(f"smoke predict wrote {len(written)} label files")
    cuda_sync()
    LOGGER.close()
    print("SMOKE_OK", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mask R-CNN for keyhole and pore YOLO-seg labels.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_data(target):
        target.add_argument("--data", default=None, help="Dataset yaml with path/train/val/names.")
        target.add_argument("--root", default=None, help="Dataset root with images/{train,val}.")
        target.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
        target.add_argument("--workers", type=int, default=None, help="DataLoader workers. Default 0 on Windows.")
        target.add_argument("--stall-seconds", type=float, default=900, help="Stall watchdog. 0 disables it.")
        target.add_argument("--seed", type=int, default=5)
        target.add_argument("--limit", type=int, default=0, help="Use only the first N images after shuffling train.")

    train_parser = sub.add_parser("train", help="Train or resume.")
    add_data(train_parser)
    train_parser.add_argument("--out", required=True)
    train_parser.add_argument("--iters", type=int, default=4000)
    train_parser.add_argument("--batch", type=int, default=2)
    train_parser.add_argument("--lr", type=float, default=1e-4)
    train_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_parser.add_argument("--min-size", type=int, default=576)
    train_parser.add_argument("--max-size", type=int, default=960)
    train_parser.add_argument("--amp", action="store_true", help="Mixed precision. Ignored on CPU.")
    train_parser.add_argument("--resume", default=None, help="Checkpoint to resume (last.pt).")
    train_parser.add_argument("--ckpt-every", type=int, default=500)
    train_parser.add_argument("--log-every", type=int, default=50)
    train_parser.add_argument("--weights", choices=["coco", "none"], default="coco")
    train_parser.add_argument("--backbone-layers", type=int, default=3)
    train_parser.add_argument("--no-augment", action="store_true")
    train_parser.add_argument("--minutes", type=float, default=0, help="Optional wall-clock budget. 0 means no budget.")
    train_parser.add_argument("--smoke", action="store_true", help="Run the synthetic CPU smoke test into --out.")

    eval_parser = sub.add_parser("evaluate", help="Pore AP50, AP50-95, and recall.")
    add_data(eval_parser)
    eval_parser.add_argument("--weights", required=True)
    eval_parser.add_argument("--out", default=None)
    eval_parser.add_argument("--split", default="val", choices=["train", "val", "all"])
    eval_parser.add_argument("--score-thresh", type=float, default=0.01)
    eval_parser.add_argument("--batch", type=int, default=1)
    eval_parser.add_argument("--min-size", type=int, default=None)
    eval_parser.add_argument("--max-size", type=int, default=None)
    eval_parser.add_argument("--backbone-layers", type=int, default=3)
    eval_parser.add_argument(
        "--pycocotools",
        action="store_true",
        help="Note pycocotools availability. Reported AP is the numpy COCO equivalent either way.",
    )

    pred_parser = sub.add_parser("predict", help="Write YOLO-seg polygons with scores.")
    add_data(pred_parser)
    pred_parser.add_argument("--weights", required=True)
    pred_parser.add_argument("--out", required=True)
    pred_parser.add_argument("--split", default="val", choices=["train", "val", "all"])
    pred_parser.add_argument("--score-thresh", type=float, default=0.01)
    pred_parser.add_argument("--batch", type=int, default=1)
    pred_parser.add_argument("--min-size", type=int, default=None)
    pred_parser.add_argument("--max-size", type=int, default=None)
    pred_parser.add_argument("--backbone-layers", type=int, default=3)
    pred_parser.add_argument("--min-area", type=float, default=6)

    smoke_parser = sub.add_parser("smoke", help="Synthetic CPU train, evaluate, and predict.")
    smoke_parser.add_argument("--out", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    cv2.setNumThreads(0)
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        if args.cmd == "train" and getattr(args, "smoke", False):
            run_smoke(Path(args.out))
        elif args.cmd == "train":
            train(args)
        elif args.cmd == "evaluate":
            evaluate(args)
        elif args.cmd == "predict":
            predict(args)
        elif args.cmd == "smoke":
            run_smoke(Path(args.out))
        else:
            parser.error(f"unknown command {args.cmd}")
        return 0
    except SystemExit as exc:
        code = exc.code
        if code in (None, 0):
            return 0
        if isinstance(code, int):
            return code
        LOGGER(str(code))
        return 1
    except Exception:
        LOGGER("ERROR")
        traceback.print_exc()
        return 1
    finally:
        cuda_sync()
        LOGGER.close()


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    raise SystemExit(main())
