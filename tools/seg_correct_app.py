"""PySide6 window for the keyhole/pore segmentation corrector.

PySide6 plus OpenCV is the editing stack: Qt supplies the frame slider, keyboard
shortcuts, and dialogs, and OpenCV rasterizes polygons, brush strokes, and
cuts. Napari can show masks, but a vertex / brush editor on top of it pulls in
VisPy and a plugin host that is awkward to launch on a lab Windows machine.
SAM stays an optional import so the editor still opens when it is absent.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
from PySide6.QtCore import QEvent, QPointF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QKeySequence, QPainter, QShortcut
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from seg_io import (
    CLASS_NAMES,
    KEYHOLE,
    PORE,
    FrameRec,
    Instance,
    ReviewStore,
    append_edit_log,
    apply_contrast_gamma,
    assert_distinct_roots,
    audit_hit,
    default_out_root,
    delete_vertex,
    discover_frames,
    display_gray,
    gap_fill,
    hit_test,
    insert_vertex,
    load_audit,
    load_frame_instances,
    merge_instances,
    move_vertex,
    nearest_edge_point,
    nearest_vertex,
    paint_disk,
    polygon_from_mask,
    rasterize,
    read_image_rgb,
    render_overlay,
    save_correction,
    split_components,
    split_with_cut,
)

HELP_TEXT = """
Select (V)     click an instance; drag a vertex; double-click an edge to insert;
               right-click a vertex to delete it; Delete removes the instance
Polygon (A)    left-click vertices, Enter or double-click to close, Backspace undoes a point
Brush (B)      paint a new mask (added to the selected instance when one is selected)
Eraser (E)     erase the selection, or the instance under the cursor
Split (X)      draw a cut across the selected instance, Enter to apply
Split pieces   break an instance that is already disconnected
SAM (S)        left-click include, right-click exclude, Enter to accept
               (off unless segment_anything and --sam-checkpoint are available)
K / P          keyhole / pore (applies to the selection, and sets the class for new masks)
I              set the pore track id (blank clears it)
M              merge the selection
G / Shift+G    copy missing pores (and a keyhole, if absent) from the previous / next frame
C / Shift+C    copy every instance from the previous / next frame
Left / Right   previous / next frame
Shift+arrows   previous / next flagged frame (audit CSV)
R              toggle reviewed
Ctrl+S         save YOLO-seg labels and the ids sidecar (src=manual) under --out
Ctrl+Z / Ctrl+Y  undo / redo
[ ]            overlay opacity
- / =          brush size
Wheel          zoom; Alt-drag or middle-drag pans; Ctrl+0 resets the view
Esc            cancel the polygon, cut, or SAM points
""".strip()


class UndoStack:
    def __init__(self, limit: int = 100):
        self.limit = limit
        self._undo: list[list[Instance]] = []
        self._redo: list[list[Instance]] = []

    def push(self, state: list[Instance]) -> None:
        self._undo.append(state)
        if len(self._undo) > self.limit:
            self._undo.pop(0)
        self._redo.clear()

    def undo(self, current: list[Instance]):
        if not self._undo:
            return None
        self._redo.append(copy.deepcopy(current))
        return self._undo.pop()

    def redo(self, current: list[Instance]):
        if not self._redo:
            return None
        self._undo.append(copy.deepcopy(current))
        return self._redo.pop()


class SamHelper:
    """Optional Segment Anything click predictor. Failure leaves the editor usable."""

    def __init__(self, checkpoint: Path | None, device: str | None):
        self.predictor = None
        self.available = False
        self.reason = "SAM click-to-segment is off."
        self._ready = False
        self.device = device or "cpu"
        if checkpoint is None:
            self.reason = "SAM is off (no --sam-checkpoint). Polygon, brush, and eraser still work."
            return
        try:
            from segment_anything import SamPredictor, sam_model_registry
        except ImportError:
            self.reason = "segment_anything is not installed. SAM is off; the rest of the editor works."
            return
        path = Path(checkpoint)
        if not path.is_file():
            self.reason = f"SAM checkpoint not found ({path}). SAM is off."
            return
        name = path.name.lower()
        if "vit_h" in name:
            model_type = "vit_h"
        elif "vit_l" in name:
            model_type = "vit_l"
        else:
            model_type = "vit_b"
        chosen = self.device
        try:
            import torch

            if chosen == "cuda" and not torch.cuda.is_available():
                chosen = "cpu"
            if device is None:
                chosen = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            chosen = "cpu"
        self.device = chosen
        try:
            model = sam_model_registry[model_type](checkpoint=str(path))
            model.to(device=chosen)
            self.predictor = SamPredictor(model)
            self.available = True
            self.reason = f"SAM {model_type} on {chosen}"
        except Exception as exc:  # weight/device errors should not kill the GUI
            self.reason = f"SAM failed to load ({exc}). The rest of the editor works."

    def reset(self) -> None:
        self._ready = False

    def ensure_image(self, rgb: np.ndarray) -> None:
        if self.predictor is None or self._ready:
            return
        self.predictor.set_image(rgb)
        self._ready = True

    def predict(self, points: list[tuple[float, float, int]]):
        if self.predictor is None or not points:
            return None
        coords = np.array([[point[0], point[1]] for point in points], dtype=np.float32)
        labels = np.array([point[2] for point in points], dtype=np.int32)
        masks, scores, _logits = self.predictor.predict(
            point_coords=coords, point_labels=labels, multimask_output=True
        )
        return masks[int(np.argmax(scores))].astype(np.uint8)


class ImageCanvas(QWidget):
    """Zoomable RGB view. Mouse positions are emitted in image pixels."""

    pressed = Signal(float, float, int)
    moved = Signal(float, float, int)
    released = Signal(float, float, int)
    doubled = Signal(float, float, int)

    def __init__(self):
        super().__init__()
        self._image: QImage | None = None
        self._buffer = None
        self.zoom = 1.0
        self.pan = QPointF(0, 0)
        self._panning = False
        self._pan_origin = None
        self._pan_start = QPointF(0, 0)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMinimumSize(480, 360)

    def set_rgb(self, rgb: np.ndarray) -> None:
        array = np.ascontiguousarray(rgb)
        height, width = array.shape[:2]
        self._buffer = array
        image = QImage(array.data, width, height, width * 3, QImage.Format_RGB888)
        self._image = image.copy()
        self.update()

    def reset_view(self) -> None:
        self.zoom = 1.0
        self.pan = QPointF(0, 0)
        self.update()

    def _geometry(self):
        if self._image is None or self._image.width() < 1 or self._image.height() < 1:
            return 1.0, 0.0, 0.0
        fit = min(self.width() / self._image.width(), self.height() / self._image.height())
        scale = fit * self.zoom
        origin_x = (self.width() - self._image.width() * scale) / 2 + self.pan.x()
        origin_y = (self.height() - self._image.height() * scale) / 2 + self.pan.y()
        return scale, origin_x, origin_y

    def to_image(self, pos) -> tuple[float, float]:
        scale, origin_x, origin_y = self._geometry()
        if scale == 0:
            return 0.0, 0.0
        return (pos.x() - origin_x) / scale, (pos.y() - origin_y) / scale

    def _event_pos(self, event):
        return event.position() if hasattr(event, "position") else event.localPos()

    def wheelEvent(self, event):
        if self._image is None:
            return
        cursor = self._event_pos(event)
        before = self.to_image(cursor)
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.zoom = float(min(20.0, max(0.25, self.zoom * factor)))
        fit = min(self.width() / self._image.width(), self.height() / self._image.height())
        scale = fit * self.zoom
        base_x = (self.width() - self._image.width() * scale) / 2
        base_y = (self.height() - self._image.height() * scale) / 2
        self.pan = QPointF(cursor.x() - before[0] * scale - base_x, cursor.y() - before[1] * scale - base_y)
        self.update()

    def mousePressEvent(self, event):
        alt = bool(event.modifiers() & Qt.AltModifier)
        if event.button() == Qt.MiddleButton or (event.button() == Qt.LeftButton and alt):
            self._panning = True
            self._pan_origin = self._event_pos(event)
            self._pan_start = QPointF(self.pan)
            return
        x_coord, y_coord = self.to_image(self._event_pos(event))
        self.pressed.emit(x_coord, y_coord, int(event.button()))

    def mouseMoveEvent(self, event):
        if self._panning and self._pan_origin is not None:
            delta = self._event_pos(event) - self._pan_origin
            self.pan = self._pan_start + delta
            self.update()
            return
        x_coord, y_coord = self.to_image(self._event_pos(event))
        self.moved.emit(x_coord, y_coord, int(event.buttons()))

    def mouseReleaseEvent(self, event):
        if self._panning and event.button() in (Qt.MiddleButton, Qt.LeftButton):
            self._panning = False
            return
        x_coord, y_coord = self.to_image(self._event_pos(event))
        self.released.emit(x_coord, y_coord, int(event.button()))

    def mouseDoubleClickEvent(self, event):
        x_coord, y_coord = self.to_image(self._event_pos(event))
        self.doubled.emit(x_coord, y_coord, int(event.button()))

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#1b1f24"))
        if self._image is None:
            painter.setPen(QColor("#d0d4dc"))
            painter.drawText(self.rect(), Qt.AlignCenter, "No frame")
            painter.end()
            return
        scale, origin_x, origin_y = self._geometry()
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        painter.translate(origin_x, origin_y)
        painter.scale(scale, scale)
        painter.drawImage(0, 0, self._image)
        painter.end()


def _slider(minimum: int, maximum: int, value: int) -> QSlider:
    slider = QSlider(Qt.Horizontal)
    slider.setRange(minimum, maximum)
    slider.setValue(value)
    return slider


class MainWindow(QMainWindow):
    def __init__(self, root: Path, out: Path | None, resume: bool, audit: Path | None, manifest: Path | None, sam_checkpoint: Path | None, device: str | None):
        super().__init__()
        self.root = Path(root)
        self.out = Path(out) if out is not None else default_out_root(self.root)
        self.resume = bool(resume)
        self.audit_path = Path(audit) if audit else None
        self.manifest_path = Path(manifest) if manifest else None
        self.sam = SamHelper(sam_checkpoint, device or "cpu")
        self.frames: list[FrameRec] = []
        self.view: list[FrameRec] = []
        self.audit: list[dict] = []
        self.view_index = -1
        self.frame: FrameRec | None = None
        self.instances: list[Instance] = []
        self.selected: set[int] = set()
        self.undo = UndoStack()
        self.dirty = False
        self.mode = "select"
        self.image_rgb: np.ndarray | None = None
        self.from_output = False
        self.poly_pts: list[tuple[float, float]] = []
        self.hover: tuple[float, float] | None = None
        self.split_pts: list[tuple[float, float]] = []
        self.sam_points: list[tuple[float, float, int]] = []
        self.sam_mask = None
        self.brush_mask = None
        self._stroke_before = None
        self._erase_targets: set[int] = set()
        self._painting = False
        self._drag = None
        self._active_vertex = None
        self._guard = False
        self.reviews = ReviewStore(self.out / "reviewed.csv")
        self.setWindowTitle("SMAXI segmentation correction")
        self.resize(1440, 900)
        self._build_ui()
        self._install_shortcuts()
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)
        self.open_dataset(self.root, self.out, announce=False)

    def _build_ui(self) -> None:
        splitter = QSplitter()
        self.setCentralWidget(splitter)
        side = QWidget()
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(8, 8, 8, 8)

        self.path_label = QLabel()
        self.path_label.setWordWrap(True)
        side_layout.addWidget(self.path_label)

        filters = QFormLayout()
        self.split_combo = QComboBox()
        self.cine_combo = QComboBox()
        self.track_combo = QComboBox()
        self.split_combo.currentIndexChanged.connect(self._on_filter_changed)
        self.cine_combo.currentIndexChanged.connect(self._on_cine_changed)
        self.track_combo.currentIndexChanged.connect(self._on_filter_changed)
        filters.addRow("Split", self.split_combo)
        filters.addRow("Cine", self.cine_combo)
        filters.addRow("Track", self.track_combo)
        side_layout.addLayout(filters)

        self.flagged_only = QCheckBox("Flagged frames only")
        self.flagged_only.toggled.connect(lambda _checked: self._rebuild_view())
        side_layout.addWidget(self.flagged_only)
        self.reviewed_box = QCheckBox("Reviewed")
        self.reviewed_box.toggled.connect(self._on_reviewed)
        side_layout.addWidget(self.reviewed_box)
        self.mark_on_save = QCheckBox("Mark reviewed when saving")
        self.mark_on_save.setChecked(True)
        side_layout.addWidget(self.mark_on_save)
        self.progress = QProgressBar()
        self.progress_label = QLabel("Reviewed 0/0")
        side_layout.addWidget(self.progress)
        side_layout.addWidget(self.progress_label)

        modes = QGroupBox("Tool")
        mode_layout = QVBoxLayout(modes)
        self.mode_group = QButtonGroup(self)
        self.mode_group.setExclusive(True)
        self.mode_buttons = {}
        for mode, label in (
            ("select", "Select (V)"),
            ("polygon", "Polygon (A)"),
            ("brush", "Brush (B)"),
            ("eraser", "Eraser (E)"),
            ("split", "Split cut (X)"),
            ("sam", "SAM click (S)"),
        ):
            button = QPushButton(label)
            button.setCheckable(True)
            button.clicked.connect(lambda _checked=False, name=mode: self.set_mode(name))
            self.mode_group.addButton(button)
            self.mode_buttons[mode] = button
            mode_layout.addWidget(button)
        self.mode_buttons["select"].setChecked(True)
        self.mode_buttons["sam"].setEnabled(self.sam.available)
        self.mode_buttons["sam"].setToolTip(self.sam.reason)
        side_layout.addWidget(modes)

        form = QFormLayout()
        self.class_combo = QComboBox()
        self.class_combo.addItem("keyhole", KEYHOLE)
        self.class_combo.addItem("pore", PORE)
        self.class_combo.setCurrentIndex(1)
        self.id_edit = QLineEdit()
        self.id_edit.setPlaceholderText("pore track id")
        self.id_edit.returnPressed.connect(self.apply_track_id)
        id_row = QWidget()
        id_layout = QHBoxLayout(id_row)
        id_layout.setContentsMargins(0, 0, 0, 0)
        id_button = QPushButton("Set id")
        id_button.clicked.connect(self.apply_track_id)
        id_layout.addWidget(self.id_edit)
        id_layout.addWidget(id_button)
        self.channel_combo = QComboBox()
        self.channel_combo.addItems(["mean", "R", "G", "B"])
        self.channel_combo.currentTextChanged.connect(lambda _text: self._refresh())
        self.opacity = _slider(0, 100, 45)
        self.contrast = _slider(20, 300, 100)
        self.gamma = _slider(20, 300, 100)
        self.brush = _slider(1, 40, 6)
        for slider in (self.opacity, self.contrast, self.gamma, self.brush):
            slider.valueChanged.connect(lambda _value: self._refresh())
        form.addRow("New class", self.class_combo)
        form.addRow("Track id", id_row)
        form.addRow("Channel", self.channel_combo)
        form.addRow("Opacity", self.opacity)
        form.addRow("Contrast", self.contrast)
        form.addRow("Gamma", self.gamma)
        form.addRow("Brush", self.brush)
        side_layout.addLayout(form)

        copy_box = QGroupBox("Neighbor frames")
        copy_layout = QVBoxLayout(copy_box)
        for text, slot in (
            ("Copy previous (C)", lambda: self.copy_neighbor(-1, gap=False)),
            ("Copy next (Shift+C)", lambda: self.copy_neighbor(1, gap=False)),
            ("Gap-fill from previous (G)", lambda: self.copy_neighbor(-1, gap=True)),
            ("Gap-fill from next (Shift+G)", lambda: self.copy_neighbor(1, gap=True)),
        ):
            button = QPushButton(text)
            button.clicked.connect(slot)
            copy_layout.addWidget(button)
        side_layout.addWidget(copy_box)

        edit_row = QHBoxLayout()
        for text, slot in (
            ("Merge", self.merge_selected),
            ("Split pieces", self.split_selected_components),
            ("Delete", self.delete_selected),
        ):
            button = QPushButton(text)
            button.clicked.connect(slot)
            edit_row.addWidget(button)
        side_layout.addLayout(edit_row)
        save_row = QHBoxLayout()
        for text, slot in (("Save", self.save_current), ("Undo", self.do_undo), ("Redo", self.do_redo)):
            button = QPushButton(text)
            button.clicked.connect(slot)
            save_row.addWidget(button)
        side_layout.addLayout(save_row)

        self.instance_list = QListWidget()
        self.instance_list.setSelectionMode(QListWidget.ExtendedSelection)
        self.instance_list.itemSelectionChanged.connect(self._on_list_selection)
        side_layout.addWidget(self.instance_list, stretch=1)
        self.sam_label = QLabel(self.sam.reason)
        self.sam_label.setWordWrap(True)
        side_layout.addWidget(self.sam_label)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(side)
        scroll.setMinimumWidth(300)
        splitter.addWidget(scroll)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        self.canvas = ImageCanvas()
        self.canvas.pressed.connect(self._on_press)
        self.canvas.moved.connect(self._on_move)
        self.canvas.released.connect(self._on_release)
        self.canvas.doubled.connect(self._on_double)
        right_layout.addWidget(self.canvas, stretch=1)
        nav = QHBoxLayout()
        self.prev_button = QPushButton("<")
        self.next_button = QPushButton(">")
        self.prev_flag_button = QPushButton("Prev flagged")
        self.next_flag_button = QPushButton("Next flagged")
        self.prev_button.clicked.connect(lambda: self.step(-1))
        self.next_button.clicked.connect(lambda: self.step(1))
        self.prev_flag_button.clicked.connect(lambda: self.jump_flagged(-1))
        self.next_flag_button.clicked.connect(lambda: self.jump_flagged(1))
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setFocusPolicy(Qt.NoFocus)
        self.slider.valueChanged.connect(self._on_slider)
        self.frame_spin = QSpinBox()
        self.frame_spin.setRange(0, 999999)
        self.frame_spin.setKeyboardTracking(False)
        self.frame_spin.valueChanged.connect(self._on_spin)
        nav.addWidget(self.prev_button)
        nav.addWidget(self.slider, stretch=1)
        nav.addWidget(self.next_button)
        nav.addWidget(QLabel("Frame"))
        nav.addWidget(self.frame_spin)
        nav.addWidget(self.prev_flag_button)
        nav.addWidget(self.next_flag_button)
        right_layout.addLayout(nav)
        self.frame_label = QLabel("No dataset")
        self.frame_label.setWordWrap(True)
        right_layout.addWidget(self.frame_label)
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)

        file_menu = self.menuBar().addMenu("File")
        file_menu.addAction("Open dataset…", self._prompt_open)
        file_menu.addAction("Save", self.save_current)
        file_menu.addSeparator()
        file_menu.addAction("Exit", self.close)
        edit_menu = self.menuBar().addMenu("Edit")
        edit_menu.addAction("Undo", self.do_undo)
        edit_menu.addAction("Redo", self.do_redo)
        edit_menu.addAction("Delete", self.delete_selected)
        edit_menu.addAction("Merge", self.merge_selected)
        help_menu = self.menuBar().addMenu("Help")
        help_menu.addAction("Shortcuts", self.show_help)
        self.statusBar().showMessage(self.sam.reason)

    def _install_shortcuts(self) -> None:
        self._shortcuts = []

        def add(sequence, slot):
            shortcut = QShortcut(QKeySequence(sequence), self)
            shortcut.setContext(Qt.WindowShortcut)
            shortcut.activated.connect(slot)
            self._shortcuts.append(shortcut)

        add("Ctrl+S", self.save_current)
        add("Ctrl+Z", self.do_undo)
        add("Ctrl+Y", self.do_redo)
        add("Ctrl+Shift+Z", self.do_redo)
        add("Ctrl+0", self.canvas.reset_view)

    def open_dataset(self, root: Path, out: Path, announce: bool = True) -> None:
        try:
            assert_distinct_roots(root, out)
        except SystemExit as exc:
            QMessageBox.critical(self, "Output folder", str(exc))
            return
        self.root = Path(root)
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.reviews = ReviewStore(self.out / "reviewed.csv")
        manifest = self.manifest_path if self.manifest_path else (self.root / "manifest.csv")
        try:
            self.frames = discover_frames(self.root, manifest if manifest.is_file() else None)
        except Exception as exc:
            self.frames = []
            QMessageBox.critical(self, "Dataset", str(exc))
        audit_file = self.audit_path
        self.audit = load_audit(audit_file) if audit_file else []
        self.path_label.setText(f"Input: {self.root}\nOutput: {self.out}")
        self._fill_filters()
        if announce:
            self.statusBar().showMessage(f"Loaded {len(self.frames)} frames")

    def _prompt_open(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Dataset root", str(self.root))
        if not directory:
            return
        root = Path(directory)
        out = default_out_root(root)
        self.open_dataset(root, out)

    def _fill_filters(self) -> None:
        self._guard = True
        self.split_combo.clear()
        self.split_combo.addItem("All", "")
        for split in sorted({frame.split for frame in self.frames}):
            self.split_combo.addItem(split, split)
        self._refill_cines()
        self._refill_tracks()
        self._guard = False
        self._rebuild_view()

    def _refill_cines(self) -> None:
        current = self.cine_combo.currentText() if self.cine_combo.count() else ""
        self.cine_combo.blockSignals(True)
        self.cine_combo.clear()
        self.cine_combo.addItem("All", "")
        for cine in sorted({frame.cine for frame in self.frames}):
            self.cine_combo.addItem(cine, cine)
        index = self.cine_combo.findText(current)
        self.cine_combo.setCurrentIndex(index if index >= 0 else 0)
        self.cine_combo.blockSignals(False)

    def _refill_tracks(self) -> None:
        cine = self.cine_combo.currentData() or ""
        current = self.track_combo.currentData() if self.track_combo.count() else ""
        self.track_combo.blockSignals(True)
        self.track_combo.clear()
        self.track_combo.addItem("All", "")
        tracks = {frame.track for frame in self.frames if not cine or frame.cine == cine}
        for track in sorted(tracks, key=lambda value: (int(value) if str(value).isdigit() else 10**9, str(value))):
            self.track_combo.addItem(f"T{track}", track)
        index = self.track_combo.findData(current)
        self.track_combo.setCurrentIndex(index if index >= 0 else 0)
        self.track_combo.blockSignals(False)

    def _on_cine_changed(self) -> None:
        if self._guard:
            return
        self._guard = True
        self._refill_tracks()
        self._guard = False
        self._rebuild_view()

    def _on_filter_changed(self) -> None:
        if self._guard:
            return
        self._rebuild_view()

    def _rebuild_view(self, keep: FrameRec | None = None) -> None:
        if self._guard:
            return
        split = self.split_combo.currentData() or ""
        cine = self.cine_combo.currentData() or ""
        track = self.track_combo.currentData() or ""
        flagged_only = self.flagged_only.isChecked()
        view = []
        for frame in self.frames:
            if split and frame.split != split:
                continue
            if cine and frame.cine != cine:
                continue
            if track and frame.track != track:
                continue
            if flagged_only and audit_hit(frame, self.audit) is None:
                continue
            view.append(frame)
        self.view = view
        target = keep or self.frame
        index = 0
        if target is not None:
            for i, frame in enumerate(view):
                if frame.image_path == target.image_path:
                    index = i
                    break
        self.slider.blockSignals(True)
        self.slider.setRange(0, max(len(view) - 1, 0))
        self.slider.blockSignals(False)
        self._update_progress()
        if not view:
            self.view_index = -1
            self.frame = None
            self.instances = []
            self.frame_label.setText("No frames match this filter.")
            self._refresh()
            return
        self.view_index = -1
        self.set_view_index(index, force=True)

    def set_view_index(self, index: int, force: bool = False) -> None:
        if not self.view:
            return
        index = int(min(max(index, 0), len(self.view) - 1))
        if index == self.view_index and self.frame is not None and not force:
            self._sync_slider()
            return
        if self.dirty and not force:
            choice = QMessageBox.question(
                self,
                "Unsaved edits",
                "Save this frame before leaving?",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
                QMessageBox.Save,
            )
            if choice == QMessageBox.Cancel:
                self._sync_slider()
                return
            if choice == QMessageBox.Save:
                self.save_current()
            self.dirty = False
        self.view_index = index
        self._load_current()

    def _sync_slider(self) -> None:
        self.slider.blockSignals(True)
        if self.view_index >= 0:
            self.slider.setValue(self.view_index)
        self.slider.blockSignals(False)

    def _load_current(self) -> None:
        self.frame = self.view[self.view_index]
        self.poly_pts = []
        self.split_pts = []
        self.sam_points = []
        self.sam_mask = None
        self.brush_mask = None
        self._drag = None
        self._active_vertex = None
        self.hover = None
        self.undo = UndoStack()
        self.sam.reset()
        try:
            self.image_rgb = read_image_rgb(self.frame.image_path)
        except Exception as exc:
            self.image_rgb = np.zeros((64, 64, 3), np.uint8)
            self.instances = []
            self.statusBar().showMessage(str(exc))
            self._refresh()
            return
        height, width = self.image_rgb.shape[:2]
        self.instances, self.from_output = load_frame_instances(
            self.frame, self.out, self.resume, width, height
        )
        self.selected = set()
        self.dirty = False
        self.reviewed_box.blockSignals(True)
        self.reviewed_box.setChecked(self.reviews.get(self.frame))
        self.reviewed_box.blockSignals(False)
        self.frame_spin.blockSignals(True)
        self.frame_spin.setValue(self.frame.frame)
        self.frame_spin.blockSignals(False)
        self._sync_slider()
        self._refresh()
        self.canvas.setFocus()

    def _update_progress(self) -> None:
        done, total = self.reviews.count(self.frames)
        self.progress.setMaximum(max(total, 1))
        self.progress.setValue(done)
        percent = (100.0 * done / total) if total else 0.0
        self.progress_label.setText(f"Reviewed {done}/{total} ({percent:.0f}%)")

    def _on_slider(self, value: int) -> None:
        if self._guard:
            return
        self.set_view_index(value)

    def _on_spin(self, value: int) -> None:
        if self._guard or not self.view:
            return
        for index, frame in enumerate(self.view):
            if frame.frame == value:
                self.set_view_index(index)
                return
        self.statusBar().showMessage(f"Frame {value} is not in this filter")

    def step(self, delta: int) -> None:
        if not self.view:
            return
        self.set_view_index(self.view_index + delta)

    def jump_flagged(self, step: int) -> None:
        if not self.view:
            return
        for offset in range(1, len(self.view) + 1):
            index = (self.view_index + step * offset) % len(self.view)
            if audit_hit(self.view[index], self.audit):
                self.set_view_index(index)
                return
        self.statusBar().showMessage("No flagged frames in this filter")

    def _on_reviewed(self, checked: bool) -> None:
        if self.frame is None or self._guard:
            return
        self.reviews.set(self.frame, bool(checked))
        self._log("reviewed", "1" if checked else "0")
        self._update_progress()

    def _active_class(self) -> int:
        return int(self.class_combo.currentData())

    def set_mode(self, mode: str) -> None:
        if mode == "sam" and not self.sam.available:
            self.statusBar().showMessage(self.sam.reason)
            self.mode_buttons["select"].setChecked(True)
            self.mode = "select"
            return
        self.mode = mode
        self.mode_buttons[mode].setChecked(True)
        self.poly_pts = []
        self.split_pts = []
        self.hover = None
        if mode != "sam":
            self.sam_points = []
            self.sam_mask = None
        self._refresh()

    def _image_size(self) -> tuple[int, int]:
        if self.image_rgb is None:
            return 1, 1
        height, width = self.image_rgb.shape[:2]
        return height, width

    def _tolerance(self) -> float:
        return max(4.0, 8.0 / max(self.canvas.zoom, 0.25))

    def _push_undo(self) -> None:
        self.undo.push(copy.deepcopy(self.instances))

    def _log(self, action: str, detail: str = "") -> None:
        if self.frame is None:
            return
        append_edit_log(self.out / "edit_log.csv", self.frame, action, detail, len(self.instances))

    def _refresh(self) -> None:
        self._sync_list()
        self._describe_frame()
        if self.image_rgb is None:
            return
        gray = display_gray(self.image_rgb, self.channel_combo.currentText())
        gray = apply_contrast_gamma(gray, self.contrast.value() / 100.0, self.gamma.value() / 100.0)
        preview = list(self.poly_pts if self.mode == "polygon" else self.split_pts)
        if preview and self.hover is not None and self.mode in {"polygon", "split"}:
            preview = preview + [self.hover]
        image = render_overlay(
            gray,
            self.instances,
            self.selected,
            opacity=self.opacity.value() / 100.0,
            preview_points=preview,
            brush_mask=self.brush_mask,
            sam_points=self.sam_points,
            extra_mask=self.sam_mask,
        )
        self.canvas.set_rgb(image)
        star = "*" if self.dirty else ""
        source = "corrected" if self.from_output else "input"
        name = self.frame.stem if self.frame else ""
        self.setWindowTitle(f"SMAXI segmentation correction — {name}{star} ({source})")

    def _describe_frame(self) -> None:
        if self.frame is None:
            self.frame_label.setText("No frame")
            return
        hit = audit_hit(self.frame, self.audit)
        reason = f"  AUDIT: {hit.get('reason') or 'flagged'}" if hit else ""
        meta = self.frame.meta or {}
        extras = []
        for key in ("laser_on", "kh", "n_pores", "amb", "neg"):
            if meta.get(key) not in (None, ""):
                extras.append(f"{key}={meta[key]}")
        extra = ("  " + " ".join(extras)) if extras else ""
        source = "corrected output" if self.from_output else "input labels"
        self.frame_label.setText(
            f"{self.frame.split}  {self.frame.cine}  T{self.frame.track}  "
            f"frame {self.frame.frame}  ({self.view_index + 1}/{len(self.view)})  "
            f"{len(self.instances)} instances  source: {source}{extra}{reason}"
        )

    def _sync_list(self) -> None:
        self.instance_list.blockSignals(True)
        self.instance_list.clear()
        for index, inst in enumerate(self.instances):
            name = CLASS_NAMES.get(int(inst.cls), str(inst.cls))
            ident = "" if inst.track_id is None else f"  id={inst.track_id}"
            score = "" if inst.score is None else f"  {inst.score:.2f}"
            self.instance_list.addItem(f"{index}: {name}{ident}{score}  [{inst.src}]")
        for index in self.selected:
            item = self.instance_list.item(index)
            if item is not None:
                item.setSelected(True)
        self.instance_list.blockSignals(False)

    def _on_list_selection(self) -> None:
        self.selected = {item.row() for item in self.instance_list.selectedIndexes()}
        pores = [self.instances[index] for index in sorted(self.selected) if self.instances[index].cls == PORE]
        self.id_edit.blockSignals(True)
        if len(pores) == 1 and pores[0].track_id is not None:
            self.id_edit.setText(str(pores[0].track_id))
        elif len(pores) == 1:
            self.id_edit.clear()
        self.id_edit.blockSignals(False)
        if self.image_rgb is not None:
            self._refresh_canvas_only()

    def _refresh_canvas_only(self) -> None:
        """Repaint after a list click without rebuilding the list (avoids a signal loop)."""
        if self.image_rgb is None:
            return
        gray = display_gray(self.image_rgb, self.channel_combo.currentText())
        gray = apply_contrast_gamma(gray, self.contrast.value() / 100.0, self.gamma.value() / 100.0)
        image = render_overlay(
            gray,
            self.instances,
            self.selected,
            opacity=self.opacity.value() / 100.0,
            brush_mask=self.brush_mask,
            sam_points=self.sam_points,
            extra_mask=self.sam_mask,
        )
        self.canvas.set_rgb(image)

    def _on_press(self, x_coord: float, y_coord: float, button: int) -> None:
        if self.image_rgb is None:
            return
        left = int(Qt.LeftButton)
        right = int(Qt.RightButton)
        height, width = self._image_size()
        if self.mode == "select" and button == left:
            tolerance = self._tolerance()
            if self.selected:
                for index in sorted(self.selected):
                    vertex = nearest_vertex(self.instances[index], x_coord, y_coord, tolerance)
                    if vertex is not None:
                        self._drag = (index, vertex, False)
                        self._active_vertex = (index, vertex)
                        return
            hit = hit_test(self.instances, x_coord, y_coord)
            mods = QApplication.keyboardModifiers()
            if hit is None:
                if not (mods & Qt.ControlModifier):
                    self.selected = set()
                    self._active_vertex = None
                self._refresh()
                return
            if mods & Qt.ControlModifier:
                if hit in self.selected:
                    self.selected.remove(hit)
                else:
                    self.selected.add(hit)
            else:
                self.selected = {hit}
            self._active_vertex = None
            self._refresh()
            return
        if self.mode == "select" and button == right and self._active_vertex is not None:
            index, vertex = self._active_vertex
            if 0 <= index < len(self.instances) and nearest_vertex(self.instances[index], x_coord, y_coord, self._tolerance()) == vertex:
                if len(self.instances[index].poly) <= 3:
                    self.statusBar().showMessage("A polygon needs at least 3 vertices")
                    return
                self._push_undo()
                delete_vertex(self.instances[index], vertex)
                self.instances[index].src = "manual"
                self._active_vertex = None
                self.dirty = True
                self._log("delete_vertex")
                self._refresh()
            return
        if self.mode == "polygon" and button == left:
            self.poly_pts.append((x_coord, y_coord))
            self._refresh()
            return
        if self.mode == "polygon" and button == right and self.poly_pts:
            self.poly_pts.pop()
            self._refresh()
            return
        if self.mode == "split" and button == left:
            self.split_pts.append((x_coord, y_coord))
            self._refresh()
            return
        if self.mode == "sam" and button in (left, right):
            label = 1 if button == left else 0
            self.sam_points.append((x_coord, y_coord, label))
            self._run_sam_preview()
            return
        if self.mode in {"brush", "eraser"} and button == left:
            self._painting = True
            self._stroke_before = copy.deepcopy(self.instances)
            self.brush_mask = np.zeros((height, width), np.uint8)
            if self.mode == "eraser":
                if self.selected:
                    self._erase_targets = set(self.selected)
                else:
                    hit = hit_test(self.instances, x_coord, y_coord)
                    self._erase_targets = set() if hit is None else {hit}
            paint_disk(self.brush_mask, x_coord, y_coord, self.brush.value(), 1)
            self._refresh()

    def _on_move(self, x_coord: float, y_coord: float, buttons: int) -> None:
        left = int(Qt.LeftButton)
        if self._drag is not None and (buttons & left):
            index, vertex, snapped = self._drag
            if not snapped:
                self._push_undo()
                self._drag = (index, vertex, True)
            height, width = self._image_size()
            move_vertex(self.instances[index], vertex, x_coord, y_coord, width, height)
            self.instances[index].src = "manual"
            self.dirty = True
            self._refresh()
            return
        if self._painting and self.brush_mask is not None and (buttons & left) and self.mode in {"brush", "eraser"}:
            paint_disk(self.brush_mask, x_coord, y_coord, self.brush.value(), 1)
            self._refresh()
            return
        if self.mode in {"polygon", "split"} and (self.poly_pts or self.split_pts):
            self.hover = (x_coord, y_coord)
            self._refresh()

    def _on_release(self, _x: float, _y: float, button: int) -> None:
        if self._drag is not None:
            _index, _vertex, snapped = self._drag
            self._drag = None
            if snapped:
                self._log("move_vertex")
            return
        if button != int(Qt.LeftButton) or not self._painting:
            return
        self._painting = False
        if self.mode == "brush":
            self._commit_brush()
        elif self.mode == "eraser":
            self._commit_eraser()

    def _on_double(self, x_coord: float, y_coord: float, button: int) -> None:
        if button != int(Qt.LeftButton):
            return
        if self.mode == "polygon":
            self._close_polygon(drop_last=True)
            return
        if self.mode != "select" or not self.selected:
            return
        tolerance = self._tolerance()
        for index in sorted(self.selected):
            if nearest_vertex(self.instances[index], x_coord, y_coord, tolerance) is not None:
                return
            edge = nearest_edge_point(self.instances[index], x_coord, y_coord)
            if edge is not None and edge[2] <= tolerance:
                self._push_undo()
                insert_vertex(self.instances[index], edge[0], edge[1])
                self.instances[index].src = "manual"
                self.dirty = True
                self._log("insert_vertex")
                self._refresh()
                return

    def _close_polygon(self, drop_last: bool = False) -> None:
        points = list(self.poly_pts)
        if drop_last and len(points) >= 2 and _distance(points[-1], points[-2]) < 4:
            points = points[:-1]
        if len(points) >= 2 and _distance(points[0], points[-1]) < 4:
            points = points[:-1]
        if len(points) < 3:
            self.statusBar().showMessage("A polygon needs at least 3 vertices")
            return
        self._push_undo()
        inst = Instance(self._active_class(), np.asarray(points, dtype=np.float64), None, "manual")
        self.instances.append(inst)
        self.selected = {len(self.instances) - 1}
        self.poly_pts = []
        self.hover = None
        self.dirty = True
        self._log("add_polygon")
        self._refresh()

    def _commit_brush(self) -> None:
        mask = self.brush_mask
        before = self._stroke_before
        self.brush_mask = None
        self._stroke_before = None
        if mask is None or int(mask.sum()) < 6:
            self._refresh()
            return
        height, width = self._image_size()
        if len(self.selected) == 1:
            index = next(iter(self.selected))
            base = np.maximum(rasterize(self.instances[index].poly, height, width), mask)
            poly = polygon_from_mask(base, min_area=6)
            if poly is None or before is None:
                self._refresh()
                return
            self.undo.push(before)
            self.instances[index].poly = poly
            self.instances[index].src = "manual"
            self.dirty = True
            self._log("brush")
            self._refresh()
            return
        poly = polygon_from_mask(mask, min_area=6)
        if poly is None or before is None:
            self._refresh()
            return
        self.undo.push(before)
        self.instances.append(Instance(self._active_class(), poly, None, "manual"))
        self.selected = {len(self.instances) - 1}
        self.dirty = True
        self._log("brush")
        self._refresh()

    def _commit_eraser(self) -> None:
        mask = self.brush_mask
        before = self._stroke_before
        targets = set(self._erase_targets)
        self.brush_mask = None
        self._stroke_before = None
        self._erase_targets = set()
        if mask is None or before is None or int(mask.sum()) < 1 or not targets:
            self._refresh()
            return
        height, width = self._image_size()
        updated: list[Instance] = []
        changed = False
        for index, inst in enumerate(self.instances):
            if index not in targets:
                updated.append(inst)
                continue
            base = rasterize(inst.poly, height, width)
            base[mask > 0] = 0
            poly = polygon_from_mask(base, min_area=6)
            changed = True
            if poly is None:
                continue
            clone = inst.cloned(src="manual")
            clone.poly = poly
            updated.append(clone)
        if not changed:
            self._refresh()
            return
        self.undo.push(before)
        self.instances = updated
        self.selected = set()
        self._active_vertex = None
        self.dirty = True
        self._log("eraser")
        self._refresh()

    def _run_sam_preview(self) -> None:
        if not self.sam.available or self.image_rgb is None:
            self.statusBar().showMessage(self.sam.reason)
            return
        try:
            self.sam.ensure_image(self.image_rgb)
            self.sam_mask = self.sam.predict(self.sam_points)
        except Exception as exc:
            self.sam_mask = None
            self.statusBar().showMessage(f"SAM error: {exc}")
            return
        self._refresh()

    def _accept_sam(self) -> None:
        if self.sam_mask is None:
            self.statusBar().showMessage(self.sam.reason if not self.sam.available else "Click the object first")
            return
        poly = polygon_from_mask(self.sam_mask, min_area=6)
        if poly is None:
            self.statusBar().showMessage("SAM mask is empty")
            return
        self._push_undo()
        self.instances.append(Instance(self._active_class(), poly, None, "manual"))
        self.selected = {len(self.instances) - 1}
        self.sam_points = []
        self.sam_mask = None
        self.dirty = True
        self._log("sam")
        self._refresh()

    def _commit_split(self) -> None:
        if len(self.selected) != 1:
            self.statusBar().showMessage("Select one instance, then draw the cut")
            return
        if len(self.split_pts) < 2:
            self.statusBar().showMessage("A cut needs at least two points")
            return
        index = next(iter(self.selected))
        height, width = self._image_size()
        pieces = split_with_cut(
            self.instances[index],
            self.split_pts,
            height,
            width,
            thickness=max(2, self.brush.value() // 2),
        )
        self.split_pts = []
        if len(pieces) < 2:
            self.statusBar().showMessage("That cut did not divide the instance")
            self._refresh()
            return
        self._push_undo()
        self.instances.pop(index)
        self.instances[index:index] = pieces
        self.selected = set(range(index, index + len(pieces)))
        self.dirty = True
        self._log("split", f"pieces={len(pieces)}")
        self._refresh()

    def apply_track_id(self) -> None:
        pores = [index for index in sorted(self.selected) if self.instances[index].cls == PORE]
        if not pores:
            self.statusBar().showMessage("Select a pore before setting a track id")
            return
        text = self.id_edit.text().strip()
        if text == "":
            track_id = None
        else:
            try:
                track_id = int(text)
            except ValueError:
                self.statusBar().showMessage("Track id must be an integer, or blank to clear it")
                return
            if track_id < 0:
                self.statusBar().showMessage("Track id must be >= 0")
                return
        self._push_undo()
        for index in pores:
            self.instances[index].track_id = track_id
            self.instances[index].src = "manual"
        self.dirty = True
        self._log("set_id", "" if track_id is None else str(track_id))
        self._refresh()

    def assign_class(self, cls: int) -> None:
        self.class_combo.setCurrentIndex(0 if cls == KEYHOLE else 1)
        if not self.selected:
            return
        self._push_undo()
        for index in sorted(self.selected):
            self.instances[index].cls = int(cls)
            self.instances[index].src = "manual"
            if cls == KEYHOLE:
                self.instances[index].track_id = None
        self.dirty = True
        self._log("change_class", CLASS_NAMES[cls])
        self._refresh()

    def delete_selected(self) -> None:
        if self._active_vertex is not None and len(self.selected) == 1:
            index, vertex = self._active_vertex
            if 0 <= index < len(self.instances) and len(self.instances[index].poly) > 3:
                self._push_undo()
                delete_vertex(self.instances[index], vertex)
                self.instances[index].src = "manual"
                self._active_vertex = None
                self.dirty = True
                self._log("delete_vertex")
                self._refresh()
                return
        if not self.selected:
            return
        self._push_undo()
        self.instances = [inst for index, inst in enumerate(self.instances) if index not in self.selected]
        self.selected = set()
        self._active_vertex = None
        self.dirty = True
        self._log("delete")
        self._refresh()

    def merge_selected(self) -> None:
        if len(self.selected) < 2:
            self.statusBar().showMessage("Select two or more instances to merge")
            return
        height, width = self._image_size()
        chosen = [self.instances[index] for index in sorted(self.selected)]
        merged = merge_instances(chosen, height, width)
        if merged is None:
            self.statusBar().showMessage("Could not merge those masks")
            return
        self._push_undo()
        keep = [inst for index, inst in enumerate(self.instances) if index not in self.selected]
        keep.append(merged)
        self.instances = keep
        self.selected = {len(self.instances) - 1}
        self.dirty = True
        self._log("merge")
        self._refresh()

    def split_selected_components(self) -> None:
        if len(self.selected) != 1:
            self.statusBar().showMessage("Select one instance to split into pieces")
            return
        index = next(iter(self.selected))
        height, width = self._image_size()
        pieces = split_components(self.instances[index], height, width)
        if len(pieces) < 2:
            self.statusBar().showMessage("That instance is a single piece. Use Split cut to divide it.")
            return
        self._push_undo()
        self.instances.pop(index)
        self.instances[index:index] = pieces
        self.selected = set(range(index, index + len(pieces)))
        self.dirty = True
        self._log("split_components", f"pieces={len(pieces)}")
        self._refresh()

    def copy_neighbor(self, delta: int, gap: bool) -> None:
        if not self.view or self.frame is None or self.image_rgb is None:
            return
        neighbor_index = self.view_index + delta
        if neighbor_index < 0 or neighbor_index >= len(self.view):
            self.statusBar().showMessage("No frame on that side")
            return
        neighbor = self.view[neighbor_index]
        height, width = self._image_size()
        try:
            neighbor_rgb = read_image_rgb(neighbor.image_path)
        except Exception as exc:
            self.statusBar().showMessage(f"Could not read the neighbor frame: {exc}")
            return
        neighbor_h, neighbor_w = neighbor_rgb.shape[:2]
        neighbor_instances, _from_out = load_frame_instances(
            neighbor, self.out, self.resume, neighbor_w, neighbor_h
        )
        if (neighbor_h, neighbor_w) != (height, width) and neighbor_w > 0 and neighbor_h > 0:
            scale_x = width / float(neighbor_w)
            scale_y = height / float(neighbor_h)
            for inst in neighbor_instances:
                inst.poly = np.array(inst.poly, dtype=np.float64, copy=True)
                inst.poly[:, 0] *= scale_x
                inst.poly[:, 1] *= scale_y
        added = gap_fill(self.instances, neighbor_instances) if gap else [inst.cloned(src="manual") for inst in neighbor_instances]
        if not added:
            self.statusBar().showMessage("Nothing to copy")
            return
        self._push_undo()
        start = len(self.instances)
        self.instances.extend(added)
        self.selected = set(range(start, len(self.instances)))
        self.dirty = True
        action = ("gap_fill_prev" if delta < 0 else "gap_fill_next") if gap else ("copy_prev" if delta < 0 else "copy_next")
        self._log(action, f"n={len(added)}")
        self._refresh()

    def do_undo(self) -> None:
        restored = self.undo.undo(self.instances)
        if restored is None:
            self.statusBar().showMessage("Nothing to undo")
            return
        self._adopt(restored)
        self._log("undo")

    def do_redo(self) -> None:
        restored = self.undo.redo(self.instances)
        if restored is None:
            self.statusBar().showMessage("Nothing to redo")
            return
        self._adopt(restored)
        self._log("redo")

    def _adopt(self, instances: list[Instance]) -> None:
        self.instances = instances
        self.selected = {index for index in self.selected if index < len(instances)}
        self._active_vertex = None
        self.poly_pts = []
        self.split_pts = []
        self.brush_mask = None
        self.dirty = True
        self._refresh()

    def save_current(self) -> None:
        if self.frame is None or self.image_rgb is None:
            return
        height, width = self.image_rgb.shape[:2]
        try:
            save_correction(
                self.root,
                self.out,
                self.frame,
                self.instances,
                width,
                height,
                action="save",
                detail=f"instances={len(self.instances)}",
            )
        except SystemExit as exc:
            QMessageBox.critical(self, "Save", str(exc))
            return
        self.dirty = False
        self.from_output = True
        if self.mark_on_save.isChecked() and not self.reviewed_box.isChecked():
            self.reviewed_box.setChecked(True)
        self._update_progress()
        self._refresh()
        self.statusBar().showMessage(f"Saved {self.frame.stem} to {self.out}")

    def show_help(self) -> None:
        box = QMessageBox(self)
        box.setWindowTitle("Shortcuts")
        box.setText(HELP_TEXT)
        box.exec()

    def closeEvent(self, event):
        if not self.dirty:
            event.accept()
            return
        choice = QMessageBox.question(
            self,
            "Unsaved edits",
            "Save this frame before closing?",
            QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
            QMessageBox.Save,
        )
        if choice == QMessageBox.Cancel:
            event.ignore()
            return
        if choice == QMessageBox.Save:
            self.save_current()
        event.accept()

    def eventFilter(self, obj, event):
        if event.type() != QEvent.KeyPress:
            return False
        if QApplication.activeModalWidget() is not None:
            return False
        focused = QApplication.focusWidget()
        text_focus = isinstance(focused, (QLineEdit, QAbstractSpinBox, QComboBox))
        key = event.key()
        mods = event.modifiers()
        if text_focus and mods in (Qt.NoModifier, Qt.ShiftModifier, Qt.KeypadModifier):
            return False
        if event.isAutoRepeat() and self.dirty and key in (Qt.Key_Left, Qt.Key_Right):
            return True
        if key == Qt.Key_Left and not (mods & Qt.ControlModifier):
            if mods & Qt.ShiftModifier:
                self.jump_flagged(-1)
            else:
                self.step(-1)
            return True
        if key == Qt.Key_Right and not (mods & Qt.ControlModifier):
            if mods & Qt.ShiftModifier:
                self.jump_flagged(1)
            else:
                self.step(1)
            return True
        if key in (Qt.Key_Return, Qt.Key_Enter):
            if self.mode == "polygon":
                self._close_polygon(drop_last=False)
            elif self.mode == "split":
                self._commit_split()
            elif self.mode == "sam":
                self._accept_sam()
            return True
        if key == Qt.Key_Escape:
            self.poly_pts = []
            self.split_pts = []
            self.sam_points = []
            self.sam_mask = None
            self.brush_mask = None
            self.hover = None
            self._refresh()
            return True
        if key == Qt.Key_Backspace and self.mode == "polygon" and self.poly_pts:
            self.poly_pts.pop()
            self._refresh()
            return True
        if key in (Qt.Key_Delete, Qt.Key_Backspace):
            self.delete_selected()
            return True
        if mods & (Qt.ControlModifier | Qt.AltModifier):
            return False
        shortcuts = {
            Qt.Key_V: lambda: self.set_mode("select"),
            Qt.Key_A: lambda: self.set_mode("polygon"),
            Qt.Key_B: lambda: self.set_mode("brush"),
            Qt.Key_E: lambda: self.set_mode("eraser"),
            Qt.Key_X: lambda: self.set_mode("split"),
            Qt.Key_S: lambda: self.set_mode("sam"),
            Qt.Key_K: lambda: self.assign_class(KEYHOLE),
            Qt.Key_P: lambda: self.assign_class(PORE),
            Qt.Key_M: self.merge_selected,
            Qt.Key_I: self._prompt_id,
            Qt.Key_R: lambda: self.reviewed_box.toggle(),
            Qt.Key_G: lambda: self.copy_neighbor(-1 if not (mods & Qt.ShiftModifier) else 1, gap=True),
            Qt.Key_C: lambda: self.copy_neighbor(-1 if not (mods & Qt.ShiftModifier) else 1, gap=False),
            Qt.Key_BracketLeft: lambda: self.opacity.setValue(max(0, self.opacity.value() - 5)),
            Qt.Key_BracketRight: lambda: self.opacity.setValue(min(100, self.opacity.value() + 5)),
            Qt.Key_Minus: lambda: self.brush.setValue(max(1, self.brush.value() - 1)),
            Qt.Key_Equal: lambda: self.brush.setValue(min(40, self.brush.value() + 1)),
            Qt.Key_Question: self.show_help,
            Qt.Key_Slash: self.show_help,
        }
        action = shortcuts.get(key)
        if action is None:
            return False
        action()
        return True

    def _prompt_id(self) -> None:
        self.id_edit.setFocus()
        self.id_edit.selectAll()


def _distance(a, b) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def _pick_device(requested: str | None) -> str:
    if requested:
        return requested
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def launch(args) -> int:
    app = QApplication.instance() or QApplication([])
    app.setApplicationName("SMAXI segmentation correction")
    app.setStyle("Fusion")
    window = MainWindow(
        root=args.root,
        out=args.out,
        resume=args.resume,
        audit=args.audit,
        manifest=args.manifest,
        sam_checkpoint=args.sam_checkpoint,
        device=_pick_device(args.device),
    )
    window.show()
    return app.exec()
