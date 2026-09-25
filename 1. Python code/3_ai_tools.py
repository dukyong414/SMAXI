import importlib.util
import sys
from pathlib import Path
import re
from typing import Optional

_PACKAGE_DIR = Path(__file__).resolve().parent
_SUPPORT_DIR = _PACKAGE_DIR / "support"
if str(_SUPPORT_DIR) not in sys.path:
    sys.path.insert(0, str(_SUPPORT_DIR))


def _load_local_module(name, filename):
    if name not in sys.modules:
        path = _PACKAGE_DIR / filename
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


_load_local_module("utils", "0_utils.py")
from utils import *
import utils

try:
    _load_local_module("sam_benchmark_gui", "support/benchmarks/sam_benchmark_test.py")
    from sam_benchmark_gui import SamBenchmarkFrame
except ImportError:
    SamBenchmarkFrame = None

try:
    _load_local_module("yolo_demonstration", "support/benchmarks/yolo_benchmark_test.py")
    from yolo_demonstration import SMAXIYoloDemo
except ImportError:
    SMAXIYoloDemo = None

try:
    _load_local_module("txm_volume", "support/txm_volume.py")
    from txm_volume import HAS_OLEFILE, open_txm_volume, is_txm_path
except Exception:
    HAS_OLEFILE = False
    open_txm_volume = None

    def is_txm_path(path: str) -> bool:
        return bool(path) and str(path).lower().endswith(".txm")

try:
    from smaxi_tomocupy import (
        MIDPLANE_NSINO,
        TiffStackVolume,
        default_cor_guess,
        default_out_dir,
        discover_near_projection_h5,
        find_center_midplane,
        find_tomocupy_reconstruction,
        is_projection_h5,
        is_tiff_stack_dir,
        list_recon_tiff_stack,
        load_cor_settings,
        load_cor_settings_nsino,
        nsino_fraction_from_row,
        open_tomocupy_wizard,
        projection_detector_width,
        projection_stack_shape,
        resolve_tomocupy_exe,
        row_from_nsino_fraction,
        run_center_search_then_full,
        run_full_reconstruction,
        save_cor_settings,
        _h5_dataset_is_tomocupy_recon,
        _read_tiff_plane,
    )

    HAS_TOMOCUPY_VOLUME = True
    HAS_TOMOCUPY_RUNNER = True
    HAS_COR_WIZARD = True
except Exception:
    HAS_TOMOCUPY_VOLUME = False
    HAS_TOMOCUPY_RUNNER = False
    HAS_COR_WIZARD = False
    TiffStackVolume = None
    discover_near_projection_h5 = None
    find_tomocupy_reconstruction = None
    is_tiff_stack_dir = None
    list_recon_tiff_stack = None
    open_tomocupy_wizard = None
    is_projection_h5 = None
    run_center_search_then_full = None
    default_out_dir = None
    resolve_tomocupy_exe = None

    def _h5_dataset_is_tomocupy_recon(_h5_file):
        return False


def launch_sam_benchmark(parent):
    if SamBenchmarkFrame is None:
        messagebox.showerror("SAM Benchmark", "Could not load support/benchmarks/sam_benchmark_test.py.")
        return
    win = tk.Toplevel(parent)
    win.title("SAM Benchmark")
    win.geometry("1320x1080")
    try:
        win.state("zoomed")
    except tk.TclError:
        pass

    def close():
        win.destroy()

    SamBenchmarkFrame(win, on_back=close)


def launch_yolo_demonstration(parent):
    if SMAXIYoloDemo is None:
        messagebox.showerror("YOLO Demo", "Could not load support/benchmarks/yolo_benchmark_test.py.")
        return
    win = tk.Toplevel(parent)
    win.transient(parent)
    SMAXIYoloDemo(win)

try:
    import sam2.utils.misc
    import glob
    from PIL import Image
    import torch
    import numpy as np
    from natsort import natsorted

    _original_loader = sam2.utils.misc.load_video_frames_from_jpg_images

    def load_video_frames_patched(video_path, image_size, offload_video_to_cpu, async_loading_frames, **kwargs):
        """
        Patched loader that scans for PNG, TIF, and BMP in addition to JPG.
        Ignores extra normalization args (img_mean, img_std) as we handle basic loading here.
        """
        # 1. Scan for all common X-ray formats
        valid_exts = {".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG", ".tif", ".tiff", ".bmp"}
        frame_names = []
        
        if os.path.isdir(video_path):
            for f in os.listdir(video_path):
                ext = os.path.splitext(f)[1]
                if ext in valid_exts:
                    frame_names.append(os.path.join(video_path, f))
            

            try:
                frame_names = natsorted(frame_names)
            except:
                frame_names.sort()

        # 2. Check if empty
        if not frame_names:
            raise RuntimeError(f"no images found in {video_path} (Checked: jpg, png, tif, bmp)")

        # 3. Load Images
        frames = []
        for p in frame_names:
            img = Image.open(p).convert("RGB")
            
            # Resize if requested
            if image_size is not None:
                img = img.resize((image_size, image_size))
            frames.append(img)

        # 4. Convert to Tensor (B, C, H, W)
        # Helper to process frame to tensor
        def _to_tensor(pil_img):
            x = np.array(pil_img, dtype=np.float32) / 255.0
            x = torch.from_numpy(x).permute(2, 0, 1) # (C, H, W)
            return x

        img_tensors = [_to_tensor(f) for f in frames]
        img_tensors = torch.stack(img_tensors, dim=0).float()
        
        # Move to CPU first as per signature
        if offload_video_to_cpu:
            img_tensors = img_tensors.cpu()
            
        video_height, video_width = frames[0].height, frames[0].width
        return img_tensors, video_height, video_width

    # APPLY THE PATCH
    sam2.utils.misc.load_video_frames_from_jpg_images = load_video_frames_patched
    cuda_ok = torch.cuda.is_available()
    device_label = torch.cuda.get_device_name(0) if cuda_ok else "CPU"
    print(
        f"SAM 2 ready for segmentation | "
        f"CUDA: {cuda_ok} | GPU: {device_label}"
    )

except ImportError:
    print("WARNING: Could not patch SAM 2 (Library not found?)")
except Exception as e:
    print(f"WARNING: SAM 2 Patch failed: {e}")

# ==============================================================================
# SAM LABELER APP (MODULAR: 2D IMAGING vs 3D CT)
# ==============================================================================
try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False
    print("Warning: 'nibabel' not found. CT support disabled.")

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    print("Warning: 'h5py' not found. .h5 support disabled.")

try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False
    print("Warning: 'pyvista' not found. 3D visualization disabled.")

# Bridge for Windows: allow external tomography libs to find conda-provided DLLs.
_conda_tomo_bin = os.path.expandvars(r"%LOCALAPPDATA%\miniconda3\envs\xray_tomo\Library\bin")
if os.path.isdir(_conda_tomo_bin):
    try:
        os.add_dll_directory(_conda_tomo_bin)
    except Exception:
        pass
    if _conda_tomo_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = _conda_tomo_bin + os.pathsep + os.environ.get("PATH", "")

try:
    import tomocupy
    HAS_TOMOCUPY = True
except ImportError:
    HAS_TOMOCUPY = False

try:
    import tomopy
    HAS_TOMOPY = True
except ImportError:
    HAS_TOMOPY = False

if not HAS_TOMOCUPY and not HAS_TOMOPY:
    print("Warning: neither 'tomocupy' nor 'tomopy' is available. H5 projection reconstruction will use fallback preview.")


_CT_VOLUME_SUFFIXES = (".nii.gz", ".nii", ".h5", ".hdf5", ".txm")


def is_ct_volume_path(path: str) -> bool:
    name = os.path.basename(path).lower()
    return any(name.endswith(ext) for ext in _CT_VOLUME_SUFFIXES)


def collect_ct_volume_paths(source: str, recursive: bool = True) -> list:
    """Collect .nii / .nii.gz / .h5 / .txm volumes and Tomocupy TIFF stacks from a folder."""
    if not source:
        return []
    source = os.path.abspath(source)
    if os.path.isfile(source) and is_ct_volume_path(source):
        return [source]
    if not os.path.isdir(source):
        return []

    found = set()
    if recursive:
        for root, _, names in os.walk(source):
            if HAS_TOMOCUPY_VOLUME and is_tiff_stack_dir(root):
                found.add(root)
            for name in names:
                full = os.path.join(root, name)
                if is_ct_volume_path(full):
                    found.add(full)
                    if HAS_TOMOCUPY_VOLUME and full.lower().endswith((".h5", ".hdf5")):
                        alt = find_tomocupy_reconstruction(full)
                        if alt and alt[0] != full:
                            found.add(alt[0])
    else:
        if HAS_TOMOCUPY_VOLUME and is_tiff_stack_dir(source):
            found.add(source)
        for name in os.listdir(source):
            full = os.path.join(source, name)
            if os.path.isfile(full) and is_ct_volume_path(full):
                found.add(full)
    return natsorted(found)


def _sync_ml_flags_from_utils():
    """Refresh HAS_* copied by ``from utils import *`` after lazy init."""
    g = globals()
    g["HAS_SAM"] = utils.HAS_SAM
    g["HAS_YOLO"] = utils.HAS_YOLO
    g["HAS_NIBABEL"] = utils.HAS_NIBABEL or g.get("HAS_NIBABEL", False)
    g["HAS_PYVISTA"] = utils.HAS_PYVISTA or g.get("HAS_PYVISTA", False)


class SamLabelerApp:
    def __init__(self, parent_frame, on_back):
        utils.init_ml_dependencies(verbose=True)
        utils.init_viz_dependencies(verbose=True)
        _sync_ml_flags_from_utils()
        import torch

        self.parent = parent_frame
        self.on_back_callback = on_back
        
        # --- Data Variables ---
        self.image_list = []        
        self.current_idx = 0        
        self.raw_image = None       
        self.predictor = None
        self.current_mask = None
        self.device = "cuda" if (torch.cuda.is_available() and HAS_SAM) else "cpu"
        self.checkpoint = utils.SAM_CHECKPOINT_DEFAULT
        
        # --- Segmentation region limits (excluded zones) ---
        self.y_limit = None
        self.setting_limit_mode = False
        self.x_right_limit = None
        self.setting_x_right_limit_mode = False
        self.crosshair_lines = []
        
        # --- CT Specific Variables ---
        self.is_ct_mode = False
        self.ct_volume = None       
        self.ct_dims = (0,0,0)      
        self.slice_axis = tk.StringVar(value="Z") 
        self.current_slice_idx = 0  
        self.plotter = None 
        self.recon_mask_volume = None
        self.recon_label_volume = None
        self.recon_axis = "Z"
        self.recon_bounds = None
        self.recon_slab_lower = 0
        self.ct_preview_step = 1
        self.ct_label_volume_preview = None
        self.show_seg_in_3d = tk.BooleanVar(value=True)
        self.show_cross_sections_3d = tk.BooleanVar(value=False)
        self.pv_view_mode = tk.StringVar(value="Volume + surface")
        self.pv_pore_highlight = tk.BooleanVar(value=True)
        self.pv_bg_dark = tk.BooleanVar(value=True)  # default dark — easier to see CT
        self.pv_contrast_boost = tk.DoubleVar(value=1.15)
        self.pv_window_lo = tk.DoubleVar(value=0.5)   # percentile
        self.pv_window_hi = tk.DoubleVar(value=99.5)
        self._pv_vol_actor = None
        self._pv_iso_signs = (1, 1, 1)
        self._pv_launch_busy = False
        self._PV_MAX_VOXELS = 2_500_000
        self.current_ct_dataset_key = "N/A"
        self.h5_info_var = tk.StringVar(value="Volume metadata will appear here after loading .h5 / .txm / .nii.")
        self.is_h5_fast_volume = False
        self.is_h5_lazy_volume = False
        self.txm_handle = None
        self.h5_sampling_mode = tk.StringVar(value="Level 3 - Balanced")
        self.h5_slice_mode = tk.StringVar(value="Full-res slices (lazy I/O)")
        self.h5_recon_backend = "N/A"
        self.recon_backend_var = tk.StringVar(value="Backend: not loaded")
        self.tomocupy_source_h5 = None
        self._tomocupy_running = False
        self.tomocupy_cor_var = tk.StringVar(value="")
        self.tomocupy_cor_hint_var = tk.StringVar(
            value="Scroll COR search plane (detector row), detect COR, then full recon."
        )
        self.cor_plane_info_var = tk.StringVar(value="COR plane: —")
        self.cor_plane_row = tk.IntVar(value=0)
        self.projection_cor_tune_mode = False
        self._cor_n_rows = 0
        self._cor_n_proj = 0
        self._cor_preview_proj_idx = 0
        self._cor_preview_after_id = None
        self._cor_trace_guard = False
        
        # --- State Variables ---
        self.mode = tk.StringVar(value="ai") 
        self.label_preset_mode = tk.StringVar(value="lpbf") # Default to LPBF
        self.new_class_entry_var = tk.StringVar()
        self.auto_ai_var = tk.BooleanVar(value=False)
        # Prefer compact SAM masks for pores/bubbles; also tunes Auto-AI thresholds
        self.small_bubble_mode = tk.BooleanVar(value=False)
        self.undo_stack = []
        self.class_mapping = {} 
        self.class_colors = {}
        self.current_class_id = tk.IntVar(value=-1)
        
        # --- Canvas Vars ---
        self.zoom_level = 1.0; self.pan_x = 0; self.pan_y = 0
        self.scale = 1.0; self.off_x = 0; self.off_y = 0
        self.input_points = []; self.input_labels = []; self.poly_points = []
        self.current_annotations = []; self.selected_annotation_index = -1
        self.is_ai_ready = False
        
        # --- ADD THESE VARIABLES ---
        self.input_box = None
        self.drawing_box = False
        self.box_start = (0, 0)
        self.box_end = (0, 0)
        self.prompt_mode = tk.StringVar(value="box") # 'point' or 'box'

        self.PALETTE = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), 
            (255, 0, 255), (0, 255, 255), (255, 128, 0), (128, 0, 255),
            (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 0)
        ]
        self.LPBF_PRESETS = ["Keyhole", "KH-Pore", "Gas-Pore", "Bubble", "Spatter", "Plume"]

        self.show_landing_page()
        if not HAS_SAM:
            messagebox.showerror(
                "Error",
                "SAM library not found.\n\n"
                "From the '1. Root folder' directory run:\n"
                "  pip install -r requirements.txt\n\n"
                "Requires: torch, segment-anything, and sam2.",
            )
        

            
    # ==========================================================================
    # 1. LANDING PAGE (SELECTION MENU)
    # ==========================================================================
    def show_landing_page(self):
        # Clear frame
        for widget in self.parent.winfo_children(): widget.destroy()
        
        self.frame = tk.Frame(self.parent, bg="#ECF0F1")
        self.frame.pack(fill="both", expand=True)

        # Header
        tk.Button(self.frame, text="← Back to Main Menu", command=self.go_back_app, font=("Arial", 12), bg="#95A5A6", fg="white").pack(anchor="nw", padx=20, pady=20)
        
        center_frame = tk.Frame(self.frame, bg="#ECF0F1")
        center_frame.pack(expand=True)
        
        tk.Label(center_frame, text="Select type of x-ray image analysis", font=("Helvetica", 24, "bold"), bg="#ECF0F1", fg="#2C3E50").pack(pady=(0, 12))

        tk.Button(
            center_frame,
            text="SAM model benchmark →",
            font=("Arial", 10),
            bg="#D5DBDB",
            fg="#2C3E50",
            cursor="hand2",
            relief="groove",
            padx=12,
            pady=4,
            command=lambda: launch_sam_benchmark(self.parent),
        ).pack(pady=(0, 28))
        
        # Button 1: 2D Annotation
        btn_2d = tk.Button(center_frame, text="1. 2D Annotation\n(High-speed imaging)", font=("Arial", 16, "bold"), 
                           bg="#3498DB", fg="white", width=25, height=5, cursor="hand2",
                           command=self.setup_ui_2d)
        btn_2d.pack(pady=10)
        
        # Button 2: 3D Annotation (CT + Reconstruction)
        btn_3d = tk.Button(center_frame, text="2. 3D Annotation\n(CT + SAM propagation)", font=("Arial", 16, "bold"), 
                           bg="#E67E22", fg="white", width=25, height=5, cursor="hand2",
                           command=self.setup_ui_4d)
        btn_3d.pack(pady=10)

    # ==========================================================================
    # 2. UI SETUP: 2D IMAGING MODE
    # ==========================================================================
    def setup_ui_2d(self):
        # --- RESET STATE ---
        self.image_list = []
        self.raw_image = None
        self.ct_volume = None
        self.current_idx = 0
        self.is_ct_mode = False
        
        self._build_base_ui(title="Tool: X-ray Imaging (2D)", color="#3498DB")
        
        # ======================================================================
        # TOOLBAR LAYOUT (CLEAN VERSION)
        # ======================================================================

# 1. RIGHT SIDE: Auto AI / small-bubble tools
        f_right = tk.Frame(self.toolbar, bg="#D6EAF8")
        f_right.pack(side="right", padx=10)
        tk.Checkbutton(
            f_right,
            text="🫧 Small bubbles",
            variable=self.small_bubble_mode,
            indicatoron=0,
            bg="#5D6D7E",
            fg="white",
            selectcolor="#16A085",
            font=("Arial", 9, "bold"),
            width=14,
            height=1,
            command=self._on_small_bubble_mode_toggle,
        ).pack(side="left", padx=4)
        tk.Button(
            f_right,
            text="Auto-segment",
            command=self.auto_segment_2d,
            bg="#1ABC9C",
            fg="white",
            font=("Arial", 9, "bold"),
        ).pack(side="left", padx=4)
        tk.Checkbutton(f_right, text="⚡ Auto-Ready", variable=self.auto_ai_var, 
                       indicatoron=0,
                       bg="#8E44AD", fg="white", selectcolor="#2ECC71", # Purple=OFF, Green=ON
                       font=("Arial", 10, "bold"), width=12, height=1,
                       # If clicked, run embedding on the current image immediately
                       command=lambda: self.run_embedding_thread() if self.auto_ai_var.get() else None
                       ).pack(side="left", padx=5)

        # 2. Class presets on scrollable row 2 (main toolbar row 1 stays visible)
        scroll_row = self._ensure_toolbar_scroll_row()
        self.f_classes_container = tk.Frame(scroll_row, bg="#D6EAF8")
        self.f_classes_container.pack(side="left", padx=5)

        self.update_presets_2d()

        # ======================================================================
        # MAIN LAYOUT
        # ======================================================================
        main_frame = tk.Frame(self.frame)
        main_frame.pack(fill="both", expand=True)

        self.sidebar = tk.Frame(main_frame, width=280, bg="#ECF0F1")
        self.sidebar.pack(side="right", fill="y")
        self.sidebar.pack_propagate(False)
        
        # LEGEND 
        self.legend_frame = tk.LabelFrame(self.sidebar, text="Classes (Click to Select)", font=("Arial", 10, "bold"), bg="#ECF0F1", height=200)
        self.legend_frame.pack(fill="x", padx=5, pady=5)
        
        lbl_list = tk.Label(self.sidebar, text="File List", bg="#BDC3C7", font=("Arial", 10, "bold"))
        lbl_list.pack(fill="x", pady=(5,0))
        
        sb = tk.Scrollbar(self.sidebar)
        sb.pack(side="right", fill="y")
        self.file_listbox = tk.Listbox(self.sidebar, font=("Consolas", 9), yscrollcommand=sb.set, selectmode="browse")
        self.file_listbox.pack(side="left", fill="both", expand=True)
        sb.config(command=self.file_listbox.yview)
        self.file_listbox.bind("<<ListboxSelect>>", self.on_file_select)

        self.canvas_container = tk.Frame(main_frame, bg="#2C3E50")
        self.canvas_container.pack(side="left", fill="both", expand=True)
        self.canvas = tk.Canvas(self.canvas_container, bg="black", cursor="cross", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        self._bind_events()
        self.update_presets_2d()
        self.frame.after_idle(self._refresh_toolbar_scroll)

    def update_presets_2d(self):
        # Clear existing widgets in container
        for w in self.f_classes_container.winfo_children(): w.destroy()
        
        # 1. "New Class" Entry
        tk.Label(self.f_classes_container, text="New Class:", bg="#D6EAF8", font=("Arial", 9)).pack(side="left")
        ent = tk.Entry(self.f_classes_container, textvariable=self.new_class_entry_var, width=12)
        ent.pack(side="left", padx=2)
        ent.bind("<Return>", lambda e: self.add_from_entry())
        tk.Button(self.f_classes_container, text="+", command=self.add_from_entry, bg="#2ECC71", width=3, cursor="hand2").pack(side="left", padx=(0, 8))

        # Quick presets (click = create/select class)
        for preset in self.LPBF_PRESETS:
            tk.Button(
                self.f_classes_container,
                text=preset,
                command=lambda p=preset: self.add_class_by_name(p),
                bg="#AED6F1",
                font=("Arial", 8),
                cursor="hand2",
            ).pack(side="left", padx=2)

        # 2. Active Class Indicator
        tk.Label(self.f_classes_container, text="|  Active:", bg="#D6EAF8", font=("Arial", 9)).pack(side="left")
        self.lbl_active_class = tk.Label(self.f_classes_container, text="None", font=("Arial", 10, "bold"), bg="#D6EAF8", fg="blue")
        self.lbl_active_class.pack(side="left", padx=5)

        # 3. Manage Button
        tk.Button(self.f_classes_container, text="⚙ Manage", command=self.manage_active_class, 
                  bg="#95A5A6", fg="white", font=("Arial", 8), cursor="hand2").pack(side="left", padx=5)

        self.refresh_class_ui()
        self.frame.after_idle(self._refresh_toolbar_scroll)

    def auto_segment_2d(self):
        """Automatically generates masks for the entire image using SAM."""
        if not self.is_ai_ready or self.raw_image is None:
            messagebox.showwarning("AI Not Ready", "Please load an image and click '⚡ Run AI' (yellow button) first to load the model embeddings.")
            return

        # 1. Check if class is selected
        if self.current_class_id.get() == -1:
            messagebox.showwarning("No Class", "Please select a Class (e.g., Bubble, Pore) from the toolbar before running Auto AI.")
            return

        small = bool(self.small_bubble_mode.get())
        self.log(
            "Running Auto-Segment for small bubbles..." if small else "Running Auto-Segment...",
            "orange",
        )
        self.frame.update_idletasks()

        try:
            # 2. Initialize Generator
            from segment_anything import SamAutomaticMaskGenerator
            if small:
                mask_generator = SamAutomaticMaskGenerator(
                    self.predictor.model,
                    points_per_side=48,
                    pred_iou_thresh=0.80,
                    stability_score_thresh=0.85,
                    crop_n_layers=1,
                    crop_n_points_downscale_factor=2,
                    min_mask_region_area=10,
                )
                min_contour = 8
                max_frac = 0.08  # reject huge blobs when targeting bubbles
            else:
                mask_generator = SamAutomaticMaskGenerator(
                    self.predictor.model,
                    points_per_side=32,
                    pred_iou_thresh=0.86,
                    stability_score_thresh=0.92,
                    crop_n_layers=0,
                    crop_n_points_downscale_factor=1,
                    min_mask_region_area=100,
                )
                min_contour = 50
                max_frac = 1.0
            
            # 3. Generate Masks
            masks = mask_generator.generate(self.raw_image)
            
            if not masks:
                self.log("No objects detected.", "red")
                return

            self.save_state_for_undo()
            
            # 4. Convert Masks to Polygons
            h, w = self.raw_image.shape[:2]
            img_area = float(h * w)
            added_count = 0
            
            for res in masks:

                m = res['segmentation'].astype(np.uint8) * 255
                m = (self.apply_y_limit_constraint(m.astype(bool)) * 255).astype(np.uint8)
                if small:
                    m = (self._refine_small_bubble_mask(m.astype(bool)).astype(np.uint8) * 255)

                # Find contours
                cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for c in cnts:
                    area = cv2.contourArea(c)
                    if area < min_contour:
                        continue
                    if area > max_frac * img_area:
                        continue
                    
                    # Normalize coordinates (0.0 - 1.0)
                    norm_coords = [x for pt in c.reshape(-1, 2) for x in (pt[0]/w, pt[1]/h)]
                    
                    # Add to annotations list
                    self.current_annotations.append({
                        'id': self.current_class_id.get(), 
                        'coords': norm_coords
                    })
                    added_count += 1

            # 5. Refresh UI
            self.save_all_annotations_to_file(silent=True)
            self.show_image(self.raw_image)
            self.log(f"Auto AI: Added {added_count} objects", "green")
            
        except Exception as e:
            self.log(f"Auto AI Failed: {e}", "red")
            print(f"Auto AI Error: {e}")



    def _make_ct_left_scroll_area(self, parent):
        """Scrollable host for the tomography tool left sidebar."""
        scroll_host = tk.Frame(parent, bg="white")
        scroll_host.pack(fill="both", expand=True)

        canvas = tk.Canvas(scroll_host, highlightthickness=0, bg="white")
        scrollbar = tk.Scrollbar(scroll_host, orient="vertical", command=canvas.yview)
        inner = tk.Frame(canvas, bg="white")

        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        window_id = canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(window_id, width=e.width))
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        def _on_wheel(event):
            if getattr(event, "delta", 0):
                canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
            elif event.num == 4:
                canvas.yview_scroll(-1, "units")
            elif event.num == 5:
                canvas.yview_scroll(1, "units")

        for widget in (scroll_host, canvas, inner):
            widget.bind("<MouseWheel>", _on_wheel)
            widget.bind("<Button-4>", _on_wheel)
            widget.bind("<Button-5>", _on_wheel)

        self._ct_left_canvas = canvas
        return inner

    # ==========================================================================
    # 3. UI SETUP: 3D CT MODE
    # ==========================================================================

    def setup_ui_ct(self):
        # --- RESET STATE (Prevents 2D images from showing here) ---
        self.image_list = []
        self.raw_image = None
        self.ct_volume = None
        self.current_idx = 0
        self.is_ct_mode = True
        
        self._build_base_ui(title="Tool: X-ray Tomography (3D)", color="#E67E22")

        f_right = tk.Frame(self.toolbar, bg="#D6EAF8")
        f_right.pack(side="right", padx=10)
        tk.Checkbutton(
            f_right,
            text="🫧 Small pores",
            variable=self.small_bubble_mode,
            indicatoron=0,
            bg="#5D6D7E",
            fg="white",
            selectcolor="#16A085",
            font=("Arial", 9, "bold"),
            width=12,
            height=1,
            command=self._on_small_bubble_mode_toggle,
        ).pack(side="left", padx=4)
        tk.Checkbutton(
            f_right,
            text="⚡ Auto-Ready",
            variable=self.auto_ai_var,
            indicatoron=0,
            bg="#8E44AD",
            fg="white",
            selectcolor="#2ECC71",
            font=("Arial", 10, "bold"),
            width=12,
            height=1,
            command=lambda: self.run_embedding_thread() if self.auto_ai_var.get() else None,
        ).pack(side="left", padx=5)

        scroll_row = self._ensure_toolbar_scroll_row()
        self.f_classes_container = tk.Frame(scroll_row, bg="#D6EAF8")
        self.f_classes_container.pack(side="left", padx=5)
        self.update_presets_2d()
        
        # Split Layout
        self.main_split = tk.PanedWindow(self.frame, orient=tk.HORIZONTAL, sashwidth=5, bg="#BDC3C7")
        self.main_split.pack(fill="both", expand=True)

        # LEFT (scrollable)
        self.left_panel = tk.Frame(self.main_split, bg="white", width=400)
        self.main_split.add(self.left_panel, minsize=100)
        left = self._make_ct_left_scroll_area(self.left_panel)
        self.left_panel_inner = left
        
        tk.Label(left, text="3D Controls", font=("Arial", 16, "bold"), bg="white", fg="#E67E22").pack(pady=10)
        
        f_3d = tk.LabelFrame(left, text="3D View", bg="white", padx=10, pady=10, font=("Arial", 12, "bold"))
        f_3d.pack(fill="x", padx=10)
        
        self.threshold_val = tk.IntVar(value=100)
        self.opacity_val = tk.DoubleVar(value=1.0)

        tk.Label(f_3d, text="3D display mode", bg="white", font=("Arial", 11, "bold")).pack(anchor="w")
        tk.OptionMenu(
            f_3d,
            self.pv_view_mode,
            "Volume + surface",
            "3D Reconstruction",
            "Pore detail (volume)",
            "Volume render",
            command=lambda _v: self.update_isosurface_3d(),
        ).pack(fill="x", pady=(0, 6))
        
        tk.Label(f_3d, text="Reconstruction threshold", bg="white", font=("Arial", 12, "bold")).pack(anchor="w")
        tk.Scale(f_3d, from_=0, to=255, orient=tk.HORIZONTAL, variable=self.threshold_val, bg="white", 
                 command=lambda x: self.update_isosurface_3d(), font=("Arial", 11)).pack(fill="x")
        
        tk.Label(f_3d, text="Surface / volume opacity", bg="white", font=("Arial", 12, "bold")).pack(anchor="w", pady=(5,0))
        tk.Scale(f_3d, from_=0.0, to=1.0, resolution=0.05, orient=tk.HORIZONTAL, variable=self.opacity_val, bg="white", 
                 command=lambda x: self.update_isosurface_3d(), font=("Arial", 11)).pack(fill="x")

        tk.Button(f_3d, text="Open 3D Window", command=self.launch_pyvista_window, bg="#2ECC71", fg="white", font=("Arial", 12, "bold")).pack(fill="x", pady=(10, 4))
        tk.Checkbutton(
            f_3d,
            text="Black background (uncheck = white)",
            variable=self.pv_bg_dark,
            bg="white",
            font=("Arial", 10),
            command=self._apply_pv_background,
        ).pack(anchor="w")
        tk.Checkbutton(
            f_3d,
            text="Highlight bright pores in 3D",
            variable=self.pv_pore_highlight,
            bg="white",
            font=("Arial", 10),
            command=lambda: self.update_isosurface_3d(),
        ).pack(anchor="w")
        tk.Checkbutton(
            f_3d,
            text="Show mid cross-sections (optional overlay)",
            variable=self.show_cross_sections_3d,
            bg="white",
            font=("Arial", 10),
            command=lambda: self.update_isosurface_3d(),
        ).pack(anchor="w")
        tk.Checkbutton(
            f_3d,
            text="Show saved / propagated pores in 3D",
            variable=self.show_seg_in_3d,
            bg="white",
            font=("Arial", 10),
            command=lambda: self.update_isosurface_3d(),
        ).pack(anchor="w")

        tk.Label(f_3d, text="3D contrast / window", bg="white", font=("Arial", 11, "bold")).pack(anchor="w", pady=(8, 0))
        tk.Label(f_3d, text="Low % (raise to darken air/pores)", bg="white", fg="#566573", font=("Arial", 8)).pack(anchor="w")
        tk.Scale(
            f_3d, from_=0.0, to=30.0, resolution=0.5, orient=tk.HORIZONTAL,
            variable=self.pv_window_lo, bg="white", font=("Arial", 9),
        ).pack(fill="x")
        tk.Label(f_3d, text="High % (lower to brighten metal)", bg="white", fg="#566573", font=("Arial", 8)).pack(anchor="w")
        tk.Scale(
            f_3d, from_=70.0, to=100.0, resolution=0.5, orient=tk.HORIZONTAL,
            variable=self.pv_window_hi, bg="white", font=("Arial", 9),
        ).pack(fill="x")
        tk.Label(f_3d, text="Contrast boost (gamma)", bg="white", fg="#566573", font=("Arial", 8)).pack(anchor="w")
        tk.Scale(
            f_3d, from_=0.6, to=2.5, resolution=0.05, orient=tk.HORIZONTAL,
            variable=self.pv_contrast_boost, bg="white", font=("Arial", 9),
        ).pack(fill="x")
        tk.Button(
            f_3d,
            text="Apply contrast to 3D view",
            command=self._refresh_pv_contrast,
            bg="#5D6D7E",
            fg="white",
            font=("Arial", 9, "bold"),
        ).pack(fill="x", pady=(4, 2))

        tk.Button(
            f_3d,
            text="Refresh pore overlay",
            command=self._rebuild_ct_label_volume_preview,
            bg="#AED6F1",
            font=("Arial", 9),
        ).pack(fill="x", pady=(2, 6))

        tk.Label(f_3d, text="Isometric camera", bg="white", font=("Arial", 11, "bold")).pack(anchor="w", pady=(4, 2))
        tk.Label(
            f_3d,
            text="Preset corners + 90° spins (3D window must be open)",
            bg="white",
            fg="#566573",
            font=("Arial", 8),
        ).pack(anchor="w")
        f_iso = tk.Frame(f_3d, bg="white")
        f_iso.pack(fill="x", pady=(2, 4))
        iso_presets = (
            ("+X+Y+Z", 1, 1, 1),
            ("-X+Y+Z", -1, 1, 1),
            ("+X-Y+Z", 1, -1, 1),
            ("-X-Y+Z", -1, -1, 1),
            ("+X+Y-Z", 1, 1, -1),
            ("-X+Y-Z", -1, 1, -1),
            ("+X-Y-Z", 1, -1, -1),
            ("-X-Y-Z", -1, -1, -1),
        )
        for col, (label, sx, sy, sz) in enumerate(iso_presets):
            r, c = divmod(col, 4)
            tk.Button(
                f_iso,
                text=label,
                command=lambda x=sx, y=sy, z=sz: self.set_pv_isometric_view(x, y, z),
                bg="#EBF5FB",
                font=("Consolas", 8),
                width=8,
            ).grid(row=r, column=c, padx=1, pady=1, sticky="ew")
        for c in range(4):
            f_iso.grid_columnconfigure(c, weight=1)

        f_iso_rot = tk.Frame(f_3d, bg="white")
        f_iso_rot.pack(fill="x", pady=(0, 8))
        tk.Label(f_iso_rot, text="Rotate 90°:", bg="white", font=("Arial", 9, "bold")).pack(side="left")
        for axis in ("X", "Y", "Z"):
            tk.Button(
                f_iso_rot,
                text=f"↻ {axis}",
                command=lambda a=axis: self.rotate_pv_camera_90(a),
                bg="#D5DBDB",
                font=("Arial", 9, "bold"),
                width=5,
            ).pack(side="left", padx=2)
        tk.Button(
            f_iso_rot,
            text="Flip X",
            command=lambda: self.flip_pv_isometric_axis("x"),
            bg="#FADBD8",
            font=("Arial", 8),
        ).pack(side="left", padx=(8, 2))
        tk.Button(
            f_iso_rot,
            text="Flip Y",
            command=lambda: self.flip_pv_isometric_axis("y"),
            bg="#FADBD8",
            font=("Arial", 8),
        ).pack(side="left", padx=2)
        tk.Button(
            f_iso_rot,
            text="Flip Z",
            command=lambda: self.flip_pv_isometric_axis("z"),
            bg="#FADBD8",
            font=("Arial", 8),
        ).pack(side="left", padx=2)

        f_h5 = tk.LabelFrame(left, text="Volume / H5 / TXM Settings", bg="white", padx=10, pady=10, font=("Arial", 12, "bold"))
        f_h5.pack(fill="x", padx=10, pady=(0, 10))
        tk.Label(f_h5, text="Slice view mode", bg="white", fg="#2C3E50", font=("Arial", 11, "bold")).pack(anchor="w")
        tk.OptionMenu(
            f_h5,
            self.h5_slice_mode,
            "Full-res slices (lazy I/O)",
            "Downsampled preview (RAM)",
            command=lambda _v: self.refresh_current_ct_slice(),
        ).pack(fill="x", pady=(2, 8))
        tk.Label(f_h5, text="3D preview / recon resolution", bg="white", fg="#2C3E50", font=("Arial", 11, "bold")).pack(anchor="w")
        tk.OptionMenu(
            f_h5,
            self.h5_sampling_mode,
            "Level 1 - Fastest (Highest Downsampling)",
            "Level 2 - Fast",
            "Level 3 - Balanced",
            "Level 4 - High Resolution",
            "Level 5 - Ultra Resolution (Lowest Downsampling)",
            command=lambda _v: self.refresh_current_ct_slice()
        ).pack(fill="x", pady=(2, 8))
        tk.Label(
            f_h5,
            textvariable=self.h5_info_var,
            justify="left",
            anchor="w",
            bg="white",
            fg="#2C3E50",
            font=("Consolas", 10),
            wraplength=360
        ).pack(fill="x")
        self.f_tomocupy_cor = tk.LabelFrame(
            f_h5,
            text="Center of rotation (COR)",
            bg="white",
            padx=8,
            pady=8,
            font=("Arial", 10, "bold"),
        )
        tk.Label(
            self.f_tomocupy_cor,
            textvariable=self.tomocupy_cor_hint_var,
            bg="white",
            fg="#566573",
            font=("Arial", 9),
            wraplength=340,
            justify="left",
        ).pack(anchor="w")
        cor_row = tk.Frame(self.f_tomocupy_cor, bg="white")
        cor_row.pack(fill="x", pady=(6, 4))
        tk.Label(cor_row, text="COR (px):", bg="white", font=("Arial", 10, "bold")).pack(
            side="left"
        )
        self.entry_tomocupy_cor = tk.Entry(
            cor_row,
            textvariable=self.tomocupy_cor_var,
            font=("Consolas", 11),
            width=14,
        )
        self.entry_tomocupy_cor.pack(side="left", padx=(6, 0))
        tk.Label(
            self.f_tomocupy_cor,
            text="COR search plane (detector row — scroll to find sharpest COR):",
            bg="white",
            font=("Arial", 9, "bold"),
            wraplength=340,
            justify="left",
        ).pack(anchor="w", pady=(8, 2))
        plane_row = tk.Frame(self.f_tomocupy_cor, bg="white")
        plane_row.pack(fill="x")
        self.cor_plane_scale = tk.Scale(
            plane_row,
            from_=0,
            to=0,
            orient=tk.VERTICAL,
            length=140,
            variable=self.cor_plane_row,
            command=self._on_cor_plane_slider,
            bg="white",
            highlightthickness=0,
        )
        self.cor_plane_scale.pack(side="left", padx=(0, 8))
        plane_txt = tk.Frame(plane_row, bg="white")
        plane_txt.pack(side="left", fill="x", expand=True)
        tk.Label(
            plane_txt,
            textvariable=self.cor_plane_info_var,
            bg="white",
            fg="#2C3E50",
            font=("Consolas", 9),
            wraplength=260,
            justify="left",
        ).pack(anchor="w")
        tk.Label(
            plane_txt,
            text="Tip: mouse wheel on image scrolls plane.\n"
            "Main view = sinogram at selected row.\n"
            "Green line = COR (px); updates as you type.",
            bg="white",
            fg="#566573",
            font=("Arial", 8),
            justify="left",
        ).pack(anchor="w", pady=(6, 0))
        cor_btn_row = tk.Frame(self.f_tomocupy_cor, bg="white")
        cor_btn_row.pack(fill="x", pady=(8, 0))
        self.btn_detect_cor = tk.Button(
            cor_btn_row,
            text="Detect COR at this plane",
            command=self._on_detect_cor_at_plane,
            bg="#2980B9",
            fg="white",
            font=("Arial", 9, "bold"),
        )
        self.btn_detect_cor.pack(side="left", fill="x", expand=True, padx=(0, 4))
        self.btn_save_cor = tk.Button(
            cor_btn_row,
            text="Save COR",
            command=self._on_save_cor_clicked,
            bg="#95A5A6",
            fg="white",
            font=("Arial", 9, "bold"),
            width=10,
        )
        self.btn_save_cor.pack(side="left")
        self.f_tomocupy_cor.pack_forget()

        self.btn_open_cor_wizard = tk.Button(
            f_h5,
            text="Open Tomocupy workbench…",
            command=self._on_open_cor_wizard_clicked,
            bg="#8E44AD",
            fg="white",
            font=("Arial", 10, "bold"),
            wraplength=340,
        )
        self.btn_open_cor_wizard.pack(fill="x", pady=(8, 0))
        self.btn_open_cor_wizard.pack_forget()
        self.btn_run_tomocupy = tk.Button(
            f_h5,
            text="Full recon (inline, uses COR above)",
            command=self._on_run_tomocupy_clicked,
            bg="#117A65",
            fg="white",
            font=("Arial", 9),
            wraplength=340,
        )
        self.btn_run_tomocupy.pack(fill="x", pady=(4, 0))
        self.btn_run_tomocupy.pack_forget()

        f_vol = tk.LabelFrame(left, text="Volume Files", bg="white", padx=6, pady=6, font=("Arial", 11, "bold"))
        f_vol.pack(fill="x", padx=10, pady=(0, 10))
        vol_wrap = tk.Frame(f_vol, bg="white")
        vol_wrap.pack(fill="x")
        vol_sb = tk.Scrollbar(vol_wrap)
        vol_sb.pack(side="right", fill="y")
        self.file_listbox = tk.Listbox(
            vol_wrap,
            font=("Consolas", 9),
            yscrollcommand=vol_sb.set,
            selectmode="browse",
            bg="white",
            height=8,
        )
        self.file_listbox.pack(side="left", fill="x", expand=True)
        vol_sb.config(command=self.file_listbox.yview)
        self.file_listbox.bind("<<ListboxSelect>>", self.on_file_select)

        # Class list (required to select a label before saving with S)
        self.legend_frame = tk.LabelFrame(
            left,
            text="Classes (Click to Select)",
            font=("Arial", 11, "bold"),
            bg="white",
            padx=6,
            pady=6,
        )
        self.legend_frame.pack(fill="x", padx=10, pady=(0, 10))
        self.refresh_class_ui()
        
        # RIGHT
        self.right_panel = tk.Frame(self.main_split, bg="#2C3E50")
        self.main_split.add(self.right_panel, minsize=400)

        f_backend = tk.Frame(self.right_panel, bg="#2C3E50")
        f_backend.pack(side="top", fill="x", padx=6, pady=(6, 2))
        self.btn_recon_backend = tk.Button(
            f_backend,
            textvariable=self.recon_backend_var,
            bg="#5D6D7E",
            fg="white",
            font=("Arial", 11, "bold"),
            relief="raised",
            state="disabled",
            disabledforeground="white"
        )
        self.btn_recon_backend.pack(fill="x")

        self.canvas_container = tk.Frame(self.right_panel, bg="#2C3E50")
        self.canvas_container.pack(side="top", fill="both", expand=True)
        self.canvas = tk.Canvas(self.canvas_container, bg="black", cursor="cross", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        self.ct_ctrl_frame = tk.Frame(self.right_panel, bg="#34495E", pady=5)
        self.ct_ctrl_frame.pack(side="bottom", fill="x")
        
        f_axes = tk.Frame(self.ct_ctrl_frame, bg="#34495E")
        f_axes.pack(side="top", fill="x", pady=2)
        tk.Label(f_axes, text="Cross-section", fg="#F39C12", bg="#34495E", font=("Arial", 12, "bold")).pack(side="left", padx=10)
        # Values stay X/Y/Z for label file compatibility; labels show TomoBank planes.
        # For volume shape (Z,Y,X): X→XY, Y→XZ, Z→YZ
        tk.Radiobutton(f_axes, text="XY", variable=self.slice_axis, value="X", command=self.on_axis_change, bg="#34495E", fg="white", selectcolor="#2C3E50", font=("Arial", 11, "bold")).pack(side="left")
        tk.Radiobutton(f_axes, text="XZ", variable=self.slice_axis, value="Y", command=self.on_axis_change, bg="#34495E", fg="white", selectcolor="#2C3E50", font=("Arial", 11, "bold")).pack(side="left")
        tk.Radiobutton(f_axes, text="YZ", variable=self.slice_axis, value="Z", command=self.on_axis_change, bg="#34495E", fg="white", selectcolor="#2C3E50", font=("Arial", 11, "bold")).pack(side="left")
        tk.Label(f_axes, text="(TomoBank Z,Y,X)", fg="#AED6F1", bg="#34495E", font=("Arial", 9)).pack(side="left", padx=8)
        
        f_slide = tk.Frame(self.ct_ctrl_frame, bg="#34495E")
        f_slide.pack(side="top", fill="x")
        tk.Label(f_slide, text="Slice", fg="white", bg="#34495E", font=("Arial", 12, "bold")).pack(side="left", padx=5)
        self.slice_slider = tk.Scale(f_slide, from_=0, to=100, orient=tk.HORIZONTAL, command=self.on_slider_change, bg="#34495E", fg="white", length=400, font=("Arial", 10))
        self.slice_slider.pack(side="left", fill="x", expand=True, padx=10)

        self._bind_events()
        self.frame.after_idle(self._refresh_toolbar_scroll)

    def _set_recon_backend_status(self, text, color="#5D6D7E"):
        self.recon_backend_var.set(text)
        if hasattr(self, "btn_recon_backend") and self.btn_recon_backend.winfo_exists():
            self.btn_recon_backend.config(bg=color, activebackground=color)
        

    def _build_base_ui(self, title, color):
        for widget in self.parent.winfo_children(): widget.destroy()
        self.frame = tk.Frame(self.parent)
        self.frame.pack(fill="both", expand=True)
        
        # Header
        top = tk.Frame(self.frame, bg="#EAF2F8", height=50)
        top.pack(fill="x")
        tk.Button(top, text="← Menu", command=self.show_landing_page, bg="#95A5A6", fg="white").pack(side="left", padx=10, pady=10)
        tk.Label(top, text=title, bg="#EAF2F8", fg=color, font=("Arial", 16, "bold")).pack(side="left", padx=10)
        self.lbl_status = tk.Label(top, text="Status: Idle", bg="#EAF2F8", fg="gray", font=("Arial", 10, "bold"))
        self.lbl_status.pack(side="right", padx=30)
        
        # Toolbar: row 1 = main tools (always visible); row 2 = optional horizontal scroll (2D classes)
        self.toolbar_host = tk.Frame(self.frame, bg="#D6EAF8")
        self.toolbar_host.pack(fill="x")

        self.toolbar = tk.Frame(self.toolbar_host, bg="#D6EAF8", pady=5)
        self.toolbar.pack(fill="x")

        tk.Button(self.toolbar, text="Load Data", command=self.load_data_router, bg="white").pack(side="left", padx=10)
        tk.Button(self.toolbar, text="Load Model", command=self.load_model_thread, bg="white").pack(side="left", padx=5)
        tk.Button(self.toolbar, text="⚡ Run AI", command=self.run_embedding_thread, bg="#F1C40F", font=("Arial", 9, "bold")).pack(side="left", padx=15)

        self.btn_limit = tk.Button(self.toolbar, text="⛔ Set Top Limit", 
                                   command=self.toggle_limit_mode, 
                                   bg="#EC7063", fg="white", font=("Arial", 9, "bold"))
        self.btn_limit.pack(side="left", padx=5)

        self.btn_x_right_limit = tk.Button(
            self.toolbar,
            text="⛔ Set Right Limit",
            command=self.toggle_x_right_limit_mode,
            bg="#5DADE2",
            fg="white",
            font=("Arial", 9, "bold"),
        )
        self.btn_x_right_limit.pack(side="left", padx=5)

        tk.Label(self.toolbar, text="| Tool:", bg="#D6EAF8").pack(side="left", padx=5)

        # --- THE EXACT 4 BUTTONS (No duplicates) ---
        tk.Radiobutton(self.toolbar, text="Point", variable=self.mode, value="ai", command=self.clear_current_annotation, bg="#D6EAF8").pack(side="left")
        tk.Radiobutton(self.toolbar, text="Box", variable=self.mode, value="box", command=self.clear_current_annotation, bg="#D6EAF8").pack(side="left")
        tk.Radiobutton(self.toolbar, text="Poly", variable=self.mode, value="polygon", command=self.clear_current_annotation, bg="#D6EAF8").pack(side="left")
        tk.Radiobutton(self.toolbar, text="Edit/Select", variable=self.mode, value="edit", command=self.clear_current_annotation, bg="#D6EAF8").pack(side="left")
        
        tk.Button(self.toolbar, text="Delete Selected", command=self.delete_selected, bg="#E74C3C", fg="white", font=("Arial", 9, "bold")).pack(side="left", padx=15)
        tk.Button(
            self.toolbar,
            text="💾 Save (S)",
            command=self.ask_label_and_save,
            bg="#27AE60",
            fg="white",
            font=("Arial", 9, "bold"),
        ).pack(side="left", padx=5)
        tk.Button(
            self.toolbar,
            text="Clear Mask (Del)",
            command=self.clear_current_annotation,
            bg="#7F8C8D",
            fg="white",
            font=("Arial", 9, "bold"),
        ).pack(side="left", padx=5)
        tk.Button(self.toolbar, text="Reset View", command=self.reset_zoom, bg="#D7BDE2").pack(side="left", padx=5)
        tk.Button(self.toolbar, text="Export view + SAM", command=self.export_current_view_with_sam, bg="#A9DFBF", font=("Arial", 9, "bold")).pack(side="left", padx=8)
        tk.Button(
            self.toolbar,
            text="💾 Export Video/Stack",
            command=self.open_export_sequence_dialog,
            bg="#16A085",
            fg="white",
            font=("Arial", 9, "bold"),
        ).pack(side="left", padx=5)

    def _ensure_toolbar_scroll_row(self):
        """Row 2 under main toolbar: horizontal scroll for class / preset buttons."""
        if hasattr(self, "toolbar_scroll_inner") and self.toolbar_scroll_inner.winfo_exists():
            return self.toolbar_scroll_inner

        outer = tk.Frame(self.toolbar_host, bg="#D6EAF8")
        outer.pack(fill="x")

        canvas = tk.Canvas(outer, bg="#D6EAF8", height=44, highlightthickness=0, bd=0)
        h_scroll = tk.Scrollbar(outer, orient="horizontal", command=canvas.xview)
        canvas.configure(xscrollcommand=h_scroll.set)

        inner = tk.Frame(canvas, bg="#D6EAF8", pady=2)
        win_id = canvas.create_window((0, 0), window=inner, anchor="nw")

        def _on_inner_configure(event):
            canvas.configure(scrollregion=(0, 0, event.width, event.height))

        def _on_canvas_configure(event):
            canvas.itemconfig(win_id, height=event.height)

        inner.bind("<Configure>", _on_inner_configure)
        canvas.bind("<Configure>", _on_canvas_configure)

        canvas.pack(side="top", fill="x")
        h_scroll.pack(side="bottom", fill="x")

        def _scroll_x(delta_units):
            canvas.xview_scroll(int(delta_units), "units")

        def _on_shift_wheel(event):
            if getattr(event, "num", None) == 4 or getattr(event, "delta", 0) > 0:
                _scroll_x(-3)
            elif getattr(event, "num", None) == 5 or getattr(event, "delta", 0) < 0:
                _scroll_x(3)

        for w in (outer, canvas, inner):
            w.bind("<Shift-MouseWheel>", _on_shift_wheel)
            w.bind("<Shift-Button-4>", lambda e: _scroll_x(-3))
            w.bind("<Shift-Button-5>", lambda e: _scroll_x(3))

        self._toolbar_scroll_canvas = canvas
        self._toolbar_scroll_inner = inner
        self.frame.after_idle(self._refresh_toolbar_scroll)
        return inner

    def _refresh_toolbar_scroll(self):
        if not hasattr(self, "_toolbar_scroll_canvas"):
            return
        try:
            if not self._toolbar_scroll_canvas.winfo_exists():
                return
            self._toolbar_scroll_inner.update_idletasks()
            self._toolbar_scroll_canvas.update_idletasks()
            bbox = self._toolbar_scroll_canvas.bbox("all")
            if bbox:
                self._toolbar_scroll_canvas.configure(scrollregion=bbox)
        except tk.TclError:
            pass

    def _bind_events(self):
        self.canvas.bind("<Button-1>", self.on_left_click)
        self.canvas.bind("<B1-Motion>", self.on_left_drag)       
        self.canvas.bind("<ButtonRelease-1>", self.on_left_release)
        self.canvas.bind("<Button-3>", self.on_right_click)
        self.canvas.bind("<ButtonPress-2>", self.start_pan)
        self.canvas.bind("<B2-Motion>", self.do_pan)
        self.canvas.bind("<Control-MouseWheel>", self.on_mousewheel) 
        self.canvas.bind("<Motion>", self.on_mouse_move)
        self.canvas.bind("<Configure>", self.on_resize)
        # Focus canvas on click so S / Delete hotkeys are reliable
        self.canvas.bind("<ButtonPress-1>", lambda e: self.canvas.focus_set(), add="+")
        self.canvas.configure(takefocus=1)

        # Avoid duplicate bind_all handlers when re-entering CT / 2D UI
        for seq in ("<Delete>", "s", "S", "<Left>", "<Right>", "<Up>", "<Down>", "<Control-z>"):
            try:
                self.frame.unbind_all(seq)
            except tk.TclError:
                pass
        self.frame.bind_all("<Delete>", self._hotkey_clear_annotation)
        self.frame.bind_all("s", self._hotkey_save_annotation)
        self.frame.bind_all("S", self._hotkey_save_annotation)
        self.frame.bind_all("<Left>", self._hotkey_left_arrow)
        self.frame.bind_all("<Right>", self._hotkey_right_arrow)
        self.frame.bind_all("<Up>", self._hotkey_up_arrow)
        self.frame.bind_all("<Down>", self._hotkey_down_arrow)
        self.frame.bind_all("<Control-z>", self._hotkey_undo)

    def _focus_is_text_input(self):
        """True when typing in Entry/Text/Spinbox — skip single-letter hotkeys."""
        try:
            w = self.frame.focus_get()
        except tk.TclError:
            return False
        if w is None:
            return False
        cls = w.winfo_class()
        return cls in ("Entry", "TEntry", "Text", "TSpinbox", "Spinbox")

    def _hotkey_save_annotation(self, e=None):
        if self._focus_is_text_input():
            return None  # allow typing "s" / "S" into Entry fields
        return self.ask_label_and_save(e)

    def _hotkey_clear_annotation(self, e=None):
        if self._focus_is_text_input():
            return None
        return self.clear_current_annotation(e)

    def _hotkey_undo(self, e=None):
        if self._focus_is_text_input():
            return
        return self.undo(e)

    def _hotkey_left_arrow(self, e=None):
        if self._focus_is_text_input():
            return
        return self.on_left_arrow(e)

    def _hotkey_right_arrow(self, e=None):
        if self._focus_is_text_input():
            return
        return self.on_right_arrow(e)

    def _hotkey_up_arrow(self, e=None):
        if self._focus_is_text_input():
            return
        return self.on_up_arrow(e)

    def _hotkey_down_arrow(self, e=None):
        if self._focus_is_text_input():
            return
        return self.on_down_arrow(e)
        
    def toggle_limit_mode(self):
        self.setting_x_right_limit_mode = False
        if self.y_limit is not None:
            self.y_limit = None
            self.btn_limit.config(text="⛔ Set Top Limit", bg="#EC7063")
            self.show_image(self.raw_image, self.current_mask)
            self.log("Top limit cleared", "green")
        else:
            self.setting_limit_mode = True
            self.btn_limit.config(text="Click top edge…", bg="#F7DC6F", fg="black")
            self.canvas.config(cursor="tcross")
            self.log("Click on image to set top exclusion (no segment above).", "blue")

    def toggle_x_right_limit_mode(self):
        """Vertical line: area to the right is never segmented."""
        self.setting_limit_mode = False
        if self.x_right_limit is not None:
            self.x_right_limit = None
            self.btn_x_right_limit.config(text="⛔ Set Right Limit", bg="#5DADE2")
            self.show_image(self.raw_image, self.current_mask)
            self.log("Right limit cleared", "green")
        else:
            self.setting_x_right_limit_mode = True
            self.btn_x_right_limit.config(text="Click right edge…", bg="#F7DC6F", fg="black")
            self.canvas.config(cursor="tcross")
            self.log("Click on image to set right exclusion (no segment to the right).", "blue")

    def on_mouse_move(self, e):
        self.canvas.delete("crosshair")
        if self.raw_image is None:
            return

        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 2 or ch < 2:
            return

        if self.setting_x_right_limit_mode and self.scale > 0:
            rx = int((e.x - self.off_x) / self.scale)
            h, w = self.raw_image.shape[:2]
            rx = max(0, min(rx, w))
            cx_line = int(rx * self.scale) + self.off_x
            self.canvas.create_line(cx_line, 0, cx_line, ch, fill="#00CED1", dash=(4, 4), width=2, tags="crosshair")
            self.canvas.create_rectangle(cx_line, 0, cw, ch, fill="#00CED1", stipple="gray50", outline="", tags="crosshair")
            self.canvas.create_text(
                cx_line + 12, 18, text="No segment →", fill="#00CED1",
                font=("Arial", 10, "bold"), anchor="nw", tags="crosshair",
            )
            return

        if not self.setting_limit_mode:
            return

        self.canvas.create_line(0, e.y, cw, e.y, fill="red", dash=(4, 4), width=2, tags="crosshair")
        self.canvas.create_line(e.x, 0, e.x, ch, fill="red", dash=(4, 4), width=2, tags="crosshair")
        self.canvas.create_rectangle(0, 0, cw, e.y, fill="red", stipple="gray50", outline="", tags="crosshair")
        self.canvas.create_text(
            e.x + 15, e.y - 15, text="No segment ↑", fill="red",
            font=("Arial", 10, "bold"), anchor="sw", tags="crosshair",
        )

    def apply_x_right_limit_constraint(self, mask):
        """Zero mask at and to the right of x_right_limit (column index in image)."""
        if self.x_right_limit is None or mask is None:
            return mask
        h, w = mask.shape[:2]
        safe_x = min(max(0, int(self.x_right_limit)), w)
        mask[:, safe_x:] = 0
        return mask

    def apply_y_limit_constraint(self, mask):
        """Apply top + right exclusion limits to a boolean/uint8 mask."""
        if mask is None:
            return mask
        if self.y_limit is not None:
            h, w = mask.shape[:2]
            safe_limit = min(max(0, self.y_limit), h)
            mask[:safe_limit, :] = 0
        return self.apply_x_right_limit_constraint(mask)
    
    def setup_ui_4d(self):
        # Do not carry 2D exclusion limits into 3D annotation.
        self.y_limit = None
        self.x_right_limit = None
        self.setting_limit_mode = False
        self.setting_x_right_limit_mode = False

        # Reuse CT UI and append dedicated 3D reconstruction controls.
        self.setup_ui_ct()
        if hasattr(self, "left_panel_inner") and self.left_panel_inner.winfo_exists():
            f_recon = tk.LabelFrame(self.left_panel_inner, text="Reconstruction", bg="white", padx=10, pady=10, font=("Arial", 12, "bold"))
            f_recon.pack(fill="x", padx=10, pady=(10, 0))
            tk.Label(
                f_recon,
                text="Annotate → set bounds → Build 3D (validation opens after propagation)",
                justify="left",
                bg="white",
                font=("Arial", 11, "bold")
            ).pack(anchor="w")
            tk.Button(
                f_recon,
                text="Build 3D",
                command=self.run_3d_reconstruction_from_current_slice,
                bg="#8E44AD",
                fg="white",
                font=("Arial", 12, "bold"),
            ).pack(fill="x", pady=(8, 4))
    # ==========================================================================
    # DATA LOADING & ROUTING
    # ==========================================================================
    def load_data_router(self):
        if self.is_ct_mode:
            self._choose_ct_load_source()
        else:
            self.load_folder_2d()

    def _choose_ct_load_source(self):
        """Ask whether to open one reconstructed volume file or a folder of volumes."""
        if not self._ensure_ct_dependencies():
            return

        dialog = Toplevel(self.frame)
        dialog.title("Load CT Volume")
        dialog.geometry("480x300")
        dialog.resizable(False, False)
        dialog.configure(bg="white")
        dialog.transient(self.frame.winfo_toplevel())
        dialog.grab_set()
        choice = {"v": None}

        tk.Label(
            dialog,
            text="Open reconstructed CT volume",
            font=("Arial", 13, "bold"),
            bg="white",
            fg="#2C3E50",
        ).pack(pady=(18, 6))
        tk.Label(
            dialog,
            text="Use a .h5 / .nii / .txm volume,\nTomocupy output folder, or a folder of volumes.",
            bg="white",
            fg="#566573",
            justify="center",
        ).pack(pady=(0, 12))

        def pick(v):
            choice["v"] = v
            dialog.destroy()

        btn = {"font": ("Arial", 10, "bold"), "width": 32, "height": 2}
        tk.Button(
            dialog,
            text="Browse one .h5 / .nii / .txm volume…",
            bg="#E67E22",
            fg="white",
            command=lambda: pick("file"),
            **btn,
        ).pack(pady=4)
        tk.Button(
            dialog,
            text="Browse folder of volumes…",
            bg="#2980B9",
            fg="white",
            command=lambda: pick("folder"),
            **btn,
        ).pack(pady=4)
        if HAS_TOMOCUPY_VOLUME:
            tk.Button(
                dialog,
                text="Open Tomocupy reconstruction…",
                bg="#117A65",
                fg="white",
                command=lambda: pick("tomocupy"),
                **btn,
            ).pack(pady=4)
        if HAS_COR_WIZARD:
            tk.Button(
                dialog,
                text="Tomocupy workbench (projections .h5)…",
                bg="#8E44AD",
                fg="white",
                command=lambda: pick("wizard"),
                **btn,
            ).pack(pady=4)
        tk.Button(dialog, text="Cancel", bg="#95A5A6", fg="white",
                  command=lambda: pick(None), width=12).pack(pady=10)
        self.frame.wait_window(dialog)

        if choice["v"] == "file":
            self.load_file_ct()
        elif choice["v"] == "folder":
            self.load_folder_ct()
        elif choice["v"] == "tomocupy":
            self.load_tomocupy_reconstruction_ct()
        elif choice["v"] == "wizard":
            self.load_projection_h5_wizard()

    def _ensure_ct_dependencies(self) -> bool:
        utils.init_viz_dependencies(verbose=True)
        _sync_ml_flags_from_utils()
        if HAS_NIBABEL or HAS_H5PY or HAS_OLEFILE:
            return True
        messagebox.showerror(
            "Error",
            "Missing libraries.\nNeed 'nibabel' (.nii), 'h5py' (.h5), or 'olefile' (.txm)",
        )
        return False

    def _populate_ct_file_listbox(self):
        if not hasattr(self, "file_listbox"):
            return
        self.file_listbox.delete(0, tk.END)
        for path in self.image_list:
            self.file_listbox.insert(tk.END, os.path.basename(path))
        if self.image_list:
            self.file_listbox.selection_clear(0, tk.END)
            self.file_listbox.selection_set(self.current_idx)
            self.file_listbox.see(self.current_idx)

    def _prepare_ct_ui_after_paths(self, class_dir):
        self.ct_volume = None
        self.raw_image = None
        self.is_ct_mode = True
        if hasattr(self, "ct_ctrl_frame") and self.ct_ctrl_frame.winfo_exists():
            self.ct_ctrl_frame.pack(side="bottom", fill="x", pady=5)
        self.load_classes_txt(class_dir)
        self.current_idx = 0
        self._populate_ct_file_listbox()
        self.load_current_ct_volume()

    def load_file_ct(self):
        """Browse a single reconstructed CT volume (.h5 / .nii / .nii.gz / .txm)."""
        if not self._ensure_ct_dependencies():
            return
        path = filedialog.askopenfilename(
            title="Select reconstructed CT volume",
            filetypes=[
                ("CT volumes", "*.h5 *.hdf5 *.nii *.nii.gz *.txm"),
                ("Zeiss TXM", "*.txm"),
                ("HDF5 reconstructed", "*.h5 *.hdf5"),
                ("NIfTI", "*.nii *.nii.gz"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        path = os.path.abspath(path)
        low = path.lower()
        if not (
            low.endswith(".h5")
            or low.endswith(".hdf5")
            or low.endswith(".nii")
            or low.endswith(".nii.gz")
            or low.endswith(".txm")
        ):
            messagebox.showerror("Error", "Please select a .h5 / .hdf5 / .nii / .nii.gz / .txm volume.")
            return
        if low.endswith(".txm") and not HAS_OLEFILE:
            messagebox.showerror("Error", "olefile is required for .txm files.\n  pip install olefile")
            return
        resolved = self._resolve_projection_h5_load_path(path)
        if resolved is None or resolved == "__wizard__":
            return
        if resolved == "__cancel__":
            return
        self.image_list = [resolved]
        self.log(f"Loading volume: {os.path.basename(resolved)}", "blue")
        self._prepare_ct_ui_after_paths(os.path.dirname(resolved))

    def load_folder_ct(self):
        if not self._ensure_ct_dependencies():
            return

        d = filedialog.askdirectory(title="Select folder containing .nii / .h5 / .txm volumes")
        if not d:
            return

        self.image_list = collect_ct_volume_paths(d, recursive=True)
        if not self.image_list:
            messagebox.showerror(
                "Error",
                f"No .nii / .nii.gz / .h5 / .txm volumes found in:\n{d}\n\n"
                "(searched this folder and subfolders)",
            )
            return

        self.log(f"Found {len(self.image_list)} volume file(s). Loading first...", "blue")
        self._prepare_ct_ui_after_paths(d)

    def load_projection_h5_wizard(self):
        """Pick a projection HDF5 and open the Tomocupy COR + reconstruction window."""
        if not self._ensure_ct_dependencies() or not HAS_COR_WIZARD:
            messagebox.showerror("Tomocupy", "Tomocupy workbench is not available.")
            return
        path = filedialog.askopenfilename(
            title="Select projection HDF5 (exchange/data + theta)",
            filetypes=[("HDF5 projections", "*.h5 *.hdf5"), ("All", "*.*")],
        )
        if not path:
            return
        path = os.path.abspath(path)
        if not is_projection_h5 or not is_projection_h5(path):
            messagebox.showerror(
                "Tomocupy",
                "This HDF5 does not look like a projection stack.\n"
                "Need exchange/data and exchange/theta.",
            )
            return
        self.tomocupy_source_h5 = path
        self._open_tomocupy_cor_wizard(path)

    def _open_tomocupy_cor_wizard(self, projection_h5: str):
        if not HAS_COR_WIZARD or not open_tomocupy_wizard:
            return
        projection_h5 = os.path.abspath(projection_h5)
        self.tomocupy_source_h5 = projection_h5
        self._update_tomocupy_button_visibility()

        def on_done(vol_path, kind, _center):
            if not vol_path:
                return
            self.projection_cor_tune_mode = False
            self.image_list = [vol_path]
            self.log(f"Loading Tomocupy reconstruction ({kind})…", "blue")
            self._prepare_ct_ui_after_paths(os.path.dirname(projection_h5))
            messagebox.showinfo(
                "Ready for SAM",
                "Reconstructed volume is loaded.\n\n"
                "1) Choose a class (e.g. pore)\n"
                "2) Annotate on slice views (SAM)\n"
                "3) Save labels with S on each slice\n"
                "4) Click Build 3D for volumetric label propagation",
            )

        open_tomocupy_wizard(
            self.frame.winfo_toplevel(),
            projection_h5,
            on_complete=on_done,
        )

    def load_tomocupy_reconstruction_ct(self):
        """Browse Tomocupy output (recon .h5, TIFF stack folder, or try-recon TIFFs)."""
        if not self._ensure_ct_dependencies() or not HAS_TOMOCUPY_VOLUME:
            messagebox.showerror(
                "Tomocupy",
                "Tomocupy volume helper failed to load (support/smaxi_tomocupy).",
            )
            return
        path = filedialog.askopenfilename(
            title="Tomocupy — select reconstruction or projection H5",
            filetypes=[
                ("Tomocupy / HDF5", "*.h5 *.hdf5"),
                ("TIFF (try recon slice)", "*.tif *.tiff"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            folder = filedialog.askdirectory(
                title="Tomocupy — select output folder (tomocupy_out or TIFF stack)",
            )
            if not folder:
                return
            path = folder
        resolved = find_tomocupy_reconstruction(path)
        if not resolved:
            messagebox.showerror(
                "Tomocupy",
                "No Tomocupy reconstruction found.\n\n"
                "Look for:\n"
                "  • tomocupy_out.h5 (full recon)\n"
                "  • tomocupy_out/recon_*.tiff\n"
                "  • recon_slice*_center*.tiff (try mode)\n\n"
                "Run Tomocupy from the workbench, or pick the output folder.",
            )
            return
        vol_path, kind = resolved
        self.image_list = [vol_path]
        label = os.path.basename(vol_path)
        if kind == "tiff_stack":
            label = f"{os.path.basename(vol_path)} (TIFF stack)"
        self.log(f"Tomocupy {kind}: {label}", "blue")
        self._prepare_ct_ui_after_paths(os.path.dirname(vol_path) or vol_path)

    def _resolve_projection_h5_load_path(self, path: str):
        """
        For projection HDF5: offer existing Tomocupy recon, or run center+full pipeline.
        Returns None if an async Tomocupy job was started.
        """
        path = os.path.abspath(path)
        if not HAS_H5PY or not is_projection_h5 or not is_projection_h5(path):
            return path

        self.tomocupy_source_h5 = path
        self._update_tomocupy_button_visibility()

        found = None
        if HAS_TOMOCUPY_VOLUME:
            found = discover_near_projection_h5(path)

        if found:
            vol_path, kind = found
            extra = ""
            if kind == "tiff_stack" and os.path.isdir(vol_path):
                n_slices = len(list_recon_tiff_stack(vol_path))
                if n_slices:
                    extra = f"\n({n_slices} slices)"
            use_recon = messagebox.askyesno(
                "Tomocupy reconstruction found",
                "This HDF5 is raw projections.\n\n"
                "Open the existing Tomocupy volume?\n\n"
                f"{vol_path}\n({kind}){extra}\n\n"
                "No = load low-res projection preview only.\n"
                "You can re-run Tomocupy from the left panel.",
                icon="question",
            )
            if use_recon:
                return vol_path
            return path

        if HAS_COR_WIZARD:
            choice = messagebox.askyesnocancel(
                "Projection HDF5",
                "This file is raw CT projections (not a 3D volume).\n\n"
                "Yes — Open Tomocupy workbench (COR, preprocessing, GPU recon)\n"
                "No — Load low-res preview in main viewer only\n"
                "Cancel — Do not load",
                icon="question",
            )
            if choice is None:
                return "__cancel__"
            if choice:
                self.tomocupy_source_h5 = os.path.abspath(path)
                self.frame.after(50, lambda p=path: self._open_tomocupy_cor_wizard(p))
                return "__wizard__"
            return path

        run_now = messagebox.askyesno(
            "Run Tomocupy reconstruction",
            "This HDF5 contains raw CT projections.\n\n"
            "Run Tomocupy inline pipeline?\n"
            "(Install Tomocupy workbench: support/smaxi_tomocupy)",
            icon="question",
        )
        if run_now and self._run_tomocupy_pipeline(path):
            return None
        return path

    def _current_cor_nsino(self) -> str:
        if self._cor_n_rows > 0:
            return nsino_fraction_from_row(int(self.cor_plane_row.get()), self._cor_n_rows)
        return MIDPLANE_NSINO

    def _init_cor_plane_controls(self, projection_h5: str):
        if not HAS_TOMOCUPY_RUNNER:
            return
        try:
            n_proj, n_row, _n_col = projection_stack_shape(projection_h5)
        except (OSError, ValueError):
            return
        self._cor_n_rows = n_row
        self._cor_n_proj = n_proj
        self._cor_preview_proj_idx = n_proj // 2
        self.cor_plane_scale.config(from_=0, to=max(0, n_row - 1))
        saved_nsino = load_cor_settings_nsino(projection_h5)
        if saved_nsino is not None:
            row = row_from_nsino_fraction(saved_nsino, n_row)
        else:
            row = row_from_nsino_fraction(MIDPLANE_NSINO, n_row)
        self.cor_plane_row.set(row)
        self._update_cor_plane_info_label()
        self._bind_cor_plane_mousewheel()
        self._bind_cor_preview_traces()

    def _update_cor_plane_info_label(self):
        if self._cor_n_rows <= 0:
            self.cor_plane_info_var.set("COR plane: —")
            return
        row = int(self.cor_plane_row.get())
        nsino = self._current_cor_nsino()
        self.cor_plane_info_var.set(
            f"Row {row} / {self._cor_n_rows - 1}\n"
            f"nsino={nsino}\n"
            f"Preview: sinogram at this row\n"
            f"(scroll plane or edit COR px)"
        )

    def _bind_cor_plane_mousewheel(self):
        if getattr(self, "_cor_wheel_bound", False):
            return
        self._cor_wheel_bound = True

        def _wheel(event):
            if not getattr(self, "projection_cor_tune_mode", False):
                return
            step = -1 if event.delta > 0 else 1
            row = int(self.cor_plane_row.get()) + step
            row = max(0, min(max(0, self._cor_n_rows - 1), row))
            self.cor_plane_row.set(row)
            self._schedule_cor_preview_refresh()

        self.canvas.bind("<MouseWheel>", _wheel)

    def _bind_cor_preview_traces(self):
        if getattr(self, "_cor_traces_bound", False):
            return
        self._cor_traces_bound = True

        def _on_cor_var(*_args):
            if self._cor_trace_guard:
                return
            self._schedule_cor_preview_refresh()

        def _on_row_var(*_args):
            self._schedule_cor_preview_refresh()

        self.tomocupy_cor_var.trace_add("write", _on_cor_var)
        self.cor_plane_row.trace_add("write", _on_row_var)
        self.entry_tomocupy_cor.bind("<KeyRelease>", lambda _e: self._schedule_cor_preview_refresh())

    def _cor_preview_active(self) -> bool:
        return bool(
            getattr(self, "projection_cor_tune_mode", False)
            and self.tomocupy_source_h5
            and self._cor_n_rows > 0
        )

    def _schedule_cor_preview_refresh(self, delay_ms: int = 40):
        if not self._cor_preview_active():
            return
        aid = getattr(self, "_cor_preview_after_id", None)
        if aid is not None:
            try:
                self.frame.after_cancel(aid)
            except tk.TclError:
                pass
        self._cor_preview_after_id = self.frame.after(
            delay_ms, self._show_cor_projection_preview
        )

    def _read_sinogram_preview_at_row(
        self, projection_h5: str, row: int, max_angles: int = 800
    ) -> np.ndarray:
        """Sinogram slice Tomocupy uses for COR at fixed detector row (angles × width)."""
        import h5py

        def _slice_from_dataset(ds):
            n_proj, n_row, _n_col = (int(ds.shape[0]), int(ds.shape[1]), int(ds.shape[2]))
            r = max(0, min(n_row - 1, int(row)))
            step = max(1, n_proj // max(64, int(max_angles)))
            return np.asarray(ds[::step, r, :], dtype=np.float32)

        if (
            self.h5_handle is not None
            and self.image_list
            and os.path.abspath(self.image_list[self.current_idx])
            == os.path.abspath(projection_h5)
            and "exchange/data" in self.h5_handle
        ):
            return _slice_from_dataset(self.h5_handle["exchange/data"])

        with h5py.File(projection_h5, "r") as f:
            return _slice_from_dataset(f["exchange/data"])

    def _parse_cor_px_silent(self) -> Optional[float]:
        try:
            raw = (self.tomocupy_cor_var.get() or "").strip()
            if not raw:
                return None
            return float(raw)
        except ValueError:
            return None

    def _show_cor_projection_preview(self):
        """Sinogram at selected row; green line = COR (px). Updates on scroll and typing."""
        self._cor_preview_after_id = None
        if not self._cor_preview_active():
            return
        src = self.tomocupy_source_h5
        self._update_cor_plane_info_label()
        try:
            row = int(self.cor_plane_row.get())
            sino = self._read_sinogram_preview_at_row(src, row)
            lo, hi = np.percentile(sino, [1, 99])
            if hi <= lo:
                lo, hi = float(np.min(sino)), float(np.max(sino))
            if hi <= lo:
                hi = lo + 1.0
            u8 = np.clip((sino - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
            rgb = cv2.cvtColor(u8, cv2.COLOR_GRAY2RGB)
            cor = self._parse_cor_px_silent()
            if cor is not None:
                cx = int(round(cor))
                if 0 <= cx < rgb.shape[1]:
                    cv2.line(rgb, (cx, 0), (cx, rgb.shape[0] - 1), (0, 255, 0), 2)
            cv2.putText(
                rgb,
                f"sinogram row {row}  COR px",
                (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
            self.raw_image = rgb.copy()
            self.canvas.delete("all")
            self.show_image(rgb)
            try:
                self.canvas.update_idletasks()
            except tk.TclError:
                pass
        except Exception as e:
            self.log(f"COR preview error: {e}", "red")

    def _on_cor_plane_slider(self, _val=None):
        self._schedule_cor_preview_refresh()

    def _refresh_cor_ui_for_projection(self, projection_h5: str):
        if not HAS_TOMOCUPY_RUNNER or not projection_h5:
            return
        self._cor_trace_guard = True
        try:
            saved = load_cor_settings(projection_h5)
            if saved is not None:
                self.tomocupy_cor_var.set(f"{saved:.4f}")
            else:
                try:
                    self.tomocupy_cor_var.set(f"{default_cor_guess(projection_h5):.2f}")
                except OSError:
                    self.tomocupy_cor_var.set("")
        finally:
            self._cor_trace_guard = False
        try:
            det_w = projection_detector_width(projection_h5)
            self.tomocupy_cor_hint_var.set(
                f"Scroll detector row for best COR plane; one COR ({det_w}px axis) "
                f"applies to all reconstructed slices."
            )
        except OSError:
            pass
        self._init_cor_plane_controls(projection_h5)
        self._schedule_cor_preview_refresh(delay_ms=80)

    def _parse_cor_entry(self) -> Optional[float]:
        raw = (self.tomocupy_cor_var.get() or "").strip()
        if not raw:
            return None
        try:
            return float(raw)
        except ValueError:
            messagebox.showerror("COR", "Enter a numeric center of rotation (pixels).")
            return None

    def _on_save_cor_clicked(self):
        src = self.tomocupy_source_h5
        cor = self._parse_cor_entry()
        if not src or cor is None:
            return
        path = save_cor_settings(
            src, cor, nsino=self._current_cor_nsino(), source="manual_ui"
        )
        self.log(f"Saved COR {cor:.4f} px → {path}", "green")

    def _confirm_cor_adjustment(self, detected: float, det_width: int) -> Optional[float]:
        """Modal dialog: adjust midplane COR before full-volume reconstruction."""
        result = {"value": None}
        dlg = Toplevel(self.frame.winfo_toplevel())
        dlg.title("Adjust center of rotation (COR)")
        dlg.geometry("480x280")
        dlg.resizable(False, False)
        dlg.configure(bg="white")
        dlg.transient(self.frame.winfo_toplevel())
        dlg.grab_set()

        tk.Label(
            dlg,
            text="Midplane COR estimate",
            bg="white",
            font=("Arial", 12, "bold"),
        ).pack(pady=(14, 4))
        tk.Label(
            dlg,
            text=(
                f"Tomocupy uses a single rotation axis (COR) for the entire volume — "
                f"every axial / cross-sectional slice uses the same value.\n\n"
                f"Detected on plane nsino={self._current_cor_nsino()}: {detected:.4f} px\n"
                f"Detector width: {det_width} px"
            ),
            bg="white",
            fg="#2C3E50",
            wraplength=440,
            justify="left",
        ).pack(padx=16, pady=(0, 10))

        row = tk.Frame(dlg, bg="white")
        row.pack(pady=4)
        tk.Label(row, text="COR to use (px):", bg="white", font=("Arial", 10, "bold")).pack(
            side="left"
        )
        cor_var = tk.StringVar(value=f"{detected:.4f}")
        tk.Entry(row, textvariable=cor_var, font=("Consolas", 12), width=16).pack(
            side="left", padx=8
        )

        def ok():
            try:
                result["value"] = float(cor_var.get().strip())
            except ValueError:
                messagebox.showerror("COR", "Invalid number.", parent=dlg)
                return
            dlg.destroy()

        def cancel():
            dlg.destroy()

        btn_row = tk.Frame(dlg, bg="white")
        btn_row.pack(pady=16)
        tk.Button(btn_row, text="Apply & continue full recon", bg="#117A65", fg="white",
                  font=("Arial", 10, "bold"), command=ok, width=24).pack(side="left", padx=6)
        tk.Button(btn_row, text="Cancel", bg="#95A5A6", fg="white", command=cancel, width=10).pack(
            side="left"
        )
        dlg.wait_window()
        return result["value"]

    def _on_detect_cor_at_plane(self):
        src = self.tomocupy_source_h5
        if not src or not is_projection_h5(src):
            messagebox.showinfo("COR", "Load a projection HDF5 first.")
            return
        if not HAS_TOMOCUPY_RUNNER:
            return
        if self._tomocupy_running:
            messagebox.showinfo("Tomocupy", "Wait for the current job to finish.")
            return
        nsino = self._current_cor_nsino()
        self._tomocupy_running = True
        self._update_tomocupy_button_visibility()

        def worker():
            err = None
            center = None
            try:
                center = find_center_midplane(
                    src,
                    nsino=nsino,
                    log_callback=lambda m: self.log(m, "blue"),
                )
            except Exception as e:
                err = e

            def done():
                self._tomocupy_running = False
                self._update_tomocupy_button_visibility()
                if err:
                    messagebox.showerror("COR detection failed", str(err))
                    return
                self.tomocupy_cor_var.set(f"{center:.4f}")
                save_cor_settings(
                    src, center, nsino=nsino, source="plane_try_auto"
                )
                self.log(
                    f"COR at nsino={nsino}: {center:.4f} px — scroll plane & re-detect if needed",
                    "green",
                )
                if getattr(self, "projection_cor_tune_mode", False):
                    self._show_cor_projection_preview()

            self.frame.after(0, done)

        threading.Thread(target=worker, daemon=True).start()

    def _update_tomocupy_button_visibility(self):
        if not hasattr(self, "btn_run_tomocupy"):
            return
        show = (
            HAS_TOMOCUPY_RUNNER
            and self.is_ct_mode
            and self.tomocupy_source_h5
            and is_projection_h5(self.tomocupy_source_h5)
        )
        if show:
            if HAS_COR_WIZARD:
                self.btn_open_cor_wizard.pack(fill="x", pady=(8, 0))
            self.f_tomocupy_cor.pack_forget()
            self.btn_run_tomocupy.pack_forget()
            state = tk.NORMAL if not self._tomocupy_running else tk.DISABLED
            if HAS_COR_WIZARD and hasattr(self, "btn_open_cor_wizard"):
                self.btn_open_cor_wizard.config(state=state)
            if getattr(self, "_cor_ui_path", None) != self.tomocupy_source_h5:
                self._cor_ui_path = self.tomocupy_source_h5
                self._refresh_cor_ui_for_projection(self.tomocupy_source_h5)
        else:
            self._cor_ui_path = None
            if hasattr(self, "btn_open_cor_wizard"):
                self.btn_open_cor_wizard.pack_forget()
            self.f_tomocupy_cor.pack_forget()
            self.btn_run_tomocupy.pack_forget()

    def _on_open_cor_wizard_clicked(self):
        src = self.tomocupy_source_h5
        if not src and self.image_list:
            cand = self.image_list[self.current_idx]
            if is_projection_h5 and is_projection_h5(cand):
                src = cand
        if not src:
            self.load_projection_h5_wizard()
            return
        self._open_tomocupy_cor_wizard(src)

    def _on_run_tomocupy_clicked(self):
        src = self.tomocupy_source_h5
        if not src and self.image_list:
            cand = self.image_list[self.current_idx]
            if is_projection_h5 and is_projection_h5(cand):
                src = cand
        if not src:
            messagebox.showinfo(
                "Tomocupy",
                "Load a projection HDF5 first (exchange/data + exchange/theta).",
            )
            return
        if (
            HAS_TOMOCUPY_VOLUME
            and discover_near_projection_h5
            and discover_near_projection_h5(src)
        ):
            if not messagebox.askyesno(
                "Tomocupy",
                "A reconstruction already exists next to this file.\n\n"
                "Run Tomocupy again and overwrite output?",
            ):
                return
        cor = self._parse_cor_entry()
        if cor is None:
            if not messagebox.askyesno(
                "COR not set",
                "No COR in the field.\n\nRun midplane detection and adjust COR before full recon?",
            ):
                return
            self._run_tomocupy_pipeline(src, run_midplane_detect_first=True)
        else:
            self._run_tomocupy_pipeline(
                src, run_midplane_detect_first=False, rotation_axis_px=cor
            )

    def _run_tomocupy_pipeline(
        self,
        projection_h5: str,
        *,
        run_midplane_detect_first: bool = False,
        rotation_axis_px: Optional[float] = None,
    ) -> bool:
        if not HAS_TOMOCUPY_RUNNER:
            messagebox.showerror(
                "Tomocupy",
                "Tomocupy runner not available.\n"
                "Install conda env tomocupy_recon (see support/scripts/run_tomocupy_recon.ps1).",
            )
            return False
        if self._tomocupy_running:
            messagebox.showinfo("Tomocupy", "A reconstruction is already running.")
            return False
        try:
            resolve_tomocupy_exe()
        except FileNotFoundError as e:
            messagebox.showerror("Tomocupy", str(e))
            return False

        projection_h5 = os.path.abspath(projection_h5)
        self.tomocupy_source_h5 = projection_h5
        self._tomocupy_running = True
        self._update_tomocupy_button_visibility()

        progress = Toplevel(self.frame.winfo_toplevel())
        progress.title("Tomocupy reconstruction")
        progress.geometry("620x420")
        progress.configure(bg="white")
        progress.transient(self.frame.winfo_toplevel())
        tk.Label(
            progress,
            text="Tomocupy: midplane center search, then full reconstruction",
            bg="white",
            font=("Arial", 11, "bold"),
            wraplength=580,
        ).pack(pady=(12, 6), padx=12, anchor="w")
        tk.Label(
            progress,
            text=os.path.basename(projection_h5),
            bg="white",
            fg="#566573",
            font=("Consolas", 9),
        ).pack(padx=12, anchor="w")
        log_frame = tk.Frame(progress, bg="white")
        log_frame.pack(fill="both", expand=True, padx=12, pady=8)
        sb = tk.Scrollbar(log_frame)
        sb.pack(side="right", fill="y")
        log_text = tk.Text(
            log_frame,
            height=16,
            font=("Consolas", 9),
            yscrollcommand=sb.set,
            wrap="word",
        )
        log_text.pack(side="left", fill="both", expand=True)
        sb.config(command=log_text.yview)

        def append_log(line: str):
            log_text.insert(tk.END, line.rstrip() + "\n")
            log_text.see(tk.END)

        append_log(f"Output folder: {default_out_dir(projection_h5)}")
        nsino_detect = self._current_cor_nsino()

        def worker():
            err = None
            center = None
            try:
                log_cb = lambda msg: self.frame.after(0, lambda m=msg: append_log(m))
                det_width = projection_detector_width(projection_h5)
                cor_px = rotation_axis_px

                if run_midplane_detect_first:
                    append_log(
                        f"Step 1: COR at selected plane (nsino={nsino_detect})…"
                    )
                    detected = find_center_midplane(
                        projection_h5,
                        log_callback=log_cb,
                        nsino=nsino_detect,
                    )
                    append_log(f"Detected COR: {detected:.4f} px")
                    holder = {"cor": None}
                    done_evt = threading.Event()

                    def ask_cor():
                        holder["cor"] = self._confirm_cor_adjustment(
                            detected, det_width
                        )
                        self.tomocupy_cor_var.set(
                            f"{holder['cor']:.4f}" if holder["cor"] is not None else ""
                        )
                        done_evt.set()

                    self.frame.after(0, ask_cor)
                    done_evt.wait()
                    if holder["cor"] is None:
                        raise RuntimeError("Full reconstruction cancelled (COR dialog).")
                    cor_px = holder["cor"]
                    save_cor_settings(
                        projection_h5, cor_px, nsino=nsino_detect, source="pipeline_try"
                    )
                elif cor_px is None:
                    raise RuntimeError("COR is required for full reconstruction.")
                else:
                    append_log(
                        f"Using COR from field: {cor_px:.4f} px at nsino={nsino_detect}"
                    )

                center, _out_dir = run_center_search_then_full(
                    projection_h5,
                    log_callback=log_cb,
                    rotation_axis_px=cor_px,
                    skip_center_search=True,
                )
            except Exception as e:
                err = e

            def finish():
                self._tomocupy_running = False
                self._update_tomocupy_button_visibility()
                try:
                    progress.destroy()
                except tk.TclError:
                    pass
                if err is not None:
                    self.log(f"Tomocupy failed: {err}", "red")
                    messagebox.showerror("Tomocupy failed", str(err))
                    return
                self.log(f"Tomocupy done. Rotation center ≈ {center:.4f} px", "green")
                if not HAS_TOMOCUPY_VOLUME:
                    messagebox.showinfo(
                        "Tomocupy",
                        f"Reconstruction finished.\nRotation center: {center:.4f} px",
                    )
                    return
                resolved = find_tomocupy_reconstruction(projection_h5)
                if not resolved:
                    messagebox.showwarning(
                        "Tomocupy",
                        "Reconstruction finished but output path was not detected.\n"
                        f"Look in {default_out_dir(projection_h5)}",
                    )
                    return
                vol_path, kind = resolved
                self.image_list = [vol_path]
                self._prepare_ct_ui_after_paths(os.path.dirname(projection_h5))
                messagebox.showinfo(
                    "Tomocupy",
                    f"Rotation center: {center:.4f} px\n\nLoaded {kind}:\n{vol_path}",
                )

            self.frame.after(0, finish)

        threading.Thread(target=worker, daemon=True).start()
        return True

    def load_folder_2d(self):
        d = filedialog.askdirectory(title="Select 2D Image Folder")
        if not d: return
        
        # 1. Robust File Scanning (Case-Insensitive)
        valid_exts = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}
        self.image_list = []
        
        # Scan all files and check extensions manually
        if os.path.exists(d):
            for f in os.listdir(d):
                base, ext = os.path.splitext(f)
                if ext.lower() in valid_exts:
                    self.image_list.append(os.path.join(d, f))
        
        # Sort naturally (e.g., img1, img2, img10 instead of img1, img10, img2)
        self.image_list = natsorted(self.image_list)
        
        if not self.image_list: 
            messagebox.showerror("Error", f"No images found in:\n{d}\n\nLooking for: {valid_exts}"); return
        
        # 2. Load classes.txt if it exists
        self.load_classes_txt(d)
        
        # 3. Populate Sidebar List with Checkbox Logic
        self.file_listbox.delete(0, tk.END)
        for f in self.image_list: 
            base = os.path.splitext(os.path.basename(f))[0]
            txt_path = os.path.join(d, "labels_txt", base + ".txt")
            
            if os.path.exists(txt_path):
                self.file_listbox.insert(tk.END, f"[✓] {os.path.basename(f)}")
                self.file_listbox.itemconfig(tk.END, {'fg': 'green'})
            else:
                self.file_listbox.insert(tk.END, f"[  ] {os.path.basename(f)}")
                self.file_listbox.itemconfig(tk.END, {'fg': 'black'})
        
        # 4. Load the First Image
        self.current_idx = 0
        self.load_current_image_2d()

    def load_classes_txt(self, d):
        p = os.path.join(d, "classes.txt")
        self.class_mapping = {}; self.class_colors = {}
        if os.path.exists(p):
            with open(p) as f:
                for i, line in enumerate(f):
                    if line.strip(): 
                        self.class_mapping[i] = line.strip()
                        self.class_colors[i] = self.PALETTE[i%len(self.PALETTE)]
        # Auto-select first class so S-save works after load
        if self.class_mapping and self.current_class_id.get() not in self.class_mapping:
            self.current_class_id.set(min(self.class_mapping.keys()))
        elif not self.class_mapping:
            self.current_class_id.set(-1)
        self.refresh_class_ui()

    def save_classes_txt(self, d):
        p = os.path.join(d, "classes.txt")
        try:
            os.makedirs(d, exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                for i in sorted(self.class_mapping.keys()):
                    f.write(f"{self.class_mapping[i]}\n")
        except OSError as e:
            self.log(f"Could not save classes.txt: {e}", "red")

    def _classes_save_dir(self):
        """Folder that owns classes.txt (volume/image parent)."""
        if self.image_list:
            return os.path.dirname(self.image_list[0])
        return None

    def rename_class_by_id(self, class_id: int, new_name: str) -> bool:
        new_name = (new_name or "").strip()
        if not new_name:
            return False
        for i, n in self.class_mapping.items():
            if i != class_id and n.lower() == new_name.lower():
                messagebox.showwarning("Rename", f"Name '{new_name}' already used by class {i}.")
                return False
        self.class_mapping[class_id] = new_name
        if self.image_list:
            self.save_classes_txt(os.path.dirname(self.image_list[0]))
        self.refresh_class_ui()
        self.log(f"Renamed class {class_id} -> {new_name}", "green")
        return True

    def _load_annotations_list_2d(self, image_path: str) -> list:
        ann = []
        bn = os.path.splitext(os.path.basename(image_path))[0]
        tp = os.path.join(os.path.dirname(image_path), "labels_txt", bn + ".txt")
        if os.path.exists(tp):
            with open(tp) as f:
                for line in f:
                    parts = line.split()
                    if len(parts) >= 2:
                        ann.append({"id": int(parts[0]), "coords": [float(x) for x in parts[1:]]})
        return ann

    def _compose_overlay_for_annotations(self, rgb, annotations, mask=None):
        """Render saved annotations on rgb (export / sequence; does not use live canvas state)."""
        if rgb is None:
            return None
        disp = rgb.copy()
        h, w = rgb.shape[:2]
        overlay = disp.copy()
        for ann in annotations:
            pts = np.array(
                [[int(c * w) if k % 2 == 0 else int(c * h) for k, c in enumerate(ann["coords"])]],
                np.int32,
            ).reshape((-1, 1, 2))
            col = self.class_colors.get(ann["id"], (255, 255, 255))
            cv2.fillPoly(overlay, [pts], col)
        cv2.addWeighted(overlay, 0.4, disp, 0.6, 0, disp)
        if mask is not None:
            mask = self.apply_y_limit_constraint(mask)
            m_u8 = (mask * 255).astype(np.uint8)
            c_mask = np.zeros_like(disp)
            c_mask[:] = (0, 255, 0)
            c_mask = cv2.bitwise_and(c_mask, c_mask, mask=m_u8)
            disp = cv2.addWeighted(disp, 1.0, c_mask, 0.5, 0)
        if self.y_limit is not None:
            limit_overlay = disp.copy()
            cv2.rectangle(limit_overlay, (0, 0), (w, self.y_limit), (255, 0, 0), -1)
            cv2.line(limit_overlay, (0, self.y_limit), (w, self.y_limit), (255, 255, 0), 2)
            cv2.addWeighted(limit_overlay, 0.3, disp, 0.7, 0, disp)
        if self.x_right_limit is not None:
            sx = min(max(0, int(self.x_right_limit)), w)
            limit_overlay = disp.copy()
            cv2.rectangle(limit_overlay, (sx, 0), (w, h), (0, 200, 255), -1)
            cv2.line(limit_overlay, (sx, 0), (sx, h), (0, 255, 255), 2)
            cv2.addWeighted(limit_overlay, 0.3, disp, 0.7, 0, disp)
        return disp

    def _annotations_to_mask(self, annotations, h: int, w: int) -> np.ndarray:
        mask = np.zeros((h, w), dtype=np.uint8)
        for ann in annotations:
            pts = np.array(
                [[int(c * w) if k % 2 == 0 else int(c * h) for k, c in enumerate(ann["coords"])]],
                np.int32,
            ).reshape((-1, 1, 2))
            cv2.fillPoly(mask, [pts], 255)
        return mask

    # ==========================================================================
    # CT SEGMENTATION → 3D PREVIEW VOLUME
    # ==========================================================================

    def _ct_raw_slice_shape(self, axis: str):
        d0, d1, d2 = (int(x) for x in self.ct_dims)
        if axis == "X":
            return d1, d2
        if axis == "Y":
            return d0, d2
        return d0, d1

    def _annotations_to_raw_slice_mask(self, annotations, axis: str) -> np.ndarray:
        """Rasterize saved polygons (display-normalized coords) into raw slice indices."""
        raw_h, raw_w = self._ct_raw_slice_shape(axis)
        h_disp, w_disp = raw_w, raw_h  # rot90 swaps axes
        mask_disp = np.zeros((h_disp, w_disp), dtype=np.uint8)
        for ann in annotations:
            coords = ann.get("coords", [])
            if len(coords) < 6:
                continue
            pts = []
            for i in range(0, len(coords), 2):
                x = int(np.clip(coords[i] * w_disp, 0, w_disp - 1))
                y = int(np.clip(coords[i + 1] * h_disp, 0, h_disp - 1))
                pts.append([x, y])
            if len(pts) >= 3:
                cv2.fillPoly(mask_disp, [np.array(pts, dtype=np.int32)], 1)
        return self._mask_display_to_raw(mask_disp)

    def _downsample_slice_mask(self, mask_raw: np.ndarray, step: int) -> np.ndarray:
        step = max(1, int(step))
        rh, rw = mask_raw.shape[:2]
        th, tw = max(1, rh // step), max(1, rw // step)
        if th == rh and tw == rw:
            return (mask_raw > 0).astype(np.uint8)
        return cv2.resize(
            (mask_raw > 0).astype(np.uint8),
            (tw, th),
            interpolation=cv2.INTER_NEAREST,
        )

    def _paint_mask_into_preview_volume(self, label_vol, axis: str, idx_full: int, mask_raw: np.ndarray, class_id: int):
        if label_vol is None or mask_raw is None or not np.any(mask_raw):
            return
        step = max(1, int(getattr(self, "ct_preview_step", 1)))
        idx_p = int(idx_full) // step
        label_val = int(class_id) + 1
        m = self._downsample_slice_mask(mask_raw, step).astype(bool)
        if axis == "X":
            if idx_p < 0 or idx_p >= label_vol.shape[0]:
                return
            plane = label_vol[idx_p, :, :]
            if m.shape != plane.shape:
                m = cv2.resize(m.astype(np.uint8), (plane.shape[1], plane.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
            plane[m] = label_val
        elif axis == "Y":
            if idx_p < 0 or idx_p >= label_vol.shape[1]:
                return
            plane = label_vol[:, idx_p, :]
            if m.shape != plane.shape:
                m = cv2.resize(m.astype(np.uint8), (plane.shape[1], plane.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
            plane[m] = label_val
        else:
            if idx_p < 0 or idx_p >= label_vol.shape[2]:
                return
            plane = label_vol[:, :, idx_p]
            if m.shape != plane.shape:
                m = cv2.resize(m.astype(np.uint8), (plane.shape[1], plane.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
            plane[m] = label_val

    def _load_saved_ct_annotations_into_preview(self, label_vol: np.ndarray):
        if not self.image_list or label_vol is None:
            return 0
        bd = os.path.dirname(self.image_list[self.current_idx])
        vn = os.path.splitext(os.path.basename(self.image_list[self.current_idx]))[0]
        lbl_dir = os.path.join(bd, "labels_ct_slices")
        if not os.path.isdir(lbl_dir):
            return 0
        pat = re.compile(rf"^{re.escape(vn)}_([XYZ])_(\d+)\.txt$", re.IGNORECASE)
        n_files = 0
        for fn in os.listdir(lbl_dir):
            m = pat.match(fn)
            if not m:
                continue
            axis = m.group(1).upper()
            idx_full = int(m.group(2))
            anns = []
            with open(os.path.join(lbl_dir, fn), encoding="utf-8") as f:
                for line in f:
                    parts = line.split()
                    if len(parts) >= 2:
                        anns.append({"id": int(parts[0]), "coords": [float(x) for x in parts[1:]]})
            if not anns:
                continue
            by_class = {}
            for ann in anns:
                cid = int(ann["id"])
                by_class.setdefault(cid, []).append(ann)
            for cid, group in by_class.items():
                mask_raw = self._annotations_to_raw_slice_mask(group, axis)
                self._paint_mask_into_preview_volume(label_vol, axis, idx_full, mask_raw, cid)
            n_files += 1
        return n_files

    def _merge_recon_slab_into_preview(self, label_vol: np.ndarray):
        if label_vol is None or self.recon_label_volume is None:
            return
        if not np.any(self.recon_label_volume):
            return
        axis = getattr(self, "recon_axis", "X")
        lower = int(getattr(self, "recon_slab_lower", 0))
        recon = np.asarray(self.recon_label_volume)

        if axis == "X":
            for i in range(recon.shape[0]):
                slab = recon[i, :, :]
                for label_val in np.unique(slab):
                    lv = int(label_val)
                    if lv <= 0:
                        continue
                    m = (slab == lv).astype(np.uint8)
                    self._paint_mask_into_preview_volume(label_vol, axis, lower + i, m, lv - 1)
        elif axis == "Y":
            for i in range(recon.shape[1]):
                slab = recon[:, i, :]
                for label_val in np.unique(slab):
                    lv = int(label_val)
                    if lv <= 0:
                        continue
                    m = (slab == lv).astype(np.uint8)
                    self._paint_mask_into_preview_volume(label_vol, axis, lower + i, m, lv - 1)
        else:
            for i in range(recon.shape[2]):
                slab = recon[:, :, i]
                for label_val in np.unique(slab):
                    lv = int(label_val)
                    if lv <= 0:
                        continue
                    m = (slab == lv).astype(np.uint8)
                    self._paint_mask_into_preview_volume(label_vol, axis, lower + i, m, lv - 1)

    def _rebuild_ct_label_volume_preview(self):
        """Build label volume aligned with downsampled ct_volume (for 3D pore overlay)."""
        if not self.is_ct_mode or self.ct_volume is None:
            self.ct_label_volume_preview = None
            return
        label_vol = np.zeros(self.ct_volume.shape, dtype=np.uint8)
        n_saved = self._load_saved_ct_annotations_into_preview(label_vol)
        # Include current slice (even if not yet written to disk)
        if self.raw_image is not None and self.current_annotations:
            by_class = {}
            for ann in self.current_annotations:
                cid = int(ann.get("id", 0))
                by_class.setdefault(cid, []).append(ann)
            axis = self.slice_axis.get()
            for cid, group in by_class.items():
                mask_raw = self._annotations_to_raw_slice_mask(group, axis)
                self._paint_mask_into_preview_volume(
                    label_vol, axis, int(self.current_slice_idx), mask_raw, cid
                )
        self._merge_recon_slab_into_preview(label_vol)
        if np.any(label_vol):
            self.ct_label_volume_preview = label_vol
            n_vox = int(np.count_nonzero(label_vol))
            self.log(
                f"3D pore overlay: {n_vox} labeled voxels from {n_saved} saved slice file(s)",
                "green",
            )
        else:
            self.ct_label_volume_preview = None
            if n_saved == 0:
                self.log("3D pore overlay: no saved labels in labels_ct_slices/", "orange")
        if self.plotter is not None and getattr(self.plotter, "iren", None) and self.plotter.iren.initialized:
            self.update_isosurface_3d()

    def _add_pore_segmentation_meshes_3d(self, plotter=None):
        """Render saved / propagated pore labels on the 3D preview grid."""
        pl = plotter or self.plotter
        if pl is None or not bool(self.show_seg_in_3d.get()):
            return
        label_vol = self.ct_label_volume_preview
        if label_vol is None or not np.any(label_vol):
            return
        for label_val in sorted(int(v) for v in np.unique(label_vol) if v > 0):
            cls_id = label_val - 1
            cls_u8 = ((label_vol == label_val).astype(np.uint8)) * 255
            if not np.any(cls_u8):
                continue
            grid = pv.ImageData()
            grid.dimensions = cls_u8.shape
            grid.spacing = (1, 1, 1)
            grid.origin = (0, 0, 0)
            grid.point_data["values"] = cls_u8.flatten(order="F")
            rgb = self.class_colors.get(cls_id, (255, 60, 60))
            color = (rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0)
            cls_name = self.class_mapping.get(cls_id, f"class_{cls_id}")
            try:
                surf = grid.contour([32, 224])
                if surf.n_points == 0:
                    surf = grid.threshold([1, 255], scalars="values")
                if surf.n_points > 0:
                    pl.add_mesh(
                        surf,
                        color=color,
                        opacity=0.92,
                        name=f"pore_{cls_name}",
                        smooth_shading=True,
                        specular=0.6,
                        show_scalar_bar=False,
                    )
            except Exception:
                try:
                    pl.add_mesh(
                        grid.threshold([1, 255], scalars="values"),
                        color=color,
                        opacity=0.85,
                        name=f"pore_{cls_name}",
                    )
                except Exception:
                    pass

    # ==========================================================================
    # PYVISTA 3D LOGIC
    # ==========================================================================

    @staticmethod
    def _enhance_3d_preview_pore_contrast(vol_u8: np.ndarray) -> np.ndarray:
        """
        Slice-wise CLAHE — matches 2D slice look so pores stay visible in 3D.
        Skipped on very large previews to avoid UI freezes.
        """
        vol = np.ascontiguousarray(vol_u8, dtype=np.uint8)
        if vol.ndim != 3 or vol.size == 0:
            return vol
        if vol.shape[0] > 160 or vol.size > 3_500_000:
            return vol
        try:
            clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
            out = np.empty_like(vol)
            for z in range(vol.shape[0]):
                out[z] = clahe.apply(vol[z])
            return out
        except Exception:
            return vol

    @staticmethod
    def _preview_shape_for_step(ct_dims, step: int):
        step = max(1, int(step))
        return tuple(max(1, (int(d) + step - 1) // step) for d in ct_dims)

    def _compute_ct_preview_step(self, is_txm: bool = False) -> int:
        """Pick downsampling so 3D preview stays responsive (< ~2.5M voxels)."""
        if self.is_h5_lazy_volume:
            target_dim = 96
        elif is_txm:
            target_dim = 128
        elif self.is_h5_fast_volume:
            target_dim = 160
        else:
            target_dim = 192
        max_dim = max(int(x) for x in self.ct_dims)
        step = max(1, max_dim // target_dim)
        for _ in range(40):
            sh = self._preview_shape_for_step(self.ct_dims, step)
            if sh[0] * sh[1] * sh[2] <= self._PV_MAX_VOXELS:
                return step
            step += 1
        return step

    def _build_pv_display_volumes(self, apply_user_window=False, p_lo=None, p_hi=None, gamma=None):
        """
        Prepare CT preview grids for 3D reconstruction.
        Fast path (default open): reuse ct_volume as-is — no duplicate CLAHE.
        """
        vol = np.asanyarray(self.ct_volume, dtype=np.float32)
        vol = np.clip(vol, 0.0, 255.0)

        if not apply_user_window:
            display = vol.copy()
            pore_vol = display.copy()
            try:
                from scipy.ndimage import gaussian_filter
                iso_vol = gaussian_filter(display, sigma=0.35)
            except Exception:
                iso_vol = display.copy()
            self._pv_pore_vol = pore_vol
            return display, iso_vol, 0.0, 255.0

        try:
            p_lo = float(p_lo if p_lo is not None else self.pv_window_lo.get())
            p_hi = float(p_hi if p_hi is not None else self.pv_window_hi.get())
        except Exception:
            p_lo, p_hi = 0.5, 99.5
        p_lo = float(np.clip(p_lo, 0.0, 49.0))
        p_hi = float(np.clip(p_hi, 51.0, 100.0))
        if p_hi <= p_lo + 1.0:
            p_hi = min(100.0, p_lo + 5.0)

        lo, hi = np.percentile(vol, [p_lo, p_hi])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = float(np.min(vol)), float(np.max(vol))
        if hi <= lo:
            hi = lo + 1.0
        stretched = np.clip((vol - lo) / (hi - lo), 0.0, 1.0)

        try:
            gamma = float(gamma if gamma is not None else self.pv_contrast_boost.get())
        except Exception:
            gamma = 1.15
        gamma = float(np.clip(gamma, 0.4, 3.0))
        stretched = np.power(stretched, 1.0 / gamma)
        display = (stretched * 255.0).astype(np.float32)
        display_u8 = np.clip(display, 0, 255).astype(np.uint8)
        display_u8 = self._enhance_3d_preview_pore_contrast(display_u8)
        display = display_u8.astype(np.float32)

        pore_vol = display.copy()
        try:
            from scipy.ndimage import gaussian_filter
            iso_vol = gaussian_filter(display, sigma=0.45)
        except Exception:
            iso_vol = display.copy()
        self._pv_pore_vol = pore_vol
        return display, iso_vol, 0.0, 255.0

    def _snapshot_pv_settings(self):
        return {
            "apply_user_window": False,
            "p_lo": float(self.pv_window_lo.get()),
            "p_hi": float(self.pv_window_hi.get()),
            "gamma": float(self.pv_contrast_boost.get()),
            "mode": self.pv_view_mode.get(),
            "bg_dark": bool(self.pv_bg_dark.get()),
            "cross_sections": bool(self.show_cross_sections_3d.get()),
            "pore_highlight": bool(self.pv_pore_highlight.get()),
            "opacity": float(self.opacity_val.get()),
            "threshold": float(self.threshold_val.get()),
        }

    def _prepare_pv_scene_worker(self, settings: dict) -> dict:
        """Heavy NumPy / VTK prep — runs off the UI thread."""
        display, iso_vol, clim_lo, clim_hi = self._build_pv_display_volumes(
            apply_user_window=bool(settings.get("apply_user_window", False)),
            p_lo=settings.get("p_lo"),
            p_hi=settings.get("p_hi"),
            gamma=settings.get("gamma"),
        )
        suggested_t = self._suggest_iso_threshold(display)
        iso_grid = self._make_pv_image_grid(iso_vol)
        iso_mesh, used_t = self._build_iso_mesh_from_grid(iso_grid, display, threshold=suggested_t)
        return {
            "display": display,
            "iso_vol": iso_vol,
            "pore_vol": np.asanyarray(getattr(self, "_pv_pore_vol", display), dtype=np.float32),
            "clim": (clim_lo, clim_hi),
            "suggested_threshold": used_t if iso_mesh is not None else suggested_t,
            "iso_mesh": iso_mesh,
            "stats": {
                "min": float(np.min(display)),
                "max": float(np.max(display)),
                "mean": float(np.mean(display)),
                "shape": tuple(int(x) for x in display.shape),
            },
            "settings": settings,
        }

    def _finish_pv_launch(self, result: dict, progress=None):
        """Create PyVista window on the UI thread after background prep."""
        if progress is not None:
            try:
                progress.destroy()
            except Exception:
                pass
        self._pv_launch_busy = False

        if result.get("error") is not None:
            self.log(f"3D Init Error: {result['error']}", "red")
            self.plotter = None
            return

        try:
            display = result["display"]
            iso_vol = result["iso_vol"]
            settings = result.get("settings", {})
            self._pv_display_vol = display
            self._pv_iso_vol = iso_vol
            self._pv_pore_vol = result.get("pore_vol", display)
            self._pv_clim = result.get("clim", (0.0, 255.0))

            try:
                self.threshold_val.set(int(round(result.get("suggested_threshold", self.threshold_val.get()))))
            except Exception:
                pass
            if self.opacity_val.get() < 0.75:
                self.opacity_val.set(1.0)

            self.pv_grid = self._make_pv_image_grid(display)
            self.pv_iso_grid = self._make_pv_image_grid(iso_vol)

            stats = result.get("stats", {})
            if stats:
                self.log(
                    f"3D preview stats: {stats.get('shape')}  "
                    f"I={stats.get('min', 0):.0f}–{stats.get('max', 0):.0f} "
                    f"(mean {stats.get('mean', 0):.0f})",
                    "blue",
                )

            mode_l = (settings.get("mode") or self.pv_view_mode.get() or "").lower()
            show_vol, show_surf = self._pv_mode_flags()
            title = "3D CT — volume + surface"
            if show_surf and not show_vol:
                title = "3D CT — reconstruction"
            elif show_vol and "pore" in mode_l and "surface" not in mode_l:
                title = "3D CT — pore detail"
            elif show_vol and not show_surf:
                title = "3D CT — volume render"
            if settings.get("cross_sections") or bool(self.show_cross_sections_3d.get()):
                title += " + slices"

            self.plotter = pv.Plotter(title=title, window_size=(1100, 900))
            self._style_pv_plotter()

            iso_mesh = result.get("iso_mesh")
            has_surface = False
            if show_surf and iso_mesh is not None and iso_mesh.n_points > 0:
                dark = bool(self.pv_bg_dark.get())
                surf_opacity = float(self.opacity_val.get())
                if show_vol:
                    surf_opacity = min(surf_opacity, 0.72)
                self.plotter.add_mesh(
                    iso_mesh,
                    color="#CFD8DC" if dark else "#78909C",
                    opacity=surf_opacity,
                    name="iso",
                    smooth_shading=True,
                    specular=0.55,
                    specular_power=40,
                    ambient=0.25,
                    diffuse=0.88,
                    show_edges=False,
                    show_scalar_bar=False,
                )
                has_surface = True
            elif show_surf:
                self._add_pv_fallback_slice()
                has_surface = True

            if show_vol:
                self._add_volume_render_3d()

            try:
                self.plotter.reset_camera()
            except Exception:
                pass

            self.plotter.show(interactive_update=True, auto_close=False)
            self.set_pv_isometric_view(1, 1, 1)
            self._run_pv_loop()

            if bool(self.pv_pore_highlight.get()):
                self.frame.after(200, self._deferred_pore_highlight)

            if has_surface:
                self.log("3D reconstruction ready — drag to rotate, scroll to zoom.", "green")
            else:
                self.log("3D opened with fallback slice — adjust threshold or mode.", "orange")
            self.frame.after(120, self._rebuild_ct_label_volume_preview)
        except Exception as e:
            self.log(f"3D Init Error: {e}", "red")
            import traceback
            traceback.print_exc()
            self.plotter = None

    def _deferred_pore_highlight(self):
        """Add pore isosurface after the 3D window is visible (avoids open hang)."""
        if self.plotter is None:
            return
        try:
            self._add_pore_highlight_meshes_3d()
            if getattr(self.plotter, "iren", None) and self.plotter.iren.initialized:
                self.plotter.update()
        except Exception as e:
            self.log(f"Pore highlight skipped: {e}", "orange")

    def _deferred_pv_first_render(self):
        """Legacy hook — rendering now happens before show()."""
        if self.plotter is None:
            return
        try:
            self.plotter.update()
        except Exception:
            pass

    def _pv_mode_flags(self):
        """Return (show_volume, show_surface) from UI mode string."""
        mode = (self.pv_view_mode.get() or "Volume + surface").strip().lower()
        if "volume + surface" in mode:
            return True, True
        if "pore detail" in mode:
            return True, True
        if mode == "volume render" or (
            mode.startswith("volume") and "surface" not in mode and "reconstruction" not in mode
        ):
            return True, False
        return False, True

    def _pore_intensity_threshold(self, display_vol=None):
        """Threshold for bright pore regions (matches white spots in 2D slices)."""
        vol = np.asanyarray(display_vol if display_vol is not None else getattr(self, "_pv_display_vol", None))
        if vol is None or vol.size == 0:
            return 165.0
        inside = vol.ravel()[:: max(1, vol.size // 300000)]
        inside = inside[np.isfinite(inside)]
        if inside.size == 0:
            return 165.0
        # Ignore outer air (very dark voxels outside the cylinder)
        material = inside[inside > np.percentile(inside, 8)]
        if material.size < 100:
            material = inside
        p70, p82, p92 = np.percentile(material, [70, 82, 92])
        return float(np.clip(0.55 * p82 + 0.45 * p70, p70 + 4, min(p92, 240)))

    def _build_pv_pore_volume_opacity(self, display_vol=None):
        """Opacity transfer tuned for pore-rich TXM (matrix translucent, pores bright)."""
        op = np.zeros(256, dtype=np.float64)
        op[:6] = 0.0
        op[6:40] = np.linspace(0.04, 0.22, 34)
        op[40:100] = np.linspace(0.22, 0.55, 60)
        op[100:180] = np.linspace(0.55, 0.82, 80)
        op[180:] = np.linspace(0.82, 0.95, 76)
        scale = float(self.opacity_val.get())
        return np.clip(op * max(scale, 0.5), 0.04, 1.0)

    @staticmethod
    def _pv_extract_surface(mesh):
        """Extract surface mesh (PyVista 0.47+ requires explicit algorithm)."""
        if mesh is None:
            return mesh
        try:
            return mesh.extract_surface(algorithm="dataset_surface").clean()
        except TypeError:
            return mesh.extract_surface().clean()

    @staticmethod
    def _make_pv_image_grid(volume: np.ndarray) -> "pv.ImageData":
        """Build PyVista ImageData from a 3D numpy preview (Z,Y,X) stack."""
        vol = np.ascontiguousarray(volume, dtype=np.float32)
        grid = pv.ImageData()
        grid.dimensions = tuple(int(x) for x in vol.shape)
        grid.spacing = (1.0, 1.0, 1.0)
        grid.origin = (0.0, 0.0, 0.0)
        grid.point_data["values"] = vol.flatten(order="F")
        return grid

    def _build_iso_mesh_from_grid(self, iso_grid, display_vol, threshold=None):
        """Marching-cubes on a pre-built grid (safe to call off UI thread)."""
        if iso_grid is None:
            return None, float(threshold or 100)
        if threshold is None:
            threshold = float(self.threshold_val.get())
        candidates = [float(threshold)]
        try:
            t_auto = float(self._suggest_iso_threshold(display_vol))
            candidates.extend([t_auto, t_auto - 20, t_auto + 20, t_auto - 35, 60.0, 80.0, 100.0])
        except Exception:
            candidates.extend([80.0, 60.0, 100.0])
        seen = set()
        thresholds = []
        for t in candidates:
            ti = int(np.clip(round(t), 5, 250))
            if ti not in seen:
                seen.add(ti)
                thresholds.append(float(ti))

        mesh = None
        used_t = thresholds[0]
        for t in thresholds:
            try:
                m = iso_grid.contour([t], progress_bar=False)
            except Exception:
                continue
            if m is not None and m.n_points > 80:
                mesh = m
                used_t = t
                break
        if mesh is None:
            return None, used_t
        try:
            mesh = self._pv_extract_surface(mesh)
            mesh = mesh.compute_normals(
                cell_normals=False,
                point_normals=True,
                inplace=False,
                feature_angle=60,
            )
        except Exception:
            try:
                mesh = self._pv_extract_surface(mesh)
            except Exception:
                pass
        try:
            if mesh.n_cells > 350_000:
                mesh = mesh.decimate_pro(0.4, progress_bar=False).clean()
        except Exception:
            pass
        return mesh, used_t

    def _build_reconstruction_mesh(self, iso_src, threshold=None):
        """Marching-cubes surface; tries nearby thresholds if the first is empty."""
        display = getattr(self, "_pv_display_vol", None)
        if threshold is None:
            threshold = float(self.threshold_val.get())
        return self._build_iso_mesh_from_grid(iso_src, display, threshold=threshold)

    def _add_pv_fallback_slice(self):
        """Guaranteed visible mid-slice if volume/surface fail."""
        if self.plotter is None or self.pv_grid is None:
            return
        try:
            self.plotter.remove_actor("fallback_slice")
        except Exception:
            pass
        try:
            d0, d1, d2 = (int(x) for x in self.pv_grid.dimensions)
            origin = (float(d0 // 2), float(d1 // 2), float(d2 // 2))
            plane = self.pv_grid.slice(normal="x", origin=origin)
            dark = bool(self.pv_bg_dark.get())
            self.plotter.add_mesh(
                plane,
                scalars="values",
                cmap="bone" if dark else "gray",
                clim=list(getattr(self, "_pv_clim", (0.0, 255.0))),
                opacity=0.95,
                name="fallback_slice",
                show_scalar_bar=False,
                lighting=False,
                ambient=1.0,
            )
        except Exception:
            pass

    def _build_pv_volume_opacity(self):
        """Opacity transfer for volume modes (pore-aware by default)."""
        mode = (self.pv_view_mode.get() or "").lower()
        display = getattr(self, "_pv_display_vol", None)
        if "pore detail" in mode or display is not None:
            return self._build_pv_pore_volume_opacity(display)
        op = np.zeros(256, dtype=np.float64)
        op[:8] = 0.0
        op[8:35] = np.linspace(0.0, 0.08, 27)
        op[35:90] = np.linspace(0.08, 0.45, 55)
        op[90:150] = np.linspace(0.45, 0.78, 60)
        op[150:] = np.linspace(0.78, 0.92, 106)
        scale = float(self.opacity_val.get())
        return np.clip(op * max(scale, 0.35), 0.0, 1.0)

    def _add_pore_highlight_meshes_3d(self):
        """Iso-surface of bright voxels to show pore cavities in 3D."""
        if self.plotter is None or not bool(self.pv_pore_highlight.get()):
            return
        if self.pv_grid is None:
            return
        n_vox = int(np.prod(self.pv_grid.dimensions))
        if n_vox > 3_500_000:
            return
        pore_src = getattr(self, "_pv_pore_vol", None)
        if pore_src is None:
            pore_src = getattr(self, "_pv_display_vol", None)
        if pore_src is None:
            return
        try:
            self.plotter.remove_actor("pore_hi")
        except Exception:
            pass
        try:
            grid = self.pv_grid
            t_pore = self._pore_intensity_threshold(pore_src)
            mesh = grid.contour([t_pore], progress_bar=False)
            if mesh is None or mesh.n_points < 40:
                return
            try:
                mesh = self._pv_extract_surface(mesh)
                if mesh.n_cells > 250_000:
                    mesh = mesh.decimate_pro(0.5, progress_bar=False).clean()
            except Exception:
                pass
            dark = bool(self.pv_bg_dark.get())
            self.plotter.add_mesh(
                mesh,
                color="#FFFFFF" if dark else "#37474F",
                opacity=0.32 if dark else 0.40,
                name="pore_hi",
                smooth_shading=True,
                specular=0.7,
                specular_power=50,
                ambient=0.35,
                diffuse=0.75,
                show_scalar_bar=False,
            )
        except Exception:
            pass

    def _suggest_iso_threshold(self, display_vol):
        """Pick a threshold near the solid/background transition."""
        sample = display_vol.ravel()[:: max(1, display_vol.size // 250000)]
        sample = sample[np.isfinite(sample)]
        if sample.size == 0:
            return 85
        p10, p25, p50, p75, p90 = np.percentile(sample, [10, 25, 50, 75, 90])
        # Favor the material band, not the outer air cylinder
        t = 0.35 * p25 + 0.65 * p50
        if p90 - p10 > 60:
            t = max(t, p10 + 0.30 * (p90 - p10))
        return int(np.clip(round(t), 35, 210))

    def _remove_pv_volume_actor(self):
        if self.plotter is None:
            return
        if getattr(self, "_pv_vol_actor", None) is not None:
            try:
                self.plotter.remove_actor(self._pv_vol_actor)
            except Exception:
                pass
            self._pv_vol_actor = None
        try:
            self.plotter.remove_volume_render()
        except Exception:
            pass

    def _add_volume_render_3d(self):
        """Optional ray-cast volume overlay."""
        if self.plotter is None or self.pv_grid is None:
            return
        self._remove_pv_volume_actor()
        show_vol, _show_surf = self._pv_mode_flags()
        if not show_vol:
            return
        clim = getattr(self, "_pv_clim", (0.0, 255.0))
        cmap = "bone" if bool(self.pv_bg_dark.get()) else "gray"
        opacity = self._build_pv_volume_opacity()
        for kwargs in (
            {"mapper": "smart"},
            {},
        ):
            try:
                self._pv_vol_actor = self.plotter.add_volume(
                    self.pv_grid,
                    scalars="values",
                    cmap=cmap,
                    opacity=opacity,
                    shade=True,
                    clim=list(clim),
                    show_scalar_bar=False,
                    blending="composite",
                    **kwargs,
                )
                return
            except TypeError:
                continue
            except Exception:
                continue
        try:
            self._pv_vol_actor = self.plotter.add_volume(
                self.pv_grid,
                scalars="values",
                cmap=cmap,
                opacity="linear",
                shade=True,
                clim=list(clim),
                show_scalar_bar=False,
            )
        except Exception as e:
            self.log(f"Volume render unavailable: {e}", "orange")

    def _apply_pv_background(self):
        """Toggle black/white background on the open 3D window."""
        if self.plotter is None:
            return
        self._style_pv_plotter()
        try:
            if self.plotter.iren.initialized:
                self.plotter.update()
        except Exception:
            pass

    def _style_pv_plotter(self):
        """Background + axes + lighting for CT isosurface."""
        if self.plotter is None:
            return
        dark = bool(self.pv_bg_dark.get())
        if dark:
            try:
                self.plotter.set_background("#0b0d10", top="#1a1f27")
            except TypeError:
                self.plotter.set_background("black")
            axis_color = "white"
        else:
            try:
                self.plotter.set_background("white", top="#e8eaed")
            except TypeError:
                self.plotter.set_background("white")
            axis_color = "black"
        try:
            self.plotter.remove_bounds_axes()
        except Exception:
            pass
        try:
            self.plotter.add_axes(
                line_width=2,
                color=axis_color,
                xlabel="X",
                ylabel="Y",
                zlabel="Z",
            )
        except TypeError:
            try:
                self.plotter.add_axes()
            except Exception:
                pass
        try:
            self.plotter.enable_3_lights()
        except Exception:
            pass

    def _refresh_pv_contrast(self):
        """Rebuild display grids from current contrast settings (if 3D window open)."""
        if self.plotter is None or self.ct_volume is None:
            return
        if not HAS_PYVISTA:
            return
        try:
            display, iso_vol, clim_lo, clim_hi = self._build_pv_display_volumes(apply_user_window=True)
            self._pv_display_vol = display
            self._pv_iso_vol = iso_vol
            self._pv_clim = (clim_lo, clim_hi)

            self.pv_grid = self._make_pv_image_grid(display)
            self.pv_iso_grid = self._make_pv_image_grid(iso_vol)

            self._remove_pv_volume_actor()
            self.update_isosurface_3d(reset_cam=False)
            self.log(
                f"3D contrast updated (p{self.pv_window_lo.get():.1f}–p{self.pv_window_hi.get():.1f}, "
                f"boost={self.pv_contrast_boost.get():.2f})",
                "blue",
            )
        except Exception as e:
            self.log(f"Contrast refresh failed: {e}", "orange")

    def _remove_pv_cross_section_actors(self):
        if self.plotter is None:
            return
        for name in ("mid_xy", "mid_xz", "mid_yz"):
            try:
                self.plotter.remove_actor(name)
            except Exception:
                pass

    def _pv_scene_center_and_radius(self):
        """Center and camera distance for the downsampled preview volume."""
        if self.pv_grid is not None:
            dims = np.array(self.pv_grid.dimensions, dtype=np.float64)
        elif self.ct_volume is not None:
            dims = np.array(self.ct_volume.shape, dtype=np.float64)
        else:
            dims = np.array([1.0, 1.0, 1.0])
        center = (dims - 1.0) * 0.5
        radius = float(np.linalg.norm(dims) * 0.55 + 1.0)
        return center, radius

    @staticmethod
    def _rodrigues_rotate(vec, axis, degrees: float):
        v = np.asarray(vec, dtype=np.float64)
        ax = np.asarray(axis, dtype=np.float64)
        n = np.linalg.norm(ax)
        if n < 1e-9:
            return v
        ax = ax / n
        rad = np.deg2rad(float(degrees))
        c, s = np.cos(rad), np.sin(rad)
        return v * c + np.cross(ax, v) * s + ax * np.dot(ax, v) * (1.0 - c)

    def _set_pv_camera(self, position, focal_point, view_up):
        """Set camera pose using PyVista 0.47+ safe APIs (camera.up, not view_up)."""
        if self.plotter is None:
            return
        pos = np.asarray(position, dtype=float).tolist()
        fp = np.asarray(focal_point, dtype=float).tolist()
        up = np.asarray(view_up, dtype=float).tolist()
        try:
            # Preferred: (position, focal_point, view_up) tuple
            self.plotter.camera_position = (pos, fp, up)
        except Exception:
            cam = self.plotter.camera
            cam.position = pos
            cam.focal_point = fp
            # PyVista >=0.47: use .up (VTK .view_up is blocked)
            if hasattr(cam, "up"):
                cam.up = up
            elif hasattr(cam, "SetViewUp"):
                cam.SetViewUp(*up)
        try:
            self.plotter.reset_camera_clipping_range()
        except Exception:
            pass
        if getattr(self.plotter, "iren", None) and self.plotter.iren.initialized:
            self.plotter.update()

    def set_pv_isometric_view(self, sign_x=1, sign_y=1, sign_z=1):
        """Place camera on a true isometric corner (equal axis foreshortening)."""
        if self.plotter is None:
            self.log("Open the 3D window first, then pick an isometric view.", "orange")
            return
        try:
            sx, sy, sz = int(np.sign(sign_x) or 1), int(np.sign(sign_y) or 1), int(np.sign(sign_z) or 1)
            self._pv_iso_signs = (sx, sy, sz)
            center, radius = self._pv_scene_center_and_radius()
            direction = np.array([sx, sy, sz], dtype=np.float64)
            direction = direction / max(np.linalg.norm(direction), 1e-9)
            cam_pos = center + direction * (radius * 2.2)
            if abs(direction[2]) > 0.85:
                up = np.array([0.0, 1.0, 0.0])
            else:
                up = np.array([0.0, 0.0, 1.0])
            self._set_pv_camera(cam_pos, center, up)
            self.log(f"Isometric view: ({sx:+d}, {sy:+d}, {sz:+d})", "blue")
        except Exception as e:
            try:
                self.plotter.reset_camera()
            except Exception:
                pass
            self.log(f"Isometric camera fallback ({e})", "orange")

    def flip_pv_isometric_axis(self, axis: str):
        """Mirror the current isometric view across X / Y / Z."""
        sx, sy, sz = self._pv_iso_signs
        ax = (axis or "").lower()
        if ax == "x":
            sx *= -1
        elif ax == "y":
            sy *= -1
        elif ax == "z":
            sz *= -1
        else:
            return
        self.set_pv_isometric_view(sx, sy, sz)

    def rotate_pv_camera_90(self, axis: str):
        """Orbit the camera 90° about the volume center on a world axis."""
        if self.plotter is None:
            self.log("Open the 3D window first, then rotate the view.", "orange")
            return
        ax = (axis or "Z").upper()
        axis_vec = {"X": (1, 0, 0), "Y": (0, 1, 0), "Z": (0, 0, 1)}.get(ax, (0, 0, 1))
        cam = self.plotter.camera
        fp = np.array(cam.focal_point, dtype=np.float64)
        pos = np.array(cam.position, dtype=np.float64)
        # Prefer PyVista API (.up); fall back to VTK getter
        if hasattr(cam, "up"):
            up = np.array(cam.up, dtype=np.float64)
        else:
            up = np.array(cam.GetViewUp(), dtype=np.float64)
        rel = pos - fp
        new_rel = self._rodrigues_rotate(rel, axis_vec, 90)
        new_up = self._rodrigues_rotate(up, axis_vec, 90)
        if np.linalg.norm(new_up) < 1e-6:
            new_up = np.array([0.0, 0.0, 1.0])
        else:
            new_up = new_up / np.linalg.norm(new_up)
        self._set_pv_camera(fp + new_rel, fp, new_up)
        self.log(f"Rotated camera 90° about {ax}", "blue")

    def launch_pyvista_window(self):
        if not HAS_PYVISTA or self.ct_volume is None:
            return
        if getattr(self, "_pv_launch_busy", False):
            self.log("3D viewer is already preparing…", "orange")
            return

        if self.plotter is not None:
            try:
                self.plotter.close()
            except Exception:
                pass
            self.plotter = None

        self._pv_launch_busy = True
        settings = self._snapshot_pv_settings()

        progress = Toplevel(self.frame)
        progress.title("3D Viewer")
        progress.geometry("420x130")
        progress.transient(self.frame)
        progress.resizable(False, False)
        try:
            progress.grab_set()
        except tk.TclError:
            pass
        tk.Label(progress, text="Preparing 3D reconstruction…", font=("Arial", 11, "bold")).pack(pady=(18, 6))
        pv_lbl = tk.Label(
            progress,
            text="Building preview volume (keeps UI responsive)…",
            font=("Arial", 9),
            wraplength=380,
        )
        pv_lbl.pack(pady=4)
        progress.update_idletasks()

        def _worker():
            payload = {"error": None}
            try:
                payload.update(self._prepare_pv_scene_worker(settings))
            except Exception as e:
                payload["error"] = e
            self.frame.after(0, lambda: self._finish_pv_launch(payload, progress))

        threading.Thread(target=_worker, daemon=True).start()

    def _run_pv_loop(self):
        """
        Critical Fix: This function runs every 10ms to update the 3D window.
        Without this, the 3D window opens but freezes/hangs (White Screen).
        """
        if self.plotter is None: return
        
        try:
            # Update the PyVista render window
            self.plotter.update()
            
            # If the window is still open, schedule the next update
            if not self.plotter.iframe.is_destroyed(): 
                self.frame.after(10, self._run_pv_loop)
            else:
                self.plotter = None
        except:
            self.plotter = None

    def _add_mid_cross_sections_3d(self):
        """Mid XY / XZ / YZ planes — optional, contrast-windowed grayscale."""
        if self.plotter is None or self.pv_grid is None or self.ct_volume is None:
            return
        if not bool(self.show_cross_sections_3d.get()):
            self._remove_pv_cross_section_actors()
            return

        d0, d1, d2 = (int(x) for x in self.pv_grid.dimensions)
        m0, m1, m2 = d0 // 2, d1 // 2, d2 // 2
        origin = (float(m0), float(m1), float(m2))

        plane_specs = (
            ("mid_xy", "x", origin),
            ("mid_xz", "y", origin),
            ("mid_yz", "z", origin),
        )
        dark = bool(self.pv_bg_dark.get())
        cmap = "bone" if dark else "gray"
        clim_use = list(getattr(self, "_pv_clim", (0.0, 255.0)))
        for name, normal, org in plane_specs:
            try:
                plane = self.pv_grid.slice(normal=normal, origin=org)
                self.plotter.add_mesh(
                    plane,
                    scalars="values",
                    cmap=cmap,
                    clim=clim_use,
                    opacity=0.55 if dark else 0.5,
                    name=name,
                    show_scalar_bar=False,
                    lighting=False,
                    interpolate_before_map=True,
                    ambient=1.0,
                )
            except Exception:
                pass

    def update_isosurface_3d(self, reset_cam=False):
        """Refresh 3D reconstruction surface / optional volume / slices."""
        if self.plotter is None or self.pv_grid is None:
            return

        show_vol, show_surf = self._pv_mode_flags()
        
        try:
            if show_vol:
                self._add_volume_render_3d()
            else:
                self._remove_pv_volume_actor()

            if bool(self.show_cross_sections_3d.get()):
                self._add_mid_cross_sections_3d()
            else:
                self._remove_pv_cross_section_actors()

            try:
                self.plotter.remove_actor("iso")
            except Exception:
                pass
            try:
                self.plotter.remove_actor("pore_hi")
            except Exception:
                pass

            if show_surf:
                iso_src = getattr(self, "pv_iso_grid", None) or self.pv_grid
                mesh, used_t = self._build_reconstruction_mesh(
                    iso_src, threshold=float(self.threshold_val.get())
                )
                if mesh is not None and mesh.n_points > 0:
                    if abs(used_t - float(self.threshold_val.get())) > 0.5:
                        try:
                            self.threshold_val.set(int(round(used_t)))
                        except Exception:
                            pass
                    dark = bool(self.pv_bg_dark.get())
                    surf_color = "#CFD8DC" if dark else "#78909C"
                    surf_opacity = float(self.opacity_val.get())
                    if show_vol:
                        surf_opacity = min(surf_opacity, 0.65)
                    self.plotter.add_mesh(
                        mesh,
                        color=surf_color,
                        opacity=surf_opacity,
                        name="iso",
                        smooth_shading=True,
                        specular=0.55,
                        specular_power=40,
                        ambient=0.22,
                        diffuse=0.88,
                        show_edges=False,
                        show_scalar_bar=False,
                    )
                else:
                    self.log(
                        "No 3D surface at this threshold — showing fallback slice.",
                        "orange",
                    )
                    self._add_pv_fallback_slice()

            if bool(self.pv_pore_highlight.get()):
                self._add_pore_highlight_meshes_3d()
            else:
                try:
                    self.plotter.remove_actor("pore_hi")
                except Exception:
                    pass

            self._add_pore_segmentation_meshes_3d()
            
            if reset_cam:
                self.set_pv_isometric_view(*self._pv_iso_signs)
            if self.plotter.iren.initialized:
                self.plotter.update()
        except Exception as e:
            self.log(f"3D update error: {e}", "orange")
            try:
                if show_vol:
                    self._add_volume_render_3d()
                if bool(self.show_cross_sections_3d.get()):
                    self._add_mid_cross_sections_3d()
                else:
                    self._remove_pv_cross_section_actors()
                self._add_pore_segmentation_meshes_3d()
                if self.plotter.iren.initialized:
                    self.plotter.update()
            except Exception:
                pass

    def update_slice_plane_3d(self):
        """Fast operation: Called when Slicing."""
        if self.plotter is None or self.pv_grid is None: return
        
        try:
            idx = self.current_slice_idx
            axis = self.slice_axis.get()
            
            if axis == 'Z': 
                safe_idx = min(max(0, idx), self.ct_dims[2]-1)
                plane = self.pv_grid.slice(normal='z', origin=(0, 0, safe_idx))
            elif axis == 'Y': 
                safe_idx = min(max(0, idx), self.ct_dims[1]-1)
                plane = self.pv_grid.slice(normal='y', origin=(0, safe_idx, 0))
            else: 
                safe_idx = min(max(0, idx), self.ct_dims[0]-1)
                plane = self.pv_grid.slice(normal='x', origin=(safe_idx, 0, 0))
                
            # use name="slice" so PyVista replaces it instantly
            self.plotter.add_mesh(
                plane,
                scalars="values",
                cmap="bone",
                clim=[5.0, 250.0],
                opacity=0.95,
                lighting=False,
                name="slice",
                show_scalar_bar=False,
            )
            
            if self.plotter.iren.initialized: self.plotter.update()
        except: pass

    # ==========================================================================
    # DISPLAY LOGIC (2D & CT)
    # ==========================================================================
    def _set_h5_info_text(self, text):
        self.h5_info_var.set(text)

    def _extract_h5_metadata_text(self, h5_file, dataset, dataset_key):
        lines = []
        shape = tuple(int(x) for x in dataset.shape)
        dtype = str(dataset.dtype)
        lines.append(f"Dataset: {dataset_key}")
        lines.append(f"Shape: {shape} | dtype: {dtype}")

        # Try to read standard projection metadata if present.
        has_theta = "exchange/theta" in h5_file
        has_exchange_data = "exchange/data" in h5_file
        if has_theta and has_exchange_data:
            try:
                theta = np.asarray(h5_file["exchange/theta"][:], dtype=np.float32).ravel()
                proj_shape = h5_file["exchange/data"].shape
                if theta.size > 1:
                    theta_min = float(np.min(theta))
                    theta_max = float(np.max(theta))
                    is_rad = theta_max <= (2.0 * np.pi + 1.0)
                    theta_deg = theta * (180.0 / np.pi) if is_rad else theta
                    step_deg = float(np.median(np.abs(np.diff(theta_deg))))
                    frames_per_180 = int(np.round(180.0 / max(step_deg, 1e-6)))
                    est_steps = int(proj_shape[0] // max(frames_per_180, 1))
                    center_hint = proj_shape[2] / 2.0 if len(proj_shape) >= 3 else None

                    lines.append(f"Theta: {theta.size} angles ({theta_deg.min():.2f} to {theta_deg.max():.2f} deg)")
                    lines.append(f"Step: ~{step_deg:.4f} deg | Frames/180: ~{frames_per_180}")
                    lines.append(f"Estimated time steps: ~{est_steps}")
                    if center_hint is not None:
                        lines.append(f"Center-of-rotation hint: x ~= {center_hint:.2f} px")
            except Exception:
                lines.append("Theta metadata: present but failed to parse")

        # Look for explicit center-of-rotation attributes on file/dataset.
        cor_value = None
        cor_keys = ("center", "rot", "rotation", "center_of_rotation", "rotation_center")
        for attrs in (dataset.attrs, h5_file.attrs):
            for k in attrs.keys():
                lk = str(k).lower()
                if any(tag in lk for tag in cor_keys):
                    try:
                        raw_val = attrs[k]
                        if np.isscalar(raw_val):
                            cor_value = float(raw_val)
                        else:
                            arr = np.asarray(raw_val).ravel()
                            if arr.size == 1:
                                cor_value = float(arr[0])
                        if cor_value is not None:
                            lines.append(f"Center-of-rotation (metadata): {cor_value:.4f}")
                            break
                    except Exception:
                        continue
            if cor_value is not None:
                break

        if cor_value is None and len(shape) >= 3:
            lines.append(f"Center-of-rotation fallback: x ~= {shape[2] / 2.0:.2f} px")

        return "\n".join(lines)

    def _get_h5_center_hint(self, h5_file, dataset):
        cor_keys = ("center", "rot", "rotation", "center_of_rotation", "rotation_center")
        for attrs in (dataset.attrs, h5_file.attrs):
            for k in attrs.keys():
                lk = str(k).lower()
                if any(tag in lk for tag in cor_keys):
                    try:
                        raw_val = attrs[k]
                        if np.isscalar(raw_val):
                            return float(raw_val)
                        arr = np.asarray(raw_val).ravel()
                        if arr.size == 1:
                            return float(arr[0])
                    except Exception:
                        continue
        return float(dataset.shape[2] / 2.0)

    def _get_h5_sampling_config(self):
        mode = self.h5_sampling_mode.get()
        if mode == "Level 1 - Fastest (Highest Downsampling)":
            return {"target_dim": 96, "target_angles": 180, "target_h": 96, "target_w": 96}
        if mode == "Level 2 - Fast":
            return {"target_dim": 112, "target_angles": 210, "target_h": 112, "target_w": 112}
        if mode == "Level 4 - High Resolution":
            return {"target_dim": 160, "target_angles": 300, "target_h": 160, "target_w": 160}
        if mode == "Level 5 - Ultra Resolution (Lowest Downsampling)":
            return {"target_dim": 192, "target_angles": 360, "target_h": 192, "target_w": 192}
        # Level 3 default
        return {"target_dim": 128, "target_angles": 240, "target_h": 128, "target_w": 128}

    def _reconstruct_h5_from_metadata_fast(self, h5_file, dataset, sampling_cfg):
        self.h5_recon_backend = "N/A"
        if ("exchange/theta" not in h5_file) or (not (HAS_TOMOCUPY or HAS_TOMOPY)):
            return None

        theta = np.asarray(h5_file["exchange/theta"][:], dtype=np.float32).ravel()
        if theta.size < 8 or dataset.ndim != 3:
            return None

        # Extreme downsampling for fast interactive browsing.
        n_proj, det_h, det_w = dataset.shape
        target_angles = int(sampling_cfg["target_angles"])
        target_h = int(sampling_cfg["target_h"])
        target_w = int(sampling_cfg["target_w"])
        angle_step = max(1, n_proj // target_angles)
        row_step = max(1, det_h // target_h)
        col_step = max(1, det_w // target_w)

        prj = np.asarray(dataset[::angle_step, ::row_step, ::col_step], dtype=np.float32)
        theta_sub = theta[::angle_step]
        if prj.shape[0] < 8:
            return None

        # Convert to radians if needed.
        if float(np.max(np.abs(theta_sub))) > (2.0 * np.pi + 1.0):
            theta_sub = theta_sub * (np.pi / 180.0)

        # Lightweight normalization/log transform for stable reconstruction.
        p99 = np.percentile(prj, 99.5, axis=(1, 2), keepdims=True)
        p99 = np.maximum(p99, 1e-6)
        prj = np.clip(prj / p99, 1e-6, None)
        prj = -np.log(prj)
        prj = np.ascontiguousarray(prj, dtype=np.float32)
        theta_sub = np.ascontiguousarray(theta_sub, dtype=np.float32)

        center_hint = self._get_h5_center_hint(h5_file, dataset)
        center_scaled = center_hint / float(col_step)

        recon = None
        # Prefer tomocupy (GPU) when available.
        if HAS_TOMOCUPY:
            try:
                if hasattr(tomocupy, "recon"):
                    recon = tomocupy.recon(prj, theta_sub, center=center_scaled)
                    if recon is not None:
                        self.h5_recon_backend = "tomocupy (GPU)"
                elif hasattr(tomocupy, "reconstruct"):
                    recon = tomocupy.reconstruct(prj, theta_sub, center=center_scaled)
                    if recon is not None:
                        self.h5_recon_backend = "tomocupy (GPU)"
            except Exception:
                recon = None

        # Stable CPU fallback path.
        if recon is None and HAS_TOMOPY:
            try:
                recon = tomopy.recon(prj, theta_sub, center=center_scaled, algorithm='gridrec', filter_name='parzen', ncore=1)
                recon = tomopy.circ_mask(recon, axis=0, ratio=0.95, ncore=1)
                self.h5_recon_backend = "tomopy (CPU)"
            except Exception:
                return None

        recon = np.asarray(recon, dtype=np.float32)
        if recon.ndim == 3:
            # Basic cylindrical mask (fast replacement for tomopy.circ_mask).
            nz, ny, nx = recon.shape
            yy, xx = np.ogrid[:ny, :nx]
            cy, cx = ny / 2.0, nx / 2.0
            rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
            mask = rr <= (min(nx, ny) * 0.475)
            recon = recon * mask[np.newaxis, :, :]
        return recon

    def _h5_is_reconstructed_volume(self, h5_file, dataset, dataset_key, path=""):
        """Detect TomoBank / Tomocupy reconstructed volumes vs projection stacks."""
        if dataset is None or getattr(dataset, "ndim", 0) != 3:
            return False
        if HAS_TOMOCUPY_VOLUME and _h5_dataset_is_tomocupy_recon(h5_file):
            return True
        # Projection stacks normally store angles; reconstructed cubes usually do not.
        if "exchange/theta" in h5_file:
            return False
        name = os.path.basename(path).lower()
        if "_rec" in name or name.endswith("_rec.h5") or "tomocupy_out" in name:
            return True
        s = tuple(int(x) for x in dataset.shape)
        # Typical reconstructed cube: last two dims equal (square FOV)
        if s[1] == s[2] and min(s) >= 64:
            return True
        if dataset_key in ("volume", "reconstruction", "data") and "exchange/theta" not in h5_file:
            return True
        return False

    def load_current_ct_volume(self):
        # 1. CLEANUP (Prevents memory leaks from previous loads)
        if hasattr(self, 'h5_handle') and self.h5_handle:
            try: self.h5_handle.close()
            except: pass
            self.h5_handle = None
        if getattr(self, "txm_handle", None) is not None:
            try:
                self.txm_handle.close()
            except Exception:
                pass
            self.txm_handle = None
        self.current_ct_dataset_key = "N/A"
        self.is_h5_fast_volume = False
        self.is_h5_lazy_volume = False
        self.is_tiff_stack_volume = False
        self.h5_recon_backend = "N/A"
        self._set_recon_backend_status("Backend: loading...", "#7D6608")
        self._set_h5_info_text("Volume metadata will appear here after loading .h5 / .txm / .nii.")
        
        self.ct_volume = None 
        self.ct_proxy = None
        import gc; gc.collect()

        path = self.image_list[self.current_idx]
        path = os.path.abspath(path)
        if HAS_TOMOCUPY_VOLUME and os.path.isdir(path):
            resolved = find_tomocupy_reconstruction(path)
            if resolved:
                path, _kind = resolved
                self.image_list[self.current_idx] = path
        fname = os.path.basename(path)
        low = path.lower()
        
        self.log(f"Smart Loading {fname}...", "orange")
        self.canvas.delete("all")
        self.canvas.create_text(
            200, 200,
            text="LOADING CT VOLUME…\nPreparing cross-section viewer…",
            fill="white", font=("Arial", 14),
        )
        self.frame.update_idletasks()
        
        try:
            # 2. FILE OPENING — Tomocupy TIFF stack or single try-recon slice
            if HAS_TOMOCUPY_VOLUME and os.path.isdir(path):
                stack_paths = list_recon_tiff_stack(path)
                if len(stack_paths) >= 2:
                    self.ct_proxy = TiffStackVolume(stack_paths)
                    self.ct_dims = tuple(int(x) for x in self.ct_proxy.shape)
                    self.is_h5_lazy_volume = True
                    self.is_tiff_stack_volume = True
                    self.current_ct_dataset_key = "Tomocupy/recon_*.tiff"
                    meta = (
                        f"Tomocupy TIFF stack\nFolder: {path}\n"
                        f"Slices: {len(stack_paths)} | shape {self.ct_dims}"
                    )
                    self._set_h5_info_text(meta)
                    self._set_recon_backend_status("Backend: Tomocupy TIFF stack (lazy)", "#117A65")
                else:
                    raise Exception("Folder is not a Tomocupy recon TIFF stack (need recon_*.tiff)")

            elif HAS_TOMOCUPY_VOLUME and low.endswith((".tif", ".tiff")):
                plane = _read_tiff_plane(path)
                if plane.ndim != 2:
                    raise Exception("Expected a 2D reconstruction TIFF")
                self.ct_proxy = plane[np.newaxis, :, :]
                self.ct_dims = (1, int(plane.shape[0]), int(plane.shape[1]))
                self.is_h5_fast_volume = True
                self.is_h5_lazy_volume = False
                self.current_ct_dataset_key = "Tomocupy/try_slice"
                self._set_h5_info_text(
                    f"Tomocupy try-recon slice\nFile: {fname}\nShape: {plane.shape}"
                )
                self._set_recon_backend_status("Backend: Tomocupy try TIFF (2D)", "#117A65")

            elif low.endswith((".h5", ".hdf5")):
                if not HAS_H5PY:
                    raise ImportError("h5py not installed")
                self.h5_handle = h5py.File(path, 'r')
                
                # Auto-find the dataset key
                dataset = None
                dataset_key = "unknown"
                for k in ['data', 'exchange/data', 'volume', 'reconstruction']:
                    if k in self.h5_handle:
                        dataset = self.h5_handle[k]
                        dataset_key = k
                        break
                
                # Fallback search if standard keys fail
                if dataset is None:
                    def find_3d(name, node):
                        nonlocal dataset, dataset_key
                        if dataset: return
                        if isinstance(node, h5py.Dataset) and node.ndim == 3:
                            dataset = node
                            dataset_key = name
                    self.h5_handle.visititems(find_3d)

                if dataset is None: raise Exception("No 3D dataset found")
                self.current_ct_dataset_key = dataset_key
                h5_meta_text = self._extract_h5_metadata_text(self.h5_handle, dataset, dataset_key)
                sampling_cfg = self._get_h5_sampling_config()
                prefer_lazy = "Full-res" in self.h5_slice_mode.get()
                is_recon = self._h5_is_reconstructed_volume(
                    self.h5_handle, dataset, dataset_key, path=path
                )

                # Metadata-guided fast reconstruction for projection stacks.
                recon_fast = None
                if dataset_key == "exchange/data" and not is_recon:
                    recon_fast = self._reconstruct_h5_from_metadata_fast(self.h5_handle, dataset, sampling_cfg)

                if recon_fast is not None:
                    self.ct_proxy = recon_fast
                    self.ct_dims = self.ct_proxy.shape
                    self.is_h5_fast_volume = True
                    self.is_h5_lazy_volume = False
                    h5_meta_text += f"\nResolution mode: {self.h5_sampling_mode.get()}"
                    h5_meta_text += "\nMode: metadata-guided reconstruction"
                    h5_meta_text += f"\nReconstruction backend: {self.h5_recon_backend}"
                    self._set_recon_backend_status(f"Backend: {self.h5_recon_backend}", "#1F618D")
                elif is_recon and prefer_lazy:
                    # Reconstructed TomoBank / Tomocupy volume: lazy planes from disk.
                    self.ct_proxy = dataset
                    self.ct_dims = tuple(int(x) for x in dataset.shape)
                    self.is_h5_fast_volume = False
                    self.is_h5_lazy_volume = True
                    h5_meta_text += f"\nVolume shape: {self.ct_dims}"
                    h5_meta_text += "\nMode: reconstructed volume — full-res lazy XY/XZ/YZ slices"
                    if HAS_TOMOCUPY_VOLUME and _h5_dataset_is_tomocupy_recon(self.h5_handle):
                        h5_meta_text += "\nSource: Tomocupy GPU reconstruction"
                        self._set_recon_backend_status("Backend: Tomocupy recon H5 (lazy)", "#117A65")
                    else:
                        h5_meta_text += "\nTip: turn on 'Small pores' + select class 'Gas-Pore'/'Bubble' for SAM"
                        self._set_recon_backend_status("Backend: lazy H5 reconstructed volume", "#117A65")
                else:
                    # Fallback: configurable in-memory downsampling for slicing.
                    target_dim_h5 = int(sampling_cfg["target_dim"])
                    max_dim_h5 = max(dataset.shape)
                    step_h5 = max(1, max_dim_h5 // target_dim_h5)
                    self.log(f"Downsampling volume preview by {step_h5}x…", "blue")
                    self.ct_proxy = np.asarray(dataset[::step_h5, ::step_h5, ::step_h5], dtype=np.float32)
                    self.ct_dims = self.ct_proxy.shape
                    self.is_h5_fast_volume = True
                    self.is_h5_lazy_volume = False
                    h5_meta_text += f"\nResolution mode: {self.h5_sampling_mode.get()}"
                    h5_meta_text += f"\nMode: direct downsample preview (1/{step_h5})"
                    h5_meta_text += "\nReconstruction backend: preview only (tomocupy/tomopy not used)"
                    self._set_recon_backend_status("Backend: H5 preview only", "#7D3C98")

                self._set_h5_info_text(h5_meta_text)

            elif low.endswith(".txm"):
                if not HAS_OLEFILE or open_txm_volume is None:
                    raise ImportError("olefile / txm_volume.py required for .txm\n  pip install olefile")

                def _txm_progress(i, n):
                    self.log(f"Loading TXM volume… {i}/{n}", "blue")
                    if i == 1 or i == n or (i % 25 == 0):
                        self.frame.update_idletasks()

                self.log("Reading Zeiss TXM into RAM (needed for XZ/YZ + Build 3D)…", "orange")
                self.frame.update_idletasks()
                txm = open_txm_volume(path, progress_cb=_txm_progress)
                # Keep numpy array as proxy; txm.close() already ran after buffering
                self.txm_handle = txm
                self.ct_proxy = np.asarray(txm)
                self.ct_dims = tuple(int(x) for x in self.ct_proxy.shape)
                self.is_h5_fast_volume = True
                self.is_h5_lazy_volume = False
                self.current_ct_dataset_key = "TXM/ImageData"
                meta = txm.info_text()
                meta += "\nMode: full-res TXM in RAM"
                meta += "\nTip: annotate pores (S), then Build 3D to propagate"
                self._set_h5_info_text(meta)
                self._set_recon_backend_status("Backend: Zeiss TXM (olefile)", "#117A65")

            else:
                if not HAS_NIBABEL:
                    raise ImportError("nibabel not installed")
                nii = nib.load(path)
                self.ct_proxy = nii.dataobj 
                self.ct_dims = self.ct_proxy.shape
                self.is_h5_lazy_volume = True
                self._set_h5_info_text("Current file is NIfTI (.nii/.nii.gz).\nH5/TXM metadata panel is only used for .h5/.txm files.")
                self._set_recon_backend_status("Backend: nibabel + pyvista (NIfTI)", "#117A65")
            
            # 3. SMART 3D GENERATION (small preview only — never full volume)
            is_txm = low.endswith(".txm")
            step = self._compute_ct_preview_step(is_txm=is_txm)
            self.ct_preview_step = int(step)
            prev_shape = self._preview_shape_for_step(self.ct_dims, step)
            
            self.log(f"Building 3D preview (step={step}, ~{prev_shape[0]}×{prev_shape[1]}×{prev_shape[2]})…", "blue")

            # Contrast from a mid orthogonal plane (for 2D slice hints)
            mid0 = self.ct_dims[0] // 2
            mid_slice = np.asanyarray(self.ct_proxy[mid0, :, :], dtype=np.float32)
            self.v_min = float(np.min(mid_slice))
            self.v_max = float(np.max(mid_slice))

            # Load the sparse 3D preview used by PyVista
            raw_sample = np.asanyarray(self.ct_proxy[::step, ::step, ::step], dtype=np.float32)

            # Robust window from the whole preview — NOT one slice (fixes blown-out white 3D)
            p_lo, p_hi = np.percentile(raw_sample, [1.0, 99.0])
            if not np.isfinite(p_lo) or not np.isfinite(p_hi) or p_hi <= p_lo:
                p_lo, p_hi = float(np.min(raw_sample)), float(np.max(raw_sample))
            if p_hi <= p_lo:
                p_hi = p_lo + 1.0
            norm_float = np.clip((raw_sample - p_lo) / (p_hi - p_lo), 0.0, 1.0)
            # Lift bright pores (same idea as 2D slice windowing — no histogram flattening)
            norm_float = np.power(norm_float, 0.72)
            ct_u8 = (norm_float * 255.0).astype(np.uint8)
            self.ct_volume = self._enhance_3d_preview_pore_contrast(ct_u8)
            self._pv_norm_range = (float(p_lo), float(p_hi))
            
            # 4. LOAD 2D VIEW FIRST — default to XY (axis X) for TomoBank (Z,Y,X)
            # TXM is also (Z,Y,X) reconstructed stack → start on XY
            default_axis = "X" if (self.is_h5_lazy_volume or low.endswith(".txm")) else "Z"
            self.slice_axis.set(default_axis)
            if default_axis == "X":
                self.current_slice_idx = self.ct_dims[0] // 2
            elif default_axis == "Y":
                self.current_slice_idx = self.ct_dims[1] // 2
            else:
                self.current_slice_idx = self.ct_dims[2] // 2
            is_raw_projection_h5 = (
                HAS_TOMOCUPY_RUNNER
                and is_projection_h5(path)
                and low.endswith((".h5", ".hdf5"))
            )
            self.projection_cor_tune_mode = False
            if is_raw_projection_h5:
                self.tomocupy_source_h5 = path
                self._update_tomocupy_button_visibility()
                self.log(
                    "Projection HDF5: use “Open Tomocupy workbench” for COR + GPU reconstruction.",
                    "green",
                )
            self.update_slider_range()
            self.load_current_slice_ct()
            self._rebuild_ct_label_volume_preview()
            
            # For H5/TXM, skip auto-launch of PyVista to reduce load delay.
            if low.endswith(('.h5', '.hdf5', '.txm')):
                if low.endswith(".txm"):
                    self.log(
                        "TXM ready: use XY/XZ/YZ + slider. "
                        "SAM-annotate pores, Save (S), then Build 3D to propagate.",
                        "green",
                    )
                else:
                    mode_txt = "lazy full-res" if self.is_h5_lazy_volume else "preview"
                    self.log(
                        f"H5 ready ({mode_txt}): use XY/XZ/YZ + slider. "
                        "SAM-annotate pores on the current slice. Open 3D Window optional.",
                        "green",
                    )
            else:
                self.launch_pyvista_window() # Keep auto-launch for NIfTI
            
            self.log(f"Ready: {fname} | shape {self.ct_dims}", "green")
            if HAS_TOMOCUPY_RUNNER and is_projection_h5:
                proj_src = getattr(self, "tomocupy_source_h5", None)
                if is_projection_h5(path):
                    self.tomocupy_source_h5 = path
                elif proj_src and os.path.isfile(proj_src):
                    self.tomocupy_source_h5 = proj_src
                else:
                    self.tomocupy_source_h5 = None
            self._update_tomocupy_button_visibility()
            
        except Exception as e:
            self.log(f"Load Error: {e}", "red")
            self._set_recon_backend_status("Backend: load failed", "#922B21")
            print(f"Details: {e}")
            import traceback; traceback.print_exc()

    def on_axis_change(self):
        dx, dy, dz = self.ct_dims
        axis = self.slice_axis.get()
        self.current_slice_idx = dx//2 if axis=="X" else dy//2 if axis=="Y" else dz//2
        self.update_slider_range()
        self.load_current_slice_ct()

    def update_slider_range(self):
        axis = self.slice_axis.get()
        dx, dy, dz = self.ct_dims
        max_val = dx if axis == "X" else dy if axis == "Y" else dz
        self.slice_slider.config(to=max_val-1)
        self.slice_slider.set(self.current_slice_idx)

    def on_slider_change(self, val):
        if self.is_ct_mode and hasattr(self, "ct_proxy") and self.ct_proxy is not None:
            self.current_slice_idx = int(float(val))
            self.load_current_slice_ct()

    def refresh_current_ct_slice(self):
        if self.is_ct_mode and hasattr(self, 'ct_proxy') and self.ct_proxy is not None:
            cur = self.image_list[self.current_idx]
            cur_low = cur.lower()
            if cur_low.endswith(('.h5', '.hdf5', '.txm')) or os.path.isdir(cur):
                self.load_current_ct_volume()
                return
            self.load_current_slice_ct()

    def load_current_slice_ct(self):
        # Safety check
        if not hasattr(self, 'ct_proxy') or self.ct_proxy is None:
            return

        axis = self.slice_axis.get()
        idx = self.current_slice_idx
        
        # 1. Read ONLY the requested slice (lazy H5 / NIfTI / in-memory proxy)
        try:
            plane_name = {"X": "XY", "Y": "XZ", "Z": "YZ"}.get(axis, axis)
            if self.is_h5_lazy_volume or getattr(self, "is_tiff_stack_volume", False):
                self.log(f"Reading {plane_name} slice {idx}…", "blue")
                self.frame.update_idletasks()

            if axis == "Z": 
                safe_idx = min(max(0, idx), self.ct_dims[2]-1)
                slice_raw = np.asanyarray(self.ct_proxy[:, :, safe_idx])
                slice_raw = np.rot90(slice_raw) 
            elif axis == "Y": 
                safe_idx = min(max(0, idx), self.ct_dims[1]-1)
                slice_raw = np.asanyarray(self.ct_proxy[:, safe_idx, :])
                slice_raw = np.rot90(slice_raw)
            else: 
                safe_idx = min(max(0, idx), self.ct_dims[0]-1)
                slice_raw = np.asanyarray(self.ct_proxy[safe_idx, :, :])
                slice_raw = np.rot90(slice_raw)
                
            # 2. Normalize using per-slice percentiles (robust for pores)
            s_min = np.percentile(slice_raw, 1)     
            s_max = np.percentile(slice_raw, 99)
    
            if s_max - s_min > 0:
                norm_float = (slice_raw.astype(np.float32) - s_min) / (s_max - s_min)
                norm_float = np.clip(norm_float, 0.0, 1.0)
                slice_u8 = (norm_float * 255).astype(np.uint8)
            else:
                slice_u8 = np.clip(slice_raw, 0, 255).astype(np.uint8)

            self.raw_image = cv2.cvtColor(slice_u8, cv2.COLOR_GRAY2RGB)
                
            # 3. Update Canvas
            self._reset_annotation_state()
            self.load_annotations_ct()
            self.canvas.delete("all")
            self.show_image(self.raw_image)
            ax_i = {"X": 0, "Y": 1, "Z": 2}[axis]
            self.log(f"{plane_name} slice {safe_idx} / {self.ct_dims[ax_i] - 1}", "green")
            
            if self.auto_ai_var.get(): 
                # Run embedding on this specific slice
                self.run_embedding_thread()
                
        except Exception as e:
            print(f"Slice Load Error: {e}")
            self.log(f"Slice load error: {e}", "red")
            import traceback; traceback.print_exc()

    def _get_ct_slice_raw(self, axis, idx):
        """Returns raw (non-rotated) CT slice for predictor propagation."""
        if axis == "Z":
            safe_idx = min(max(0, idx), self.ct_dims[2] - 1)
            return np.asanyarray(self.ct_proxy[:, :, safe_idx]), safe_idx
        if axis == "Y":
            safe_idx = min(max(0, idx), self.ct_dims[1] - 1)
            return np.asanyarray(self.ct_proxy[:, safe_idx, :]), safe_idx
        safe_idx = min(max(0, idx), self.ct_dims[0] - 1)
        return np.asanyarray(self.ct_proxy[safe_idx, :, :]), safe_idx

    def _mask_display_to_raw(self, mask_display):
        return np.rot90(mask_display.astype(np.uint8), k=-1).astype(bool)

    def _mask_raw_to_display(self, mask_raw):
        return np.rot90(mask_raw.astype(np.uint8), k=1).astype(bool)

    def _build_display_mask_from_annotations(self):
        if self.raw_image is None or not self.current_annotations:
            return None
        h, w = self.raw_image.shape[:2]
        merged = np.zeros((h, w), dtype=np.uint8)
        for ann in self.current_annotations:
            coords = ann.get('coords', [])
            if len(coords) < 6:
                continue
            pts = []
            for i in range(0, len(coords), 2):
                x = int(np.clip(coords[i] * w, 0, w - 1))
                y = int(np.clip(coords[i + 1] * h, 0, h - 1))
                pts.append([x, y])
            if len(pts) >= 3:
                cv2.fillPoly(merged, [np.array(pts, dtype=np.int32)], 1)
        merged = self.apply_y_limit_constraint(merged)
        return merged.astype(bool)

    def _build_display_masks_by_class(self):
        if self.raw_image is None:
            return {}
        h, w = self.raw_image.shape[:2]
        masks_by_class = {}
        for ann in self.current_annotations:
            coords = ann.get('coords', [])
            class_id = int(ann.get('id', -1))
            if class_id < 0 or len(coords) < 6:
                continue
            if class_id not in masks_by_class:
                masks_by_class[class_id] = np.zeros((h, w), dtype=np.uint8)
            pts = []
            for i in range(0, len(coords), 2):
                x = int(np.clip(coords[i] * w, 0, w - 1))
                y = int(np.clip(coords[i + 1] * h, 0, h - 1))
                pts.append([x, y])
            if len(pts) >= 3:
                cv2.fillPoly(masks_by_class[class_id], [np.array(pts, dtype=np.int32)], 1)

        for class_id in list(masks_by_class.keys()):
            masks_by_class[class_id] = self.apply_y_limit_constraint(masks_by_class[class_id]).astype(bool)
        return masks_by_class

    def _slab_shape_for_bounds(self, axis, lower, upper):
        """Allocate only the [lower, upper] slab — never the full CT volume."""
        n = int(upper) - int(lower) + 1
        d0, d1, d2 = (int(x) for x in self.ct_dims)
        if axis == "Z":
            return (d0, d1, n)
        if axis == "Y":
            return (d0, n, d2)
        return (n, d1, d2)  # X

    def _put_mask_into_slab(self, vol_mask, axis, idx, lower, mask_raw):
        local = int(idx) - int(lower)
        if local < 0 or local >= vol_mask.shape[{"Z": 2, "Y": 1, "X": 0}[axis]]:
            return
        m = mask_raw.astype(np.uint8)
        if axis == "Z":
            if m.shape != vol_mask[:, :, local].shape:
                m = cv2.resize(m, (vol_mask.shape[1], vol_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
            vol_mask[:, :, local] = np.maximum(vol_mask[:, :, local], m)
        elif axis == "Y":
            if m.shape != vol_mask[:, local, :].shape:
                m = cv2.resize(m, (vol_mask.shape[2], vol_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
            vol_mask[:, local, :] = np.maximum(vol_mask[:, local, :], m)
        else:
            if m.shape != vol_mask[local, :, :].shape:
                m = cv2.resize(m, (vol_mask.shape[2], vol_mask.shape[1]), interpolation=cv2.INTER_NEAREST)
            vol_mask[local, :, :] = np.maximum(vol_mask[local, :, :], m)

    def _put_label_into_slab(self, label_volume, axis, idx, lower, mask_raw_bool, cls_id):
        local = int(idx) - int(lower)
        label_val = int(cls_id) + 1
        m = np.asarray(mask_raw_bool, dtype=bool)
        if axis == "Z":
            if m.shape != label_volume[:, :, local].shape:
                m = cv2.resize(m.astype(np.uint8), (label_volume.shape[1], label_volume.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
            target = label_volume[:, :, local]
            target[m] = label_val
        elif axis == "Y":
            if m.shape != label_volume[:, local, :].shape:
                m = cv2.resize(m.astype(np.uint8), (label_volume.shape[2], label_volume.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
            target = label_volume[:, local, :]
            target[m] = label_val
        else:
            if m.shape != label_volume[local, :, :].shape:
                m = cv2.resize(m.astype(np.uint8), (label_volume.shape[2], label_volume.shape[1]), interpolation=cv2.INTER_NEAREST).astype(bool)
            target = label_volume[local, :, :]
            target[m] = label_val

    def _normalize_slice_for_sam(self, slice_raw, max_side=1024):
        """
        Percentile-normalize + rot90 for display SAM space.
        Large CT planes are downscaled for speed (returns scale to map boxes/masks back).
        """
        s_min = np.percentile(slice_raw, 1)
        s_max = np.percentile(slice_raw, 99)
        if s_max - s_min > 0:
            norm_float = (slice_raw.astype(np.float32) - s_min) / (s_max - s_min)
            norm_float = np.clip(norm_float, 0.0, 1.0)
            slice_u8 = (norm_float * 255).astype(np.uint8)
        else:
            slice_u8 = np.clip(slice_raw, 0, 255).astype(np.uint8)
        rgb = cv2.cvtColor(np.rot90(slice_u8), cv2.COLOR_GRAY2RGB)
        h, w = rgb.shape[:2]
        scale = 1.0
        lim = int(max_side)
        if lim > 0 and max(h, w) > lim:
            scale = lim / float(max(h, w))
            rgb = cv2.resize(
                rgb,
                (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                interpolation=cv2.INTER_AREA,
            )
        return rgb, scale

    def _sam_predict_box_on_slice(self, slice_raw, bbox_display, max_side=1024):
        """Run SAM box predict; bbox/mask are in full display (rotated) coordinates."""
        rgb, scale = self._normalize_slice_for_sam(slice_raw, max_side=max_side)
        box = np.asarray(bbox_display, dtype=np.float32).copy()
        if scale != 1.0:
            box = box * scale
        self.predictor.set_image(rgb)
        masks, _, _ = self.predictor.predict(box=box, multimask_output=False)
        pred = masks[0].astype(bool)
        if scale != 1.0:
            # Restore to full display size used by annotations
            # Infer full size from unscaled rot90 of slice
            full_h = slice_raw.shape[1]  # after rot90: (cols, rows) wait
            # rot90 of (H,W) -> (W, H) display
            disp_h, disp_w = slice_raw.shape[1], slice_raw.shape[0]
            pred = cv2.resize(
                pred.astype(np.uint8),
                (disp_w, disp_h),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        return pred

    def _get_bounds_from_user(self):
        axis = self.slice_axis.get()
        max_idx = self.ct_dims[2] - 1 if axis == "Z" else self.ct_dims[1] - 1 if axis == "Y" else self.ct_dims[0] - 1
        default_lower = max(0, self.current_slice_idx - 5)
        default_upper = min(max_idx, self.current_slice_idx + 5)
        lower = simpledialog.askinteger(
            "Lower Bound",
            f"Enter lower {axis}-slice index (0 to {max_idx}):\n"
            f"(Keep span small — e.g. ±5. Full-volume allocation is disabled.)",
            initialvalue=default_lower,
            minvalue=0,
            maxvalue=max_idx,
            parent=self.frame,
        )
        if lower is None:
            return None
        upper = simpledialog.askinteger(
            "Upper Bound",
            f"Enter upper {axis}-slice index ({lower} to {max_idx}):",
            initialvalue=max(default_upper, lower),
            minvalue=lower,
            maxvalue=max_idx,
            parent=self.frame,
        )
        if upper is None:
            return None
        span = upper - lower + 1
        if span > 40:
            if not messagebox.askyesno(
                "Large span",
                f"You selected {span} slices. Propagation may take a while.\nContinue?",
                parent=self.frame,
            ):
                return None
        return (lower, upper)

    def run_3d_reconstruction_from_current_slice(self):
        if self.ct_proxy is None:
            messagebox.showwarning("3D Reconstruction", "Load a .nii / .h5 / .txm volume first.")
            return
        if self.predictor is None:
            messagebox.showwarning("3D Reconstruction", "Load a SAM model first.")
            return
        if self.current_mask is None and not self.current_annotations:
            messagebox.showwarning("3D Reconstruction", "Create one or more annotations on the current slice first.")
            return

        bounds = self._get_bounds_from_user()
        if bounds is None:
            return

        axis = self.slice_axis.get()
        source_idx = int(self.current_slice_idx)
        lower, upper = int(bounds[0]), int(bounds[1])
        if not (lower <= source_idx <= upper):
            messagebox.showwarning(
                "3D Reconstruction",
                f"Current slice ({source_idx}) must be inside bounds [{lower}, {upper}].",
            )
            return

        seed_masks = self._build_display_masks_by_class()
        if not seed_masks:
            fallback_class = self.current_class_id.get() if self.current_class_id.get() >= 0 else 0
            if self.current_mask is None:
                messagebox.showwarning("3D Reconstruction", "No seed mask / annotations found.")
                return
            seed_masks = {fallback_class: self.current_mask.astype(bool)}

        # Snapshot seeds (UI thread) — worker must not touch Tk widgets
        seed_snapshot = {int(k): np.asarray(v, dtype=bool).copy() for k, v in seed_masks.items()}
        span = upper - lower + 1
        slab_shape = self._slab_shape_for_bounds(axis, lower, upper)
        bytes_est = int(np.prod(slab_shape)) * (1 + 2)  # uint8 mask + int16 labels
        if bytes_est > 1_500_000_000:  # ~1.5 GB safety
            messagebox.showerror(
                "3D Reconstruction",
                f"Slab too large for RAM (~{bytes_est / 1e9:.1f} GB).\n"
                f"Shape {slab_shape}. Use a smaller bound range or downsampled preview mode.",
            )
            return

        progress = Toplevel(self.frame)
        progress.title("Build 3D")
        progress.geometry("420x140")
        progress.transient(self.frame)
        progress.resizable(False, False)
        try:
            progress.grab_set()
        except tk.TclError:
            pass
        tk.Label(progress, text="Propagating SAM masks…", font=("Arial", 11, "bold")).pack(pady=(16, 6))
        lbl = tk.Label(progress, text="Starting…", font=("Arial", 9), wraplength=380)
        lbl.pack(pady=4)
        progress.update_idletasks()

        result = {"ok": False, "err": None, "vol_mask": None, "label_volume": None, "n_parts": 0}

        def _worker():
            try:
                vol_mask = np.zeros(slab_shape, dtype=np.uint8)
                label_volume = np.zeros(slab_shape, dtype=np.int16)

                def _bbox_from_mask(mask_bool):
                    ys, xs = np.where(mask_bool)
                    if xs.size == 0 or ys.size == 0:
                        return None
                    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)

                n_parts = 0
                total_steps = max(1, len(seed_snapshot) * (span))
                step_i = 0

                for cls_id, seed_display_mask in seed_snapshot.items():
                    if seed_display_mask is None or not np.any(seed_display_mask):
                        continue
                    n_parts += 1
                    base_raw = self._mask_display_to_raw(seed_display_mask)
                    self._put_mask_into_slab(vol_mask, axis, source_idx, lower, base_raw)
                    self._put_label_into_slab(label_volume, axis, source_idx, lower, base_raw.astype(bool), cls_id)

                    prev_display_mask = seed_display_mask.astype(bool)
                    for idx in range(source_idx + 1, upper + 1):
                        step_i += 1
                        self.frame.after(
                            0,
                            lambda i=idx, s=step_i: lbl.config(
                                text=f"Class {cls_id}: slice {i} (up)  [{s}/{total_steps}]"
                            ),
                        )
                        bbox = _bbox_from_mask(prev_display_mask)
                        if bbox is None:
                            break
                        slice_raw, safe_idx = self._get_ct_slice_raw(axis, idx)
                        pred_disp = self._sam_predict_box_on_slice(slice_raw, bbox, max_side=1024)
                        pred_disp = self.apply_y_limit_constraint(pred_disp).astype(bool)
                        pred_raw = self._mask_display_to_raw(pred_disp)
                        self._put_mask_into_slab(vol_mask, axis, safe_idx, lower, pred_raw)
                        self._put_label_into_slab(label_volume, axis, safe_idx, lower, pred_raw.astype(bool), cls_id)
                        prev_display_mask = pred_disp

                    prev_display_mask = seed_display_mask.astype(bool)
                    for idx in range(source_idx - 1, lower - 1, -1):
                        step_i += 1
                        self.frame.after(
                            0,
                            lambda i=idx, s=step_i: lbl.config(
                                text=f"Class {cls_id}: slice {i} (down)  [{s}/{total_steps}]"
                            ),
                        )
                        bbox = _bbox_from_mask(prev_display_mask)
                        if bbox is None:
                            break
                        slice_raw, safe_idx = self._get_ct_slice_raw(axis, idx)
                        pred_disp = self._sam_predict_box_on_slice(slice_raw, bbox, max_side=1024)
                        pred_disp = self.apply_y_limit_constraint(pred_disp).astype(bool)
                        pred_raw = self._mask_display_to_raw(pred_disp)
                        self._put_mask_into_slab(vol_mask, axis, safe_idx, lower, pred_raw)
                        self._put_label_into_slab(label_volume, axis, safe_idx, lower, pred_raw.astype(bool), cls_id)
                        prev_display_mask = pred_disp

                result["ok"] = True
                result["vol_mask"] = vol_mask
                result["label_volume"] = label_volume
                result["n_parts"] = max(n_parts, len(self.current_annotations) if self.current_annotations else 1)
            except Exception as ex:
                result["err"] = ex
                import traceback
                traceback.print_exc()

            def _done():
                try:
                    progress.destroy()
                except tk.TclError:
                    pass
                if not result["ok"]:
                    err = result["err"] or "Unknown error"
                    self.log(f"Build 3D failed: {err}", "red")
                    messagebox.showerror("Build 3D failed", str(err))
                    return
                self.recon_mask_volume = result["vol_mask"]
                self.recon_label_volume = result["label_volume"]
                self.recon_axis = axis
                self.recon_bounds = (lower, upper)
                self.recon_slab_lower = lower
                self._rebuild_ct_label_volume_preview()
                self.log(
                    f"3D slab ready on {axis}-axis [{lower}, {upper}] "
                    f"({span} slices, {result['n_parts']} part(s)) — opening validation…",
                    "green",
                )
                # Rebuild SAM embedding for the current viewer slice (propagation overwrote it)
                if self.raw_image is not None and self.predictor is not None:
                    try:
                        self.predictor.set_image(self.raw_image)
                        self.is_ai_ready = True
                    except Exception:
                        self.is_ai_ready = False
                self.open_validation_panel()

            self.frame.after(0, _done)

        threading.Thread(target=_worker, daemon=True).start()

    def open_3d_validation_popup(self):
        if not HAS_PYVISTA:
            messagebox.showerror("Validation", "PyVista is required for 3D validation.")
            return
        if self.ct_volume is None:
            messagebox.showwarning("Validation", "Load a CT/.nii volume first.")
            return
        self._rebuild_ct_label_volume_preview()
        if self.ct_label_volume_preview is None and self.recon_mask_volume is None:
            messagebox.showwarning(
                "Validation",
                "No segmentations found.\n\n"
                "Save pores on slices (S), then reopen this view — or run Build 3D for propagation.",
            )
            return

        try:
            ct_grid = pv.ImageData()
            ct_grid.dimensions = self.ct_volume.shape
            ct_grid.spacing = (1, 1, 1)
            ct_grid.point_data["values"] = self.ct_volume.flatten(order="F")

            pl = pv.Plotter(shape=(1, 2), title="3D Validation: CT preview + segmented pores")
            pl.subplot(0, 0)
            pl.add_text("CT preview (isosurface)", font_size=10)
            try:
                pl.add_mesh(ct_grid.contour([self.threshold_val.get()]), color="lightgray", opacity=0.55)
            except Exception:
                pass
            pl.add_axes()

            pl.subplot(0, 1)
            pl.add_text("Segmented pores", font_size=10)
            try:
                pl.add_mesh(ct_grid.contour([self.threshold_val.get()]), color="lightgray", opacity=0.12)
            except Exception:
                pass

            label_vol = self.ct_label_volume_preview
            if label_vol is not None and np.any(label_vol):
                legend_entries = []
                for label_val in sorted(int(v) for v in np.unique(label_vol) if v > 0):
                    cls_id = label_val - 1
                    cls_u8 = (label_vol == label_val).astype(np.uint8) * 255
                    cls_grid = pv.ImageData()
                    cls_grid.dimensions = cls_u8.shape
                    cls_grid.spacing = (1, 1, 1)
                    cls_grid.origin = (0, 0, 0)
                    cls_grid.point_data["values"] = cls_u8.flatten(order="F")
                    rgb = self.class_colors.get(cls_id, (255, 0, 0))
                    color = (rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0)
                    cls_name = self.class_mapping.get(cls_id, f"class_{cls_id}")
                    try:
                        surf = cls_grid.contour([32, 224])
                        if surf.n_points == 0:
                            surf = cls_grid.threshold([1, 255], scalars="values")
                        if surf.n_points > 0:
                            pl.add_mesh(surf, color=color, opacity=0.92, name=f"recon_{cls_name}")
                            legend_entries.append([cls_name, color])
                    except Exception:
                        continue
                if legend_entries:
                    pl.add_legend(
                        legend_entries,
                        bcolor=(0.1, 0.1, 0.1),
                        face=None,
                        loc="upper right",
                        border=True,
                        size=(0.28, 0.25),
                    )
            pl.add_axes()

            pl.link_views()
            pl.show()
        except Exception as e:
            self.log(f"Validation Error: {e}", "red")
            messagebox.showerror("Validation Error", str(e))

    def open_validation_panel(self):
        """Validation hub: step frames, rename objects, export sequence, open 3D view."""
        popup = Toplevel(self.frame)
        popup.title("Validation")
        popup.geometry("460x620")
        popup.transient(self.frame)
        self._validation_popup = popup

        tk.Label(
            popup,
            text="Validation — rename objects, review frames, export",
            font=("Arial", 11, "bold"),
            wraplength=420,
        ).pack(pady=(10, 6), padx=12)

        f_nav = tk.LabelFrame(popup, text="Frame navigation", padx=8, pady=6)
        f_nav.pack(fill="x", padx=12, pady=4)
        lbl_frame = tk.Label(f_nav, text="", font=("Arial", 9), anchor="w")
        lbl_frame.pack(fill="x")

        def _update_frame_label():
            if self.is_ct_mode and self.ct_volume is not None:
                nsl = max(0, self._ct_slice_count() - 1)
                lbl_frame.config(
                    text=f"CT slice: axis {self.slice_axis.get()}, index {self.current_slice_idx} / {nsl}"
                )
            elif self.image_list:
                p = self.image_list[self.current_idx]
                lbl_frame.config(text=f"Frame {self.current_idx + 1}/{len(self.image_list)}: {os.path.basename(p)}")
            else:
                lbl_frame.config(text="No frames loaded")

        def _step(delta):
            if self.is_ct_mode and self.ct_volume is not None:
                n = self._ct_slice_count()
                self.current_slice_idx = int(np.clip(self.current_slice_idx + delta, 0, max(0, n - 1)))
                if hasattr(self, "slice_slider"):
                    self.slice_slider.set(self.current_slice_idx)
                self.load_current_slice_ct()
            elif self.image_list:
                self.current_idx = int(np.clip(self.current_idx + delta, 0, len(self.image_list) - 1))
                self.load_current_image_2d()
            _update_frame_label()

        row_nav = tk.Frame(f_nav)
        row_nav.pack(fill="x", pady=4)
        tk.Button(row_nav, text="◀ Prev", command=lambda: _step(-1), width=8).pack(side="left", padx=2)
        tk.Button(row_nav, text="Next ▶", command=lambda: _step(1), width=8).pack(side="left", padx=2)
        _update_frame_label()

        f_cls = tk.LabelFrame(popup, text="Objects (rename)", padx=8, pady=6)
        f_cls.pack(fill="both", expand=True, padx=12, pady=4)
        cls_canvas = tk.Canvas(f_cls, highlightthickness=0)
        cls_sb = tk.Scrollbar(f_cls, orient="vertical", command=cls_canvas.yview)
        cls_inner = tk.Frame(cls_canvas)
        cls_inner.bind("<Configure>", lambda e: cls_canvas.configure(scrollregion=cls_canvas.bbox("all")))
        cls_canvas.create_window((0, 0), window=cls_inner, anchor="nw")
        cls_canvas.configure(yscrollcommand=cls_sb.set)
        cls_canvas.pack(side="left", fill="both", expand=True)
        cls_sb.pack(side="right", fill="y")

        entry_vars = {}

        def _rebuild_class_rows():
            for w in cls_inner.winfo_children():
                w.destroy()
            entry_vars.clear()
            if not self.class_mapping:
                tk.Label(cls_inner, text="No classes yet — add objects in the main UI.", fg="gray").pack(anchor="w")
                return
            for cid in sorted(self.class_mapping.keys()):
                row = tk.Frame(cls_inner, pady=2)
                row.pack(fill="x")
                hex_col = "#%02x%02x%02x" % self.class_colors.get(cid, (128, 128, 128))
                tk.Label(row, bg=hex_col, width=2).pack(side="left", padx=(0, 6))
                tk.Label(row, text=f"[{cid}]", width=4, anchor="w").pack(side="left")
                var = tk.StringVar(value=self.class_mapping[cid])
                entry_vars[cid] = var
                tk.Entry(row, textvariable=var, width=22).pack(side="left", fill="x", expand=True, padx=4)

                def _apply_rename(c=cid, v=var):
                    if self.rename_class_by_id(c, v.get()):
                        _rebuild_class_rows()
                        self.refresh_class_ui()

                tk.Button(row, text="Rename", command=_apply_rename, font=("Arial", 8)).pack(side="right")

        _rebuild_class_rows()

        f_act = tk.Frame(popup, padx=12, pady=8)
        f_act.pack(fill="x")

        can_3d = HAS_PYVISTA and self.ct_volume is not None

        if can_3d:
            tk.Button(
                f_act,
                text="Open 3D validation view",
                command=lambda: (popup.destroy(), self.open_3d_validation_popup()),
                bg="#2E86C1",
                fg="white",
                font=("Arial", 10, "bold"),
            ).pack(fill="x", pady=3)
            tk.Label(
                f_act,
                text="Shows saved slice labels + Build 3D propagation (save with S first).",
                fg="gray",
                font=("Arial", 8),
                wraplength=400,
            ).pack(fill="x", pady=(0, 2))
        elif self.is_ct_mode:
            tk.Label(
                f_act,
                text="Run Build 3D first to enable side-by-side 3D validation.",
                fg="gray",
                font=("Arial", 9),
                wraplength=400,
            ).pack(fill="x", pady=2)

        tk.Button(
            f_act,
            text="💾 Export video / image stack…",
            command=self.open_export_sequence_dialog,
            bg="#16A085",
            fg="white",
            font=("Arial", 10, "bold"),
        ).pack(fill="x", pady=3)
        tk.Button(f_act, text="Close", command=popup.destroy).pack(fill="x", pady=(6, 0))

        popup.protocol("WM_DELETE_WINDOW", popup.destroy)

    def _axis_index(self):
        return {"X": 0, "Y": 1, "Z": 2}.get(self.slice_axis.get().upper(), 2)

    def _ct_slice_count(self):
        if not hasattr(self, "ct_dims") or self.ct_dims is None:
            return 0
        ax = self.slice_axis.get().upper()
        if ax == "X":
            return int(self.ct_dims[0])
        if ax == "Y":
            return int(self.ct_dims[1])
        return int(self.ct_dims[2])

    def open_export_sequence_dialog(self):
        dlg = Toplevel(self.frame)
        dlg.title("Export annotation sequence")
        dlg.geometry("400x280")
        dlg.transient(self.frame)

        out_var = tk.StringVar()
        fps_var = tk.DoubleVar(value=10.0)
        video_var = tk.BooleanVar(value=True)
        stack_var = tk.BooleanVar(value=True)
        masks_var = tk.BooleanVar(value=False)

        tk.Label(dlg, text="Output folder", font=("Arial", 10, "bold")).pack(anchor="w", padx=12, pady=(12, 4))
        row = tk.Frame(dlg)
        row.pack(fill="x", padx=12)
        tk.Entry(row, textvariable=out_var, width=36).pack(side="left", fill="x", expand=True)

        def _browse():
            d = filedialog.askdirectory(title="Select export folder")
            if d:
                out_var.set(d)

        tk.Button(row, text="Browse…", command=_browse).pack(side="left", padx=4)

        opts = tk.Frame(dlg, padx=12, pady=8)
        opts.pack(fill="x")
        tk.Checkbutton(opts, text="Write MP4 video (annotated overlays)", variable=video_var).pack(anchor="w")
        tk.Checkbutton(opts, text="Write PNG image stack (annotated overlays)", variable=stack_var).pack(anchor="w")
        tk.Checkbutton(opts, text="Also save binary masks per frame", variable=masks_var).pack(anchor="w")

        tk.Label(opts, text="Video FPS:").pack(anchor="w", pady=(8, 0))
        tk.Spinbox(opts, from_=1, to=60, increment=1, textvariable=fps_var, width=6).pack(anchor="w")

        def _run():
            folder = out_var.get().strip()
            if not folder:
                messagebox.showwarning("Export", "Choose an output folder.", parent=dlg)
                return
            if not video_var.get() and not stack_var.get():
                messagebox.showwarning("Export", "Enable video and/or image stack.", parent=dlg)
                return
            dlg.destroy()
            self.export_annotation_sequence(
                folder,
                fps=float(fps_var.get()),
                write_video=video_var.get(),
                write_stack=stack_var.get(),
                write_masks=masks_var.get(),
            )

        tk.Button(dlg, text="Export", command=_run, bg="#16A085", fg="white", font=("Arial", 10, "bold")).pack(
            fill="x", padx=12, pady=12
        )

    def export_annotation_sequence(self, out_dir, fps=10.0, write_video=True, write_stack=True, write_masks=False):
        try:
            os.makedirs(out_dir, exist_ok=True)
            stack_dir = os.path.join(out_dir, "annotated_frames")
            mask_dir = os.path.join(out_dir, "masks")
            if write_stack:
                os.makedirs(stack_dir, exist_ok=True)
            if write_masks:
                os.makedirs(mask_dir, exist_ok=True)

            frames_bgr = []
            n_total = 0

            if self.is_ct_mode and self.ct_volume is not None:
                saved_si = self.current_slice_idx
                n_slices = self._ct_slice_count()
                for si in range(n_slices):
                    self.current_slice_idx = si
                    if hasattr(self, "slice_slider"):
                        self.slice_slider.set(si)
                    self.load_current_slice_ct()
                    if self.raw_image is None:
                        continue
                    h, w = self.raw_image.shape[:2]
                    m = self._annotations_to_mask(self.current_annotations, h, w) if self.current_annotations else None
                    comp = self._compose_overlay_for_annotations(
                        self.raw_image, self.current_annotations, m
                    )
                    bgr = cv2.cvtColor(comp, cv2.COLOR_RGB2BGR)
                    frames_bgr.append(bgr)
                    if write_stack:
                        imwrite_safe(os.path.join(stack_dir, f"slice_{si:04d}.png"), bgr)
                    if write_masks and m is not None:
                        imwrite_safe(os.path.join(mask_dir, f"slice_{si:04d}_mask.png"), m)
                    self.log(f"Export CT slice {si + 1}/{n_slices}", "blue")
                    self.frame.update_idletasks()
                self.current_slice_idx = saved_si
                if hasattr(self, "slice_slider"):
                    self.slice_slider.set(saved_si)
                self.load_current_slice_ct()
                n_total = len(frames_bgr)
            elif self.image_list:
                saved_idx = self.current_idx
                for i, path in enumerate(self.image_list):
                    img = imread_safe(path)
                    if img is None:
                        continue
                    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    ann = self._load_annotations_list_2d(path)
                    h, w = rgb.shape[:2]
                    m = self._annotations_to_mask(ann, h, w) if ann else None
                    comp = self._compose_overlay_for_annotations(rgb, ann, m)
                    stem = os.path.splitext(os.path.basename(path))[0]
                    bgr = cv2.cvtColor(comp, cv2.COLOR_RGB2BGR)
                    frames_bgr.append(bgr)
                    if write_stack:
                        imwrite_safe(os.path.join(stack_dir, f"{stem}.png"), bgr)
                    if write_masks and m is not None:
                        imwrite_safe(os.path.join(mask_dir, f"{stem}_mask.png"), m)
                    self.log(f"Export frame {i + 1}/{len(self.image_list)}", "blue")
                    self.frame.update_idletasks()
                self.current_idx = saved_idx
                if self.image_list and not self.is_ct_mode:
                    self.load_current_image_2d()
                n_total = len(frames_bgr)
            else:
                messagebox.showwarning("Export", "Load a 2D image sequence or CT volume first.")
                return

            if not frames_bgr:
                messagebox.showwarning("Export", "No frames to export.")
                return

            if write_video:
                h, w = frames_bgr[0].shape[:2]
                mp4_path = os.path.join(out_dir, "sam_annotated_sequence.mp4")
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                vw = cv2.VideoWriter(mp4_path, fourcc, max(1.0, float(fps)), (w, h))
                if not vw.isOpened():
                    avi_path = os.path.join(out_dir, "sam_annotated_sequence.avi")
                    fourcc = cv2.VideoWriter_fourcc(*"XVID")
                    vw = cv2.VideoWriter(avi_path, fourcc, max(1.0, float(fps)), (w, h))
                    mp4_path = avi_path
                for fr in frames_bgr:
                    if fr.shape[0] != h or fr.shape[1] != w:
                        fr = cv2.resize(fr, (w, h))
                    vw.write(fr)
                vw.release()
                self.log(f"Video saved: {mp4_path}", "green")

            msg = f"Exported {n_total} frame(s) to:\n{out_dir}"
            if write_stack:
                msg += f"\n• Stack: {stack_dir}"
            messagebox.showinfo("Export complete", msg)
        except Exception as e:
            self.log(f"Export sequence error: {e}", "red")
            messagebox.showerror("Export failed", str(e))

    def load_current_image_2d(self):
        path = self.image_list[self.current_idx]
        self.file_listbox.selection_clear(0, tk.END)
        self.file_listbox.selection_set(self.current_idx)
        self.file_listbox.see(self.current_idx)
        
        img = imread_safe(path)
        if img is None: return
        self.raw_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self._reset_annotation_state()
        self.load_annotations_2d(path)
        self.canvas.delete("all"); self.show_image(self.raw_image)
        if self.auto_ai_var.get(): self.run_embedding_thread()

    def _reset_annotation_state(self):
        self.current_mask = None; self.input_points = []; self.input_labels = []
        self.poly_points = []; self.is_ai_ready = False; self.current_annotations = []
        self.undo_stack = []

    def add_from_entry(self):
        name = self.new_class_entry_var.get().strip()
        if not name:
            messagebox.showwarning("New Class", "Type a class name, then press Enter or +.")
            return
        self.add_class_by_name(name)

    def add_class_by_name(self, name):
        name = (name or "").strip()
        if not name:
            return
        for i, n in self.class_mapping.items():
            if n.lower() == name.lower():
                self.current_class_id.set(i)
                self.new_class_entry_var.set("")
                self.refresh_class_ui()
                self.log(f"Selected class '{name}'", "blue")
                return
        nid = (max(self.class_mapping.keys()) + 1) if self.class_mapping else 0
        self.class_mapping[nid] = name
        self.class_colors[nid] = self.PALETTE[nid % len(self.PALETTE)]
        self.current_class_id.set(nid)
        self.new_class_entry_var.set("")
        save_dir = self._classes_save_dir()
        if save_dir:
            self.save_classes_txt(save_dir)
        self.refresh_class_ui()
        self.log(f"Added class '{name}' (id={nid})", "green")

    def refresh_class_ui(self):
        cid = self.current_class_id.get()
        
        if hasattr(self, 'lbl_active_class') and self.lbl_active_class.winfo_exists():
            if cid in self.class_mapping:
                name = self.class_mapping[cid]
                col = self.class_colors.get(cid, (0,0,0))
                hex_col = '#%02x%02x%02x' % col
                self.lbl_active_class.config(text=name, fg=hex_col)
            else:
                self.lbl_active_class.config(text="None", fg="gray")


        if hasattr(self, 'legend_frame') and self.legend_frame.winfo_exists():
            for w in self.legend_frame.winfo_children(): w.destroy()

            if not self.class_mapping:
                tk.Label(
                    self.legend_frame,
                    text="No classes yet.\nType a name in toolbar → Enter / +\nor click a preset button.",
                    bg="white" if self.is_ct_mode else "#ECF0F1",
                    fg="#7F8C8D",
                    font=("Arial", 9),
                    justify="left",
                ).pack(anchor="w", padx=6, pady=6)
            
            for i, n in self.class_mapping.items():
                # Determine colors
                is_active = (i == cid)
                hex_col = '#%02x%02x%02x' % self.class_colors.get(i, (0,0,0))
                bg_color = "#D5DBDB" if is_active else "#ECF0F1"
                
                # Row Container (Clickable)
                row = tk.Frame(self.legend_frame, bg=bg_color, pady=2, padx=2)
                row.pack(fill="x", pady=1)
                
                # Color Box
                lbl_col = tk.Label(row, bg=hex_col, width=2)
                lbl_col.pack(side="left", padx=5)
                
                # Name Label
                font_style = ("Arial", 9, "bold") if is_active else ("Arial", 9)
                lbl_name = tk.Label(row, text=n, bg=bg_color, font=font_style, cursor="hand2")
                lbl_name.pack(side="left", fill="x", expand=True, anchor="w")

                def _rename_from_legend(idx=i):
                    new = simpledialog.askstring(
                        "Rename object",
                        f"New name for class {idx} ({self.class_mapping.get(idx, '')}):",
                        initialvalue=self.class_mapping.get(idx, ""),
                        parent=self.frame,
                    )
                    if new and self.rename_class_by_id(idx, new):
                        self.refresh_class_ui()

                tk.Button(
                    row,
                    text="✎",
                    command=_rename_from_legend,
                    bg=bg_color,
                    fg="#2C3E50",
                    relief="flat",
                    font=("Arial", 8),
                    cursor="hand2",
                ).pack(side="right", padx=2)
                
                if is_active:
                    tk.Label(row, text="◄", bg=bg_color, fg="#2E86C1", font=("Arial", 8)).pack(side="right", padx=2)

                # Bind Clicks
                def set_class(e, idx=i):
                    self.current_class_id.set(idx)
                    self.refresh_class_ui()
                
                row.bind("<Button-1>", set_class)
                lbl_col.bind("<Button-1>", set_class)
                lbl_name.bind("<Button-1>", set_class)
    # ==========================================================================
    # ANNOTATIONS & SAM
    # ==========================================================================
    def load_annotations_ct(self):
        vn = os.path.splitext(os.path.basename(self.image_list[self.current_idx]))[0]
        p = os.path.join(os.path.dirname(self.image_list[self.current_idx]), "labels_ct_slices", f"{vn}_{self.slice_axis.get()}_{self.current_slice_idx:04d}.txt")
        if os.path.exists(p):
            with open(p) as f:
                for l in f: self.current_annotations.append({'id':int(l.split()[0]), 'coords':[float(x) for x in l.split()[1:]]})

    def load_annotations_2d(self, p):
        bn = os.path.splitext(os.path.basename(p))[0]
        tp = os.path.join(os.path.dirname(p), "labels_txt", bn+".txt")
        if os.path.exists(tp):
            with open(tp) as f:
                for l in f: self.current_annotations.append({'id':int(l.split()[0]), 'coords':[float(x) for x in l.split()[1:]]})

    def save_all_annotations_to_file(self, silent=False):
        # 1. SETUP PATHS
        if self.is_ct_mode:
            vn = os.path.splitext(os.path.basename(self.image_list[self.current_idx]))[0]
            bd = os.path.dirname(self.image_list[self.current_idx])
            
            # Define Text and Image folders
            lbl_dir = os.path.join(bd, "labels_ct_slices")
            img_dir = os.path.join(bd, "images_ct_slices")
            os.makedirs(lbl_dir, exist_ok=True)
            os.makedirs(img_dir, exist_ok=True)
            
            # Define File Paths ({VolumeName}_{Axis}_{Index})
            p_txt = os.path.join(lbl_dir, f"{vn}_{self.slice_axis.get()}_{self.current_slice_idx:04d}.txt")
            p_img = os.path.join(img_dir, f"{vn}_{self.slice_axis.get()}_{self.current_slice_idx:04d}.png")
        else:
            # 2D Mode Paths
            p_src = self.image_list[self.current_idx]
            d = os.path.dirname(p_src)
            os.makedirs(os.path.join(d, "labels_txt"), exist_ok=True)
            os.makedirs(os.path.join(d, "labels_mask"), exist_ok=True)
            
            p_txt = os.path.join(d, "labels_txt", os.path.splitext(os.path.basename(p_src))[0]+".txt")
            p_mask = os.path.join(d, "labels_mask", os.path.splitext(os.path.basename(p_src))[0]+"_mask.png")

        # 2. LOGIC: DELETE OR SAVE
        if not self.current_annotations:
            # --- DELETE MODE ---
            if os.path.exists(p_txt): os.remove(p_txt)
            
            if self.is_ct_mode:
                if os.path.exists(p_img): os.remove(p_img)
            else:
                if os.path.exists(p_mask): os.remove(p_mask)
                # Update Listbox UI for 2D (Uncheck)
                fname = os.path.basename(self.image_list[self.current_idx])
                self.file_listbox.delete(self.current_idx)
                self.file_listbox.insert(self.current_idx, f"[  ] {fname}")
                self.file_listbox.itemconfig(self.current_idx, {'fg': 'black'})
                self.file_listbox.selection_set(self.current_idx)
                
            if not silent: self.log("Data Removed (Empty)", "red")
            return

        # --- SAVE MODE ---
        # 3. Save Text Label
        with open(p_txt, "w", encoding="utf-8") as f:
            for a in self.current_annotations:
                f.write(f"{a['id']} " + " ".join([f"{c:.6f}" for c in a['coords']]) + "\n")

        # 4. Save Image / Mask
        if self.is_ct_mode:
            # CT: Save the actual image slice (BGR for OpenCV)
            bgr_img = cv2.cvtColor(self.raw_image, cv2.COLOR_RGB2BGR)
            imwrite_safe(p_img, bgr_img)
            save_dir = self._classes_save_dir()
            if save_dir and self.class_mapping:
                self.save_classes_txt(save_dir)
        else:
            # 2D: Save the binary mask
            h, w = self.raw_image.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            for a in self.current_annotations:
                pts = np.array([[int(c*w) if i%2==0 else int(c*h) for i,c in enumerate(a['coords'])]], np.int32).reshape((-1,1,2))
                cv2.fillPoly(mask, [pts], 255)
            imwrite_safe(p_mask, mask)
            
            # Update Listbox UI for 2D (Check)
            fname = os.path.basename(self.image_list[self.current_idx])
            self.file_listbox.delete(self.current_idx)
            self.file_listbox.insert(self.current_idx, f"[✓] {fname}")
            self.file_listbox.itemconfig(self.current_idx, {'fg': 'green'})
            self.file_listbox.selection_set(self.current_idx)

        if not silent:
            self.log(f"Saved → {os.path.basename(p_txt)}", "green")
            if self.is_ct_mode:
                self.log(f"CT labels: {p_txt}", "green")

    def manage_active_class(self):
        cid = self.current_class_id.get()
        if cid == -1 or cid not in self.class_mapping:
            messagebox.showwarning("Manage Class", "No class selected.")
            return

        current_name = self.class_mapping[cid]
        
        # Create a Custom Dialog
        dialog = Toplevel(self.frame)
        dialog.title("Manage Label")
        dialog.geometry("300x150")
        
        tk.Label(dialog, text=f"Selected: {current_name}", font=("Arial", 10, "bold")).pack(pady=10)
        
        def do_rename():
            new_name = simpledialog.askstring("Rename", f"Rename '{current_name}' to:", parent=dialog)
            if new_name and new_name.strip():
                self.class_mapping[cid] = new_name.strip()
                self._save_and_refresh()
                dialog.destroy()

        def do_delete():
            if messagebox.askyesno("Delete", f"Delete class '{current_name}'?\n(Existing annotations will lose their label)", parent=dialog):
                del self.class_mapping[cid]
                # Reset selection if available
                if self.class_mapping:
                    self.current_class_id.set(next(iter(self.class_mapping)))
                else:
                    self.current_class_id.set(-1)
                self._save_and_refresh()
                dialog.destroy()

        btn_frame = tk.Frame(dialog)
        btn_frame.pack(fill="x", pady=10)
        
        tk.Button(btn_frame, text="Rename", command=do_rename, bg="#3498DB", fg="white", width=10).pack(side="left", padx=20)
        tk.Button(btn_frame, text="Delete", command=do_delete, bg="#E74C3C", fg="white", width=10).pack(side="right", padx=20)

    def _save_and_refresh(self):
        # Save changes to disk
        if self.image_list:
            folder = os.path.dirname(self.image_list[0])
            self.save_classes_txt(folder)
        # Update UI
        self.refresh_class_ui()


    def load_model_thread(self):
        """Opens a popup to select from available SAM models."""
        self.select_model_popup()

    def select_model_popup(self):
        # 1. Setup Popup Window
        popup = tk.Toplevel(self.frame)
        popup.title("Select SAM Model")
        popup.geometry("500x400")
        popup.configure(bg="#ECF0F1")
        
        tk.Label(popup, text="Select a Model Checkpoint", font=("Arial", 12, "bold"), bg="#ECF0F1", fg="#2C3E50").pack(pady=10)

        # 2. List Container
        list_frame = tk.Frame(popup, bg="white", bd=2, relief="sunken")
        list_frame.pack(fill="both", expand=True, padx=15, pady=5)
        
        sb = tk.Scrollbar(list_frame)
        sb.pack(side="right", fill="y")
        
        lb = tk.Listbox(list_frame, font=("Consolas", 10), yscrollcommand=sb.set, selectmode="browse", height=10)
        lb.pack(side="left", fill="both", expand=True)
        sb.config(command=lb.yview)


        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        root_folder = os.path.dirname(script_dir)

        model_folder = os.path.join(root_folder, "models")
        
        print(f"Searching for models in: {model_folder}")

        search_paths = [
            model_folder,                  
            os.path.join(os.getcwd(), "models"), 
            os.getcwd()                     
        ]
        # ======================================================================

        candidates = []
        seen_files = set()

        for d in search_paths:
            if os.path.exists(d):
                try:
                    for f in os.listdir(d):
                        # Filter for .pth or .pt files
                        if f.lower().endswith((".pth", ".pt")):
                            full_p = os.path.join(d, f)
                            if f not in seen_files:
                                candidates.append(full_p)
                                seen_files.add(f)
                except Exception as e:
                    print(f"Error reading {d}: {e}")
        
        # Sort and Insert into Listbox
        candidates.sort(key=lambda x: os.path.basename(x))
        for c in candidates:
            lb.insert(tk.END, os.path.basename(c))
            
            # Color coding for easy ID
            bname = os.path.basename(c).lower()
            if "sam2" in bname:
                lb.itemconfig(tk.END, {'fg': 'blue'})
            elif "vit" in bname:
                lb.itemconfig(tk.END, {'fg': 'darkgreen'})
            elif "yolo" in bname:
                lb.itemconfig(tk.END, {'fg': 'purple'})

        # 4. Action Logic
        def on_confirm():
            sel = lb.curselection()
            if not sel: return
            fname = lb.get(sel[0])
            
            # Find the full path that matches the selected filename
            full_path = next((p for p in candidates if os.path.basename(p) == fname), None)
            
            if full_path:
                self.checkpoint = full_path
                popup.destroy()
                threading.Thread(target=self._init_sam, daemon=True).start()

        def on_browse_manual():
            # Start browsing from the detected model folder
            start_dir = model_folder if os.path.exists(model_folder) else os.getcwd()
            
            path = filedialog.askopenfilename(
                title="Browse for Model",
                initialdir=start_dir,
                filetypes=[("Model Files", "*.pth *.pt"), ("All Files", "*.*")]
            )
            if path:
                self.checkpoint = path
                popup.destroy()
                threading.Thread(target=self._init_sam, daemon=True).start()

        # 5. Buttons
        btn_frame = tk.Frame(popup, bg="#ECF0F1")
        btn_frame.pack(fill="x", pady=15, padx=15)
        
        tk.Button(btn_frame, text="📂 Browse Folder...", command=on_browse_manual, bg="white").pack(side="left")
        tk.Button(btn_frame, text="✅ LOAD MODEL", command=on_confirm, bg="#27AE60", fg="white", font=("Arial", 10, "bold"), width=15).pack(side="right")
    def _init_sam(self):
        try:
            ckpt_name = os.path.basename(self.checkpoint).lower()
            self.log(f"Loading {ckpt_name}...", "orange")

            if "sam2" in ckpt_name:
                from sam2.build_sam import build_sam2
                from sam2.sam2_image_predictor import SAM2ImagePredictor
                from hydra.core.global_hydra import GlobalHydra
                from hydra import initialize_config_dir
                # import os
                
                # 1. Get the EXACT folder where you selected your model
                model_dir = os.path.abspath(os.path.dirname(self.checkpoint))
                
                # 2. Clear any broken Hydra memory
                if GlobalHydra.instance().is_initialized():
                    GlobalHydra.instance().clear()

                # 3. Match the file names exactly to your screenshot
                if "tiny" in ckpt_name or " t " in ckpt_name or "hiera_t" in ckpt_name: 
                    cfg = "sam2.1_hiera_t.yaml"
                elif "small" in ckpt_name or " s " in ckpt_name or "hiera_s" in ckpt_name: 
                    cfg = "sam2.1_hiera_s.yaml"
                elif "large" in ckpt_name or " l " in ckpt_name or "hiera_l" in ckpt_name: 
                    cfg = "sam2.1_hiera_l.yaml"
                else: 
                    cfg = "sam2.1_hiera_b+.yaml"

                if self.device == "cuda" and torch.cuda.is_bf16_supported():
                    print("✅ Using bfloat16 precision")

                # 4. Force Hydra to look ONLY in your local models folder
                with initialize_config_dir(config_dir=model_dir, version_base="1.2"):
                    sam2_model = build_sam2(cfg, self.checkpoint, device=self.device)
                    
                self.predictor = SAM2ImagePredictor(sam2_model)
                self.is_sam2 = True 

            else:
                from segment_anything import sam_model_registry, SamPredictor
                if "vit_h" in ckpt_name: model_type = "vit_h"
                elif "vit_l" in ckpt_name: model_type = "vit_l"
                else: model_type = "vit_b" 
                
                sam = sam_model_registry[model_type](checkpoint=self.checkpoint)
                sam.to(device=self.device)
                self.predictor = SamPredictor(sam)

            self.log(f"Model Ready", "green")

        except Exception as e:
            self.log("Model Load Fail", "red")
            print(f"Detailed Error: {e}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("Model Error", f"Failed to load model:\n{e}")
    def run_embedding_thread(self):
        if self.predictor and self.raw_image is not None: threading.Thread(target=self.set_embedding, daemon=True).start()
    def set_embedding(self):
        self.log("Embedding...", "orange"); self.predictor.set_image(self.raw_image); self.is_ai_ready=True; self.log("Ready", "green")
    
    # Interactions
    def on_left_click(self, e):
        # 1. Coordinate Scaling
        if self.scale == 0 or self.raw_image is None: return
        
        if self.setting_x_right_limit_mode:
            rx = int((e.x - self.off_x) / self.scale)
            h, w = self.raw_image.shape[:2]
            self.x_right_limit = max(0, min(rx, w))
            self.setting_x_right_limit_mode = False
            self.btn_x_right_limit.config(
                text=f"Right: X≥{self.x_right_limit} off (clear)", bg="#2ECC71"
            )
            self.canvas.config(cursor="cross")
            self.canvas.delete("crosshair")
            if self.current_mask is not None:
                self.current_mask = self.apply_y_limit_constraint(self.current_mask)
            self.show_image(self.raw_image, self.current_mask)
            self.log(f"Right limit set: no segmentation at x ≥ {self.x_right_limit}", "green")
            return

        if self.setting_limit_mode:
            ry = int((e.y - self.off_y) / self.scale)
            h, w = self.raw_image.shape[:2]
            self.y_limit = max(0, min(ry, h))
            self.setting_limit_mode = False
            self.btn_limit.config(text=f"Top: Y<{self.y_limit} off (clear)", bg="#2ECC71")
            self.canvas.config(cursor="cross")
            self.canvas.delete("crosshair")
            if self.current_mask is not None:
                self.current_mask = self.apply_y_limit_constraint(self.current_mask)
            self.show_image(self.raw_image, self.current_mask)
            self.log(f"Top limit set: no segmentation above row {self.y_limit}", "green")
            return

        rx = int((e.x - self.off_x) / self.scale)
        ry = int((e.y - self.off_y) / self.scale)
        
        h, w = self.raw_image.shape[:2]
        if rx < 0 or rx >= w or ry < 0 or ry >= h: return
        
        # --- BOX TOOL ---
        if self.mode.get() == "box":
            self.drawing_box = True
            self.box_start = (rx, ry)
            self.box_end = (rx, ry)

        # AI Mode Logic (single path — avoid double-adding the click)
        elif self.mode.get() == "ai":
            if not self.is_ai_ready: 
                messagebox.showwarning("AI Not Ready", "Please click '⚡ Run AI' first!")
                return
                
            self.save_state_for_undo()
            
            # Add Positive Point
            self.input_points.append([rx, ry])
            self.input_labels.append(1)
            
            # STEP A: Force Immediate Draw
            self.show_image(self.raw_image, self.current_mask)
            self.canvas.update_idletasks() 
            
            # STEP B: Run AI
            self.run_sam_prediction()

        # 3. Polygon Mode
        elif self.mode.get() == "polygon":
            self.save_state_for_undo()
            self.poly_points.append((rx, ry))
            self.show_image(self.raw_image, self.current_mask)
            
        # 4. Edit Mode
        elif self.mode.get() == "edit":
            # Check click on existing annotation
            clicked_idx = -1
            for idx, ann in reversed(list(enumerate(self.current_annotations))):
                coords = ann['coords']
                pts = []
                for i in range(0, len(coords), 2):
                    pts.append([coords[i] * w, coords[i+1] * h])
                pts = np.array(pts, np.int32)
                if cv2.pointPolygonTest(pts, (rx, ry), False) >= 0:
                    clicked_idx = idx
                    break
            self.selected_annotation_index = clicked_idx
            self.show_image(self.raw_image, self.current_mask)
            
    def on_left_drag(self, e):
        if self.mode.get() == "box" and self.drawing_box:
            rx = int((e.x - self.off_x) / self.scale)
            ry = int((e.y - self.off_y) / self.scale)
            self.box_end = (rx, ry)
            self.show_image(self.raw_image, self.current_mask)
            
    def on_left_release(self, e):
        if self.mode.get() == "box" and self.drawing_box:
            self.drawing_box = False
            rx = int((e.x - self.off_x) / self.scale)
            ry = int((e.y - self.off_y) / self.scale)
            
            x1, x2 = min(self.box_start[0], rx), max(self.box_start[0], rx)
            y1, y2 = min(self.box_start[1], ry), max(self.box_start[1], ry)
            
            if (x2 - x1) > 5 and (y2 - y1) > 5: 
                self.save_state_for_undo()
                self.input_box = [x1, y1, x2, y2]
                self.run_sam_prediction()
            self.show_image(self.raw_image, self.current_mask)
    
    def _on_small_bubble_mode_toggle(self):
        on = self.small_bubble_mode.get()
        tip = (
            "Small-bubble mode ON: SAM picks the smallest valid mask; "
            "use Point/Box on each bubble (right-click = negative)."
            if on else
            "Small-bubble mode OFF."
        )
        self.log(tip, "green" if on else "gray")
        if self.current_mask is not None and self.is_ai_ready:
            self.run_sam_prediction()

    def _mask_iou(self, a, b):
        inter = float(np.logical_and(a, b).sum())
        union = float(np.logical_or(a, b).sum())
        return inter / union if union > 0 else 0.0

    def _refine_small_bubble_mask(self, mask):
        """Light cleanup to reduce horizontal bleed while keeping small pores."""
        if mask is None:
            return None
        m = np.asarray(mask, dtype=np.uint8) * 255
        # Elliptical opening breaks thin horizontal streaks from x-ray banding
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)
        # Keep only components that touch at least one positive prompt (if any)
        if self.input_points and self.input_labels:
            num, labels = cv2.connectedComponents(m)
            keep = np.zeros_like(m)
            for i, (lab, (px, py)) in enumerate(zip(self.input_labels, self.input_points)):
                if int(lab) != 1:
                    continue
                x, y = int(round(px)), int(round(py))
                if 0 <= y < labels.shape[0] and 0 <= x < labels.shape[1]:
                    cid = labels[y, x]
                    if cid > 0:
                        keep[labels == cid] = 255
            if keep.any():
                m = keep
        return m.astype(bool)

    def _pick_small_feature_mask(self, masks, scores):
        """From multimask outputs, prefer compact bubble-like masks."""
        bool_masks = [np.asarray(m, dtype=bool) for m in masks]
        score_list = [float(s) for s in scores]

        if self.input_box is not None:
            bx1, by1, bx2, by2 = [int(round(v)) for v in self.input_box]
            h, w = bool_masks[0].shape
            bx1, bx2 = max(0, bx1), min(w - 1, bx2)
            by1, by2 = max(0, by1), min(h - 1, by2)
            hint = np.zeros((h, w), dtype=bool)
            hint[by1:by2 + 1, bx1:bx2 + 1] = True
            best = max(range(len(bool_masks)), key=lambda i: self._mask_iou(bool_masks[i], hint))
            return bool_masks[best]

        if self.input_points:
            fg_idx = [i for i, lb in enumerate(self.input_labels) if int(lb) == 1]
            candidates = []
            for i, m in enumerate(bool_masks):
                ok = True
                for pi in fg_idx:
                    px, py = int(round(self.input_points[pi][0])), int(round(self.input_points[pi][1]))
                    if py < 0 or px < 0 or py >= m.shape[0] or px >= m.shape[1] or not m[py, px]:
                        ok = False
                        break
                if ok:
                    candidates.append(i)
            if candidates:
                # Prefer smallest area among masks that cover all positive clicks
                best = min(candidates, key=lambda i: (bool_masks[i].sum(), -score_list[i]))
                return bool_masks[best]

        best = int(np.argmax(score_list))
        return bool_masks[best]

    def run_sam_prediction(self):
        if self.predictor is None: return
        try:
            small = bool(self.small_bubble_mode.get())
            kwargs = {'multimask_output': True if small else False}
            if self.input_points:
                kwargs['point_coords'] = np.array(self.input_points)
                kwargs['point_labels'] = np.array(self.input_labels)
            
            # --- ADD BOX SUPPORT ---
            if self.input_box is not None:
                kwargs['box'] = np.array(self.input_box, dtype=np.float32)
                
            if not self.input_points and self.input_box is None: return

            masks, scores, _ = self.predictor.predict(**kwargs)
            if small and len(masks) > 1:
                raw_mask = self._pick_small_feature_mask(masks, scores)
                raw_mask = self._refine_small_bubble_mask(raw_mask)
            else:
                raw_mask = np.asarray(masks[0], dtype=bool)
                if small:
                    raw_mask = self._refine_small_bubble_mask(raw_mask)
            self.current_mask = self.apply_y_limit_constraint(raw_mask)
            self.show_image(self.raw_image, self.current_mask)
        except Exception as e:
            print(f"SAM Prediction Error: {e}")

    def delete_selected(self):
        if self.selected_annotation_index != -1:
            self.save_state_for_undo()
            del self.current_annotations[self.selected_annotation_index]
            self.selected_annotation_index = -1
            self.save_all_annotations_to_file()
            self.show_image(self.raw_image)
            self.log("Annotation Deleted", "red")
        else:
            messagebox.showinfo("Delete", "No annotation selected.\n\n1. Switch Tool to 'Edit/Select'\n2. Click a shape\n3. Click Delete")        
    
    def on_right_click(self, e):
        if self.scale == 0 or self.raw_image is None: return
        rx = int((e.x - self.off_x) / self.scale)
        ry = int((e.y - self.off_y) / self.scale)
        
        if self.mode.get() == "ai":
            if not self.is_ai_ready: return
            self.save_state_for_undo()
            
            # Add Negative Point
            self.input_points.append([rx, ry])
            self.input_labels.append(0)
            
            # Draw Red Dot Immediately
            self.show_image(self.raw_image, self.current_mask)
            self.canvas.update_idletasks()
            
            # Run AI
            self.run_sam_prediction()
            
        elif self.mode.get() == "polygon":
            # Close polygon
            if len(self.poly_points) > 2:
                self.save_state_for_undo()
                h, w = self.raw_image.shape[:2]
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(mask, [np.array(self.poly_points, np.int32)], 1)
                self.current_mask = self.apply_y_limit_constraint(mask.astype(bool))
                self.show_image(self.raw_image, self.current_mask)
                self.poly_points = []

    def ask_label_and_save(self, e=None):
        if self.raw_image is None:
            self.log("Load a volume/image before saving.", "orange")
            return "break"
        if self.current_mask is None:
            self.log("No segmented mask — use Point/Box/Poly first, then Save (S).", "orange")
            return "break"

        # Ensure a class is active (common CT failure: segment OK, no class → S does nothing useful)
        cid = self.current_class_id.get()
        if cid == -1 or cid not in self.class_mapping:
            if self.class_mapping:
                self.current_class_id.set(min(self.class_mapping.keys()))
                self.refresh_class_ui()
                cid = self.current_class_id.get()
            else:
                name = simpledialog.askstring(
                    "Class name required",
                    "Enter a class name for this segmentation\n(e.g. Gas-Pore, Bubble, Pore):",
                    initialvalue="Pore",
                    parent=self.frame,
                )
                if not name or not name.strip():
                    self.log("Save cancelled — class name required.", "orange")
                    return "break"
                self.add_class_by_name(name.strip())
                cid = self.current_class_id.get()
                if cid == -1 or cid not in self.class_mapping:
                    return "break"

        self.save_state_for_undo()
        m_u8 = (np.asarray(self.current_mask) > 0).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(m_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            h, w = self.raw_image.shape[:2]
            small = bool(self.small_bubble_mode.get())
            if small:
                min_area = 5
                max_area = 0.15 * h * w
                kept = [c for c in cnts if min_area <= cv2.contourArea(c) <= max_area]
                if not kept:
                    kept = [max(cnts, key=cv2.contourArea)]
            else:
                kept = [max(cnts, key=cv2.contourArea)]
            for c in kept:
                norm = [x for pt in c.reshape(-1, 2) for x in (pt[0] / w, pt[1] / h)]
                self.current_annotations.append({"id": cid, "coords": norm})
            try:
                self.save_all_annotations_to_file()
                self._rebuild_ct_label_volume_preview()
            except Exception as ex:
                self.log(f"Save failed: {ex}", "red")
                messagebox.showerror("Save failed", str(ex))
                return "break"
            self.current_mask = None
            self.input_points = []
            self.input_labels = []
            self.poly_points = []
            self.input_box = None
            self.show_image(self.raw_image)
            cname = self.class_mapping.get(cid, "?")
            self.log(f"Saved '{cname}' — green mask committed (Del clears unsaved mask only)", "green")
            try:
                self.canvas.focus_set()
            except tk.TclError:
                pass
        else:
            self.log("Mask has no contours to save.", "orange")
        return "break"

    def clear_current_annotation(self, e=None):
        self.save_state_for_undo()
        self.current_mask = None
        self.input_points = []
        self.input_labels = []
        self.poly_points = []
        self.input_box = None
        self.show_image(self.raw_image)
        self.log("Cleared green mask (unsaved). Saved labels unchanged.", "blue")
        return "break"
    def undo(self, e=None):
        if self.undo_stack:
            s=self.undo_stack.pop(); self.input_points=s['input_points']; self.input_labels=s['input_labels']
            self.current_mask=s['current_mask']; self.current_annotations=s['current_annotations']
            self.show_image(self.raw_image, self.current_mask); self.save_all_annotations_to_file(silent=True)
    def save_state_for_undo(self):
        self.undo_stack.append({'input_points':copy.deepcopy(self.input_points), 'input_labels':copy.deepcopy(self.input_labels), 'current_mask':self.current_mask.copy() if self.current_mask is not None else None, 'current_annotations':copy.deepcopy(self.current_annotations)})
        if len(self.undo_stack)>20: self.undo_stack.pop(0)

    # Zoom/Pan/Nav
    def on_mousewheel(self, e): 
        self.zoom_level *= (1.0/1.2 if (e.num==5 or e.delta<0) else 1.2); 
        if self.zoom_level<1: self.zoom_level=1
        self.show_image(self.raw_image, self.current_mask)
    def start_pan(self, e): self.drag_start_x, self.drag_start_y = e.x, e.y
    def do_pan(self, e): self.pan_x+=(e.x-self.drag_start_x); self.pan_y+=(e.y-self.drag_start_y); self.drag_start_x, self.drag_start_y = e.x, e.y; self.show_image(self.raw_image, self.current_mask)
    def on_resize(self, e): 
        if self.raw_image is not None: self.show_image(self.raw_image, self.current_mask)
    def reset_zoom(self): self.zoom_level=1.0; self.pan_x=0; self.pan_y=0; self.show_image(self.raw_image, self.current_mask)
    def _step_frame_prev(self):
        """Previous image (2D) or slice (CT)."""
        if self.is_ct_mode:
            if hasattr(self, "slice_slider") and self.slice_slider.winfo_exists():
                v = self.slice_slider.get()
                if v > 0:
                    self.slice_slider.set(v - 1)
        elif self.image_list and self.current_idx > 0:
            self.current_idx -= 1
            self.load_current_image_2d()

    def _step_frame_next(self):
        """Next image (2D) or slice (CT)."""
        if self.is_ct_mode:
            if hasattr(self, "slice_slider") and self.slice_slider.winfo_exists():
                v = self.slice_slider.get()
                if v < self.slice_slider.cget("to"):
                    self.slice_slider.set(v + 1)
        elif self.image_list and self.current_idx < len(self.image_list) - 1:
            self.current_idx += 1
            self.load_current_image_2d()

    def on_left_arrow(self, e):
        self._step_frame_prev()

    def on_right_arrow(self, e):
        self._step_frame_next()

    def on_up_arrow(self, e):
        self._step_frame_prev()

    def on_down_arrow(self, e):
        self._step_frame_next()
    def on_file_select(self, e):
        sel = self.file_listbox.curselection()
        if not sel:
            return
        self.current_idx = sel[0]
        if self.is_ct_mode:
            self.load_current_ct_volume()
        else:
            self.load_current_image_2d()
    def log(self, m, c="gray"): self.lbl_status.config(text=f"Status: {m}", fg=c)
    def go_back_app(self):
        try: self.plotter.close()
        except: pass
        self.frame.destroy(); self.on_back_callback()

    def _compose_sam_overlay_rgb(self, rgb, mask=None):
        """Full-resolution RGB: class polygons, SAM mask, y-limit tint (same as canvas base layer)."""
        if rgb is None:
            return None
        disp = rgb.copy()
        h, w = rgb.shape[:2]
        overlay = disp.copy()
        for idx, ann in enumerate(self.current_annotations):
            pts = np.array([[int(c*w) if k%2==0 else int(c*h) for k,c in enumerate(ann['coords'])]], np.int32).reshape((-1,1,2))
            if idx == self.selected_annotation_index:
                cv2.polylines(disp, [pts], True, (255, 255, 255), 3)
                col = (255, 255, 255)
            else:
                col = self.class_colors.get(ann['id'], (255,255,255))
            cv2.fillPoly(overlay, [pts], col)
        cv2.addWeighted(overlay, 0.4, disp, 0.6, 0, disp)
        if mask is not None:
            mask = self.apply_y_limit_constraint(mask)
            m_u8 = (mask * 255).astype(np.uint8)
            c_mask = np.zeros_like(disp)
            c_mask[:] = (0, 255, 0)
            c_mask = cv2.bitwise_and(c_mask, c_mask, mask=m_u8)
            disp = cv2.addWeighted(disp, 1.0, c_mask, 0.5, 0)
        if self.y_limit is not None:
            limit_overlay = disp.copy()
            cv2.rectangle(limit_overlay, (0, 0), (w, self.y_limit), (255, 0, 0), -1)
            cv2.line(limit_overlay, (0, self.y_limit), (w, self.y_limit), (255, 255, 0), 2)
            cv2.addWeighted(limit_overlay, 0.3, disp, 0.7, 0, disp)
        if self.x_right_limit is not None:
            sx = min(max(0, int(self.x_right_limit)), w)
            limit_overlay = disp.copy()
            cv2.rectangle(limit_overlay, (sx, 0), (w, h), (0, 200, 255), -1)
            cv2.line(limit_overlay, (sx, 0), (sx, h), (0, 255, 255), 2)
            cv2.addWeighted(limit_overlay, 0.3, disp, 0.7, 0, disp)
        return disp

    def _draw_interactive_prompts_on_rgb(self, disp):
        """Burn point/box/polygon-in-progress prompts onto full-res RGB (matches canvas overlays)."""
        out = disp.copy()
        h, w = out.shape[:2]
        pr = max(3, int(min(h, w) * 0.004))
        if self.mode.get() == "ai" and self.input_points:
            for i, (ix, iy) in enumerate(self.input_points):
                lab = self.input_labels[i]
                col = (0, 255, 0) if lab == 1 else (255, 0, 0)
                cv2.circle(out, (int(ix), int(iy)), pr, col, -1)
                cv2.circle(out, (int(ix), int(iy)), pr, (255, 255, 255), 2)
        if self.mode.get() == "polygon" and self.poly_points:
            for i, (ix, iy) in enumerate(self.poly_points):
                cv2.circle(out, (int(ix), int(iy)), max(3, pr - 1), (255, 255, 0), -1)
                cv2.circle(out, (int(ix), int(iy)), max(3, pr - 1), (255, 0, 0), 1)
                if i > 0:
                    p0 = self.poly_points[i - 1]
                    cv2.line(out, (int(p0[0]), int(p0[1])), (int(ix), int(iy)), (255, 255, 0), 2)
        if self.mode.get() == "box":
            if self.input_box is not None:
                x1, y1, x2, y2 = [int(v) for v in self.input_box]
                cv2.rectangle(out, (x1, y1), (x2, y2), (243, 156, 18), 3)
            if self.drawing_box:
                x1, y1 = self.box_start[0], self.box_start[1]
                x2, y2 = self.box_end[0], self.box_end[1]
                ax1, ax2 = min(x1, x2), max(x1, x2)
                ay1, ay2 = min(y1, y2), max(y1, y2)
                cv2.rectangle(out, (int(ax1), int(ay1)), (int(ax2), int(ay2)), (255, 255, 255), 2)
        return out

    def export_current_view_with_sam(self):
        if self.raw_image is None:
            messagebox.showwarning("Export", "Load an image first.")
            return
        m = self.current_mask.copy() if self.current_mask is not None else None
        comp = self._compose_sam_overlay_rgb(self.raw_image, m)
        comp = self._draw_interactive_prompts_on_rgb(comp)
        initial = "sam_view.png"
        if getattr(self, "image_list", None) and 0 <= self.current_idx < len(self.image_list):
            stem, _ = os.path.splitext(os.path.basename(self.image_list[self.current_idx]))
            initial = f"{stem}_sam_view.png"
        path = filedialog.asksaveasfilename(
            title="Export image with SAM overlays",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg;*.jpeg"), ("All files", "*.*")],
            initialfile=initial,
        )
        if not path:
            return
        try:
            cv2.imwrite(path, cv2.cvtColor(comp, cv2.COLOR_RGB2BGR))
            self.log(f"Exported: {os.path.basename(path)}", "green")
        except Exception as e:
            messagebox.showerror("Export failed", str(e))

    def show_image(self, rgb, mask=None):
        if rgb is None: return
        h, w = rgb.shape[:2]
        
        # Scale & Draw — resize image with Lanczos, mask with NEAREST to avoid
        # horizontal "textile" moiré streaks on the green SAM overlay.
        cw = self.canvas.winfo_width(); ch = self.canvas.winfo_height()
        if cw < 10: cw, ch = 800, 600
        self.scale = min(cw/w, ch/h) * self.zoom_level
        nw, nh = max(1, int(w*self.scale)), max(1, int(h*self.scale))

        base = self._compose_sam_overlay_rgb(rgb, mask=None)
        pil = Image.fromarray(base).resize((nw, nh), Image.Resampling.LANCZOS)
        disp = np.asarray(pil).copy()

        if mask is not None:
            m = self.apply_y_limit_constraint(mask)
            m_u8 = (np.asarray(m) > 0).astype(np.uint8) * 255
            m_disp = cv2.resize(m_u8, (nw, nh), interpolation=cv2.INTER_NEAREST)
            green = np.zeros_like(disp)
            green[:] = (0, 255, 0)
            green = cv2.bitwise_and(green, green, mask=m_disp)
            disp = cv2.addWeighted(disp, 1.0, green, 0.5, 0)
            pil = Image.fromarray(disp)
        
        self.off_x = (cw - nw) // 2 + self.pan_x
        self.off_y = (ch - nh) // 2 + self.pan_y
        self.tk_img = ImageTk.PhotoImage(pil)
        
        self.canvas.delete("all")
        self.canvas.create_image(self.off_x, self.off_y, anchor="nw", image=self.tk_img)
        self.redraw_overlays()

    def redraw_overlays(self):
        # Draw SAM Input Points
        if self.mode.get() == "ai" and self.input_points:
            for i, (ix, iy) in enumerate(self.input_points):
                label = self.input_labels[i]
                # Transform image coords to canvas coords
                cx = (ix * self.scale) + self.off_x
                cy = (iy * self.scale) + self.off_y
                
                color = "#00FF00" if label == 1 else "#FF0000" # Green=FG, Red=BG
                self.canvas.create_oval(cx-4, cy-4, cx+4, cy+4, fill=color, outline="white", width=2)
        
        # Draw Polygon Construction Lines
        if self.mode.get() == "polygon" and self.poly_points:
            for i, (ix, iy) in enumerate(self.poly_points):
                cx = (ix * self.scale) + self.off_x
                cy = (iy * self.scale) + self.off_y
                
                self.canvas.create_oval(cx-3, cy-3, cx+3, cy+3, fill="yellow", outline="red")
                
                if i > 0: # Draw line from previous point
                    pix, piy = self.poly_points[i-1]
                    pcx = (pix * self.scale) + self.off_x
                    pcy = (piy * self.scale) + self.off_y
                    self.canvas.create_line(pcx, pcy, cx, cy, fill="yellow", width=2)
                    
        # Draw Box Guide
        if self.mode.get() == "box":
            if self.input_box is not None:
                x1 = (self.input_box[0] * self.scale) + self.off_x
                y1 = (self.input_box[1] * self.scale) + self.off_y
                x2 = (self.input_box[2] * self.scale) + self.off_x
                y2 = (self.input_box[3] * self.scale) + self.off_y
                self.canvas.create_rectangle(x1, y1, x2, y2, outline="#F39C12", width=3, dash=(4, 4))
                
            if self.drawing_box:
                x1 = (self.box_start[0] * self.scale) + self.off_x
                y1 = (self.box_start[1] * self.scale) + self.off_y
                x2 = (self.box_end[0] * self.scale) + self.off_x
                y2 = (self.box_end[1] * self.scale) + self.off_y
                self.canvas.create_rectangle(x1, y1, x2, y2, outline="white", width=2)


# ==============================================================================
# YOLO OBJECT TRACKER (opened from YoloTrainerApp in this module)
# ==============================================================================
_YOLO_TRACK_PRESETS = {
    "v8": {
        "label": "YOLOv8",
        "seg_defaults": ("yolov8n-seg.pt", "yolov8s-seg.pt", "yolov8m-seg.pt", "yolov8l-seg.pt"),
        "det_defaults": ("yolov8n.pt", "yolov8s.pt", "yolov8m.pt"),
        "default_seg": "yolov8n-seg.pt",
        "match_substrings": ("yolov8", "yolo8"),
    },
    "v12": {
        "label": "YOLOv12",
        "seg_defaults": (),
        "det_defaults": ("yolo12n.pt", "yolo12s.pt", "yolo12m.pt"),
        "default_seg": "yolo12n-seg.yaml",
        "match_substrings": ("yolo12", "yolov12"),
    },
}


def _yolo_track_version_key(version: str) -> str:
    v = (version or "v8").strip().lower()
    return v if v in _YOLO_TRACK_PRESETS else "v8"


def _yolo_trainer_version_label(version: str) -> str:
    return _YOLO_TRACK_PRESETS[_yolo_track_version_key(version)]["label"]


def _yolo_trainer_default_seg_checkpoint(version: str) -> str:
    return _YOLO_TRACK_PRESETS[_yolo_track_version_key(version)]["default_seg"]


def _yolo_track_infer_version(path: str, fallback: str = "v8") -> str:
    base = os.path.basename(path or "").lower()
    for ver, cfg in _YOLO_TRACK_PRESETS.items():
        if any(m in base for m in cfg["match_substrings"]):
            return ver
    return _yolo_track_version_key(fallback)


def _yolo_track_discover_checkpoints() -> list:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_folder = os.path.join(os.path.dirname(script_dir), "models")
    search_paths = [model_folder, os.path.join(os.getcwd(), "models"), os.getcwd()]
    seen, out = set(), []
    for d in search_paths:
        if not os.path.isdir(d):
            continue
        try:
            for f in os.listdir(d):
                if f.lower().endswith((".pt", ".pth")) and f not in seen:
                    seen.add(f)
                    out.append(os.path.join(d, f))
        except OSError:
            pass
    out.sort(key=lambda p: os.path.basename(p).lower())
    return out


def _yolo_track_preferred(version: str, candidates: list) -> list:
    cfg = _YOLO_TRACK_PRESETS[_yolo_track_version_key(version)]
    preferred = []
    for name in (*cfg["seg_defaults"], *cfg["det_defaults"]):
        for p in candidates:
            if os.path.basename(p).lower() == name.lower() and p not in preferred:
                preferred.append(p)
    for p in candidates:
        base = os.path.basename(p).lower()
        if p in preferred:
            continue
        if any(m in base for m in cfg["match_substrings"]):
            preferred.append(p)
    for p in candidates:
        if p not in preferred:
            preferred.append(p)
    return preferred


class YoloTrackingViewer:
    """YOLO track() over an image sequence with object label + export video/stack."""

    def __init__(self, master, initial_folder=None, yolo_version="v8", initial_object_name=""):
        root = master.winfo_toplevel() if hasattr(master, "winfo_toplevel") else master
        self.window = tk.Toplevel(root)
        self.yolo_version = tk.StringVar(value=_yolo_track_version_key(yolo_version))
        self.window.title("YOLO Object Tracker")
        self.window.geometry("1150x900")
        self.window.configure(bg="#2C3E50")
        self.window.protocol("WM_DELETE_WINDOW", self._on_window_close)

        self.model = None
        self.model_path = ""
        self.image_files: list[str] = []
        self.current_idx = 0
        self.csv_path = None
        self.is_tracking = False
        self.tracker_name = "bytetrack.yaml"
        self._initial_folder = initial_folder
        self.locked_track_id = None
        self.export_frames: list = []
        self.export_full_frames: list = []
        self.export_raw_frames: list = []
        # Powder-bed: hide / ignore tracked object above this image row (None = off)
        self.y_limit = None
        self._picking_y_limit = False
        self._disp_scale = 1.0
        self._disp_w = 0
        self._disp_h = 0
        self._img_h = 0
        self._img_w = 0

        self.object_label = tk.StringVar(value=(initial_object_name or "").strip())
        self._tracking_instance_name = ""
        self.object_label.trace_add("write", lambda *_: self._update_window_title())
        self.target_class_id = tk.StringVar(value="0")
        self.track_conf = tk.DoubleVar(
            value=0.10 if self.yolo_version.get() == "v12" else 0.25
        )
        self.record_for_export = tk.BooleanVar(value=True)
        self.export_crop_mode = tk.BooleanVar(value=False)
        self.y_limit_var = tk.StringVar(value="")

        ctrl = tk.Frame(self.window, bg="#34495E", pady=8)
        ctrl.pack(side="top", fill="x")

        tk.Button(
            ctrl,
            text="YOLO demo →",
            command=lambda: launch_yolo_demonstration(self.window),
            bg="#154360",
            fg="white",
            font=("Arial", 9, "bold"),
            cursor="hand2",
        ).pack(side="right", padx=8)
        tk.Button(ctrl, text="1. Load model", command=self.load_model, bg="#8E44AD", fg="white", font=("Arial", 10, "bold")).pack(
            side="left", padx=6
        )
        tk.Button(ctrl, text="2. Load image folder", command=self.load_images, bg="#2980B9", fg="white", font=("Arial", 10, "bold")).pack(
            side="left", padx=6
        )
        self.btn_track = tk.Button(
            ctrl,
            text="3. Start tracking",
            command=self.start_tracking,
            bg="#27AE60",
            fg="white",
            font=("Arial", 10, "bold"),
            state="disabled",
        )
        self.btn_track.pack(side="left", padx=6)
        self.btn_export_video = tk.Button(
            ctrl,
            text="Save video",
            command=lambda: self.export_tracked_media(video=True, stack=False),
            bg="#16A085",
            fg="white",
            font=("Arial", 10, "bold"),
            state="disabled",
        )
        self.btn_export_video.pack(side="left", padx=4)
        self.btn_export_stack = tk.Button(
            ctrl,
            text="Save image stack",
            command=lambda: self.export_tracked_media(video=False, stack=True),
            bg="#16A085",
            fg="white",
            font=("Arial", 10, "bold"),
            state="disabled",
        )
        self.btn_export_stack.pack(side="left", padx=4)

        self.lbl_info = tk.Label(
            ctrl,
            text="Label the object, load model + folder, then track. Export after tracking.",
            bg="#34495E",
            fg="white",
            font=("Arial", 9),
            wraplength=380,
            justify="left",
        )
        self.lbl_info.pack(side="left", padx=10)

        opts = tk.LabelFrame(
            self.window,
            text="Instance name (your label for this tracking run)",
            bg="#2C3E50",
            fg="white",
            font=("Arial", 10, "bold"),
            padx=10,
            pady=6,
        )
        opts.pack(side="top", fill="x", padx=10, pady=(0, 4))

        row1 = tk.Frame(opts, bg="#2C3E50")
        row1.pack(fill="x", pady=2)
        tk.Label(row1, text="Name:", bg="#2C3E50", fg="white", font=("Arial", 10, "bold")).pack(side="left")
        inst_entry = tk.Entry(row1, textvariable=self.object_label, width=28, font=("Arial", 11))
        inst_entry.pack(side="left", padx=8)
        for preset in ("Keyhole", "Tool pin", "KH-Pore", "Gas pore"):
            tk.Button(
                row1,
                text=preset,
                command=lambda p=preset: self.object_label.set(p),
                bg="#566573",
                fg="white",
                font=("Arial", 8),
                padx=4,
            ).pack(side="left", padx=2)
        tk.Label(
            row1,
            text="Used on video/CSV exports. Not the YOLO weight class name (e.g. Keyhole in best.pt).",
            bg="#2C3E50",
            fg="#AED6F1",
            font=("Arial", 8),
        ).pack(side="left", padx=(8, 0))
        row1b = tk.Frame(opts, bg="#2C3E50")
        row1b.pack(fill="x", pady=2)
        tk.Label(row1b, text="YOLO class ID (blank = any):", bg="#2C3E50", fg="white", font=("Arial", 10)).pack(side="left")
        tk.Entry(row1b, textvariable=self.target_class_id, width=6, font=("Arial", 10)).pack(side="left", padx=6)
        tk.Label(row1b, text="Min confidence:", bg="#2C3E50", fg="white", font=("Arial", 10)).pack(side="left", padx=(14, 4))
        tk.Entry(row1b, textvariable=self.track_conf, width=5, font=("Arial", 10)).pack(side="left")
        tk.Label(
            row1b,
            text="(lower = more sensitive; try 0.10 for YOLOv12)",
            bg="#2C3E50",
            fg="#AED6F1",
            font=("Arial", 8),
        ).pack(side="left", padx=6)

        row2 = tk.Frame(opts, bg="#2C3E50")
        row2.pack(fill="x", pady=2)
        tk.Checkbutton(
            row2,
            text="Record frames while tracking (for video / stack export)",
            variable=self.record_for_export,
            bg="#2C3E50",
            fg="white",
            selectcolor="#34495E",
            activebackground="#2C3E50",
            activeforeground="white",
            font=("Arial", 9),
        ).pack(side="left")
        tk.Checkbutton(
            row2,
            text="Crop video export to object (image stack always saves full frame)",
            variable=self.export_crop_mode,
            bg="#2C3E50",
            fg="white",
            selectcolor="#34495E",
            activebackground="#2C3E50",
            activeforeground="white",
            font=("Arial", 9),
        ).pack(side="left", padx=12)

        row3 = tk.Frame(opts, bg="#2C3E50")
        row3.pack(fill="x", pady=(4, 2))
        tk.Label(
            row3,
            text="Y limit (hide tracks above line):",
            bg="#2C3E50",
            fg="#F9E79F",
            font=("Arial", 10, "bold"),
        ).pack(side="left")
        self.btn_y_limit = tk.Button(
            row3,
            text="⛔ Set Y limit (click image)",
            command=self.toggle_y_limit_pick,
            bg="#EC7063",
            fg="white",
            font=("Arial", 9, "bold"),
        )
        self.btn_y_limit.pack(side="left", padx=6)
        tk.Label(row3, text="Y=", bg="#2C3E50", fg="white", font=("Arial", 9)).pack(side="left")
        tk.Entry(row3, textvariable=self.y_limit_var, width=6, font=("Arial", 10)).pack(side="left", padx=2)
        tk.Button(
            row3,
            text="Apply",
            command=self.apply_y_limit_from_entry,
            bg="#F39C12",
            fg="white",
            font=("Arial", 8, "bold"),
        ).pack(side="left", padx=4)
        tk.Button(
            row3,
            text="Clear",
            command=self.clear_y_limit,
            bg="#7F8C8D",
            fg="white",
            font=("Arial", 8),
        ).pack(side="left", padx=2)
        self.lbl_y_limit = tk.Label(
            row3,
            text="Off — tracks shown fully",
            bg="#2C3E50",
            fg="#AED6F1",
            font=("Arial", 8),
        )
        self.lbl_y_limit.pack(side="left", padx=8)

        self.canvas = tk.Label(self.window, bg="black", cursor="crosshair")
        self.canvas.pack(expand=True, fill="both", padx=10, pady=10)
        self.canvas.bind("<Button-1>", self._on_canvas_click)

        if initial_folder and os.path.isdir(initial_folder):
            self._load_images_from_dir(initial_folder)
        self._update_window_title()

    def _instance_display_name(self) -> str:
        return (self.object_label.get() or self._tracking_instance_name or "").strip()

    def _on_window_close(self):
        self.window.destroy()

    def _update_window_title(self):
        ver = self.yolo_version.get()
        yolo_label = _YOLO_TRACK_PRESETS[ver]["label"]
        inst = self._instance_display_name()
        if inst:
            self.window.title(f"{inst} — {yolo_label} Tracker")
        else:
            self.window.title(f"{yolo_label} Object Tracker")

    def _sanitize_export_name(self) -> str:
        raw = self._instance_display_name() or "tracked_object"
        safe = re.sub(r"[^\w\-]+", "_", raw)
        return safe[:64] or "tracked_object"

    def _export_buttons_state(self):
        has = len(self.export_frames) > 0
        st = "normal" if has else "disabled"
        self.btn_export_video.config(state=st)
        self.btn_export_stack.config(state=st)

    def _yolo_weights_start_dir(self) -> str:
        """SMAXI master project folder (parent of 1. Python code)."""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        root_folder = os.path.dirname(script_dir)
        if os.path.isdir(root_folder):
            return root_folder
        return os.getcwd()

    def load_model(self):
        if not HAS_YOLO:
            messagebox.showerror("YOLO", "Ultralytics not found.\nRun: pip install -U ultralytics")
            return

        path = filedialog.askopenfilename(
            parent=self.window,
            title="Select YOLO weights (.pt)",
            initialdir=self._yolo_weights_start_dir(),
            filetypes=[("YOLO weights", "*.pt"), ("All files", "*.*")],
        )
        if path:
            self._load_yolo_weights(path)

    def _load_yolo_weights(self, path: str):
        try:
            from ultralytics import YOLO

            self.model = YOLO(path)
            self.model_path = path
            ver = _yolo_track_infer_version(path, self.yolo_version.get())
            self.yolo_version.set(ver)
            self.track_conf.set(0.10 if ver == "v12" else 0.25)
            self._update_window_title()
            self.lbl_info.config(
                text=f"{_YOLO_TRACK_PRESETS[ver]['label']} | Model: {os.path.basename(path)}",
                fg="white",
            )
            self.check_ready()
        except Exception as e:
            messagebox.showerror("Model error", f"Failed to load model:\n{e}", parent=self.window)

    def load_images(self):
        start = self._initial_folder or os.getcwd()
        folder = filedialog.askdirectory(
            parent=self.window,
            title="Select image folder",
            initialdir=start,
        )
        if folder:
            self._load_images_from_dir(folder)

    def _load_images_from_dir(self, folder: str):
        path_obj = Path(folder)
        exts = ["*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff", "*.bmp"]
        found_set: set[str] = set()
        for ext in exts:
            for f in path_obj.glob(ext):
                found_set.add(str(f))
            for f in path_obj.glob(ext.upper()):
                found_set.add(str(f))
        self.image_files = natsorted(list(found_set))
        if self.image_files:
            self.lbl_info.config(
                text=f"Model: {os.path.basename(self.model_path) or '(none)'} | Images: {len(self.image_files)}",
                fg="white",
            )
            self.check_ready()
            self.current_idx = 0
            self.show_preview()
        else:
            messagebox.showwarning("Empty", f"No images found in:\n{folder}")

    def check_ready(self):
        if self.model and self.image_files:
            self.btn_track.config(state="normal")

    def show_preview(self):
        if not self.image_files:
            return
        img = imread_safe(self.image_files[0])
        if img is not None:
            self.update_canvas(self._draw_y_limit_guide(img))

    def toggle_y_limit_pick(self):
        if self._picking_y_limit:
            self._picking_y_limit = False
            self.btn_y_limit.config(text="⛔ Set Y limit (click image)", bg="#EC7063")
            self.lbl_info.config(text="Y-limit pick cancelled.", fg="#AED6F1")
            return
        if not self.image_files:
            messagebox.showinfo("Y limit", "Load an image folder first, then click on the preview.", parent=self.window)
            return
        self._picking_y_limit = True
        self.btn_y_limit.config(text="Click image… (or cancel)", bg="#F1C40F")
        self.lbl_info.config(
            text="Click the preview where the powder / ignore zone ends. Tracks above that Y are hidden.",
            fg="#F9E79F",
        )
        self.show_preview()

    def apply_y_limit_from_entry(self):
        raw = (self.y_limit_var.get() or "").strip()
        if not raw:
            messagebox.showwarning("Y limit", "Enter a Y pixel value (row from top), or click the image.", parent=self.window)
            return
        try:
            y = int(float(raw))
        except ValueError:
            messagebox.showwarning("Y limit", "Y must be an integer pixel row.", parent=self.window)
            return
        h = self._img_h
        if h <= 0 and self.image_files:
            img = imread_safe(self.image_files[0])
            h = img.shape[0] if img is not None else 0
        if h > 0:
            y = max(0, min(y, h - 1))
        self.y_limit = y
        self.y_limit_var.set(str(y))
        self._picking_y_limit = False
        self.btn_y_limit.config(text="⛔ Set Y limit (click image)", bg="#EC7063")
        self._refresh_y_limit_label()
        self.show_preview()

    def clear_y_limit(self):
        self.y_limit = None
        self.y_limit_var.set("")
        self._picking_y_limit = False
        self.btn_y_limit.config(text="⛔ Set Y limit (click image)", bg="#EC7063")
        self._refresh_y_limit_label()
        self.show_preview()

    def _refresh_y_limit_label(self):
        if self.y_limit is None:
            self.lbl_y_limit.config(text="Off — tracks shown fully", fg="#AED6F1")
        else:
            self.lbl_y_limit.config(
                text=f"Active: hide / clip above Y={self.y_limit}",
                fg="#2ECC71",
            )

    def _on_canvas_click(self, event):
        if not self._picking_y_limit:
            return
        if self._disp_scale <= 0 or self._disp_h <= 0:
            return
        lw = max(1, self.canvas.winfo_width())
        lh = max(1, self.canvas.winfo_height())
        ox = (lw - self._disp_w) // 2
        oy = (lh - self._disp_h) // 2
        ix = event.x - ox
        iy = event.y - oy
        if ix < 0 or iy < 0 or ix >= self._disp_w or iy >= self._disp_h:
            return
        y_full = int(round(iy / self._disp_scale))
        if self._img_h > 0:
            y_full = max(0, min(y_full, self._img_h - 1))
        self.y_limit = y_full
        self.y_limit_var.set(str(y_full))
        self._picking_y_limit = False
        self.btn_y_limit.config(text="⛔ Set Y limit (click image)", bg="#EC7063")
        self._refresh_y_limit_label()
        self.lbl_info.config(
            text=f"Y limit set at row {y_full} — tracked objects above this line are ignored.",
            fg="#2ECC71",
        )
        self.show_preview()

    def _draw_y_limit_guide(self, img_bgr):
        """Tint + yellow line for the ignore zone (above y_limit)."""
        if img_bgr is None or self.y_limit is None:
            return img_bgr
        out = img_bgr.copy()
        h, w = out.shape[:2]
        yl = int(max(0, min(self.y_limit, h)))
        if yl <= 0:
            return out
        tint = out.copy()
        cv2.rectangle(tint, (0, 0), (w, yl), (40, 40, 200), -1)
        cv2.addWeighted(tint, 0.28, out, 0.72, 0, out)
        cv2.line(out, (0, yl), (w, yl), (0, 255, 255), 2)
        cv2.putText(
            out,
            f"Y limit={yl} (ignore above)",
            (10, max(22, yl - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return out

    def _box_center_y(self, box) -> float:
        return float(box.xywh[0].tolist()[1])

    def _box_bottom_y(self, box) -> float:
        _x, y, _bw, bh = box.xywh[0].tolist()
        return float(y + bh / 2.0)

    def _detection_below_y_limit(self, box) -> bool:
        """True if detection should be kept (not entirely above the Y limit)."""
        if self.y_limit is None:
            return True
        yl = float(self.y_limit)
        # Center in the ignore zone → discard (typical false keyhole in powder)
        if self._box_center_y(box) < yl:
            return False
        # Entire box above the line → discard
        if self._box_bottom_y(box) <= yl:
            return False
        return True

    def _clip_xyxy_to_y_limit(self, x1, y1, x2, y2, h, w):
        if self.y_limit is None:
            return x1, y1, x2, y2
        yl = int(self.y_limit)
        y1 = max(y1, yl)
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h))
        x1 = max(0, min(x1, w))
        x2 = max(0, min(x2, w))
        return x1, y1, x2, y2

    def _clip_polygon_to_y_limit(self, poly):
        if self.y_limit is None or poly is None:
            return poly
        pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
        if pts.size == 0:
            return pts
        yl = float(self.y_limit)
        # Keep vertices on/below the line; clamp those above onto the line
        below = pts[pts[:, 1] >= yl]
        if below.shape[0] >= 3:
            return below
        clipped = pts.copy()
        clipped[:, 1] = np.maximum(clipped[:, 1], yl)
        return clipped

    def _parse_class_filter(self):
        s = (self.target_class_id.get() or "").strip()
        if not s:
            return None
        try:
            return int(s)
        except ValueError:
            return None

    def _select_detection_index(self, boxes, names):
        """Pick detection index: lock track ID after first match, else highest conf."""
        if boxes is None or len(boxes) == 0:
            return None
        cls_filter = self._parse_class_filter()

        def scan(locked_id):
            best_i = None
            best_conf = -1.0
            for i, box in enumerate(boxes):
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                if cls_filter is not None and cls_id != cls_filter:
                    continue
                if not self._detection_below_y_limit(box):
                    continue
                track_id = int(box.id[0]) if box.id is not None else -1
                if locked_id is not None and track_id != locked_id:
                    continue
                if conf > best_conf:
                    best_conf = conf
                    best_i = i
            return best_i

        best_i = scan(self.locked_track_id)
        if best_i is None and self.locked_track_id is not None:
            self.locked_track_id = None
            best_i = scan(None)

        if best_i is not None and self.locked_track_id is None:
            tid = boxes[best_i].id
            if tid is not None:
                self.locked_track_id = int(tid[0])

        return best_i

    def _track_confidence(self) -> float:
        try:
            conf = float(self.track_conf.get())
        except (TypeError, tk.TclError, ValueError):
            conf = 0.25
        return float(min(1.0, max(0.01, conf)))

    def _overlay_font_scale(self, h: int, w: int) -> float:
        """Medium label size — between the original 0.65 and the larger adaptive scale."""
        return float(max(0.72, min(1.0, min(h, w) / 620.0)))

    def _draw_tracked_overlay(
        self,
        img_bgr,
        boxes,
        masks,
        det_i,
        label: str,
        conf: float | None = None,
        show_text: bool = True,
    ) -> np.ndarray:
        """Draw only the selected track using the user label (never YOLO class names)."""
        label = (label or "").strip()
        h, w = img_bgr.shape[:2]
        out = self._draw_y_limit_guide(img_bgr.copy())
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = self._overlay_font_scale(h, w)
        thick = max(2, int(round(scale * 1.5)))

        if det_i is None or boxes is None or det_i >= len(boxes):
            if show_text:
                nd_text = f"{label} (not detected)" if label else "(not detected)"
                cv2.putText(
                    out,
                    nd_text,
                    (12, int(36 * scale)),
                    font,
                    scale,
                    (0, 200, 255),
                    thick,
                    cv2.LINE_AA,
                )
            return out

        box = boxes[det_i]
        if conf is None:
            conf = float(box.conf[0])
        x, y, bw, bh = box.xywh[0].tolist()
        x1 = int(max(0, x - bw / 2))
        y1 = int(max(0, y - bh / 2))
        x2 = int(min(w, x + bw / 2))
        y2 = int(min(h, y + bh / 2))
        x1, y1, x2, y2 = self._clip_xyxy_to_y_limit(x1, y1, x2, y2, h, w)
        if y2 <= y1:
            if show_text:
                nd_text = f"{label} (above Y limit)" if label else "(above Y limit)"
                cv2.putText(
                    out,
                    nd_text,
                    (12, int(36 * scale)),
                    font,
                    scale,
                    (0, 200, 255),
                    thick,
                    cv2.LINE_AA,
                )
            return out

        if masks is not None and det_i < len(masks.xy):
            poly = self._clip_polygon_to_y_limit(masks.xy[det_i])
            poly = np.array(poly, dtype=np.int32)
            if len(poly) >= 3:
                overlay = out.copy()
                cv2.fillPoly(overlay, [poly], (255, 128, 0))
                cv2.addWeighted(overlay, 0.4, out, 0.6, 0, out)
                cv2.polylines(out, [poly], True, (255, 200, 0), 2)

        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 180, 0), max(2, thick))
        if show_text:
            tag = f"{label} {conf:.2f}".strip() if label else f"{conf:.2f}"
            (tw, th), baseline = cv2.getTextSize(tag, font, scale, thick)
            pad = int(6 * scale)
            ty1 = max(0, y1 - th - baseline - pad)
            ty2 = ty1 + th + baseline + pad
            tx2 = min(w, x1 + tw + pad)
            cv2.rectangle(out, (x1, ty1), (tx2, ty2), (255, 128, 0), -1)
            cv2.putText(
                out,
                tag,
                (x1 + pad // 2, ty2 - baseline - pad // 4),
                font,
                scale,
                (255, 255, 255),
                thick,
                cv2.LINE_AA,
            )
        return out

    def _match_frame_size(self, img_bgr, h: int, w: int):
        if img_bgr is None:
            return None
        if img_bgr.shape[0] == h and img_bgr.shape[1] == w:
            return img_bgr
        return cv2.resize(img_bgr, (w, h))

    def _compose_side_by_side(self, raw_bgr, tracked_bgr) -> np.ndarray:
        """Raw on the left, tracked overlay on the right (same size, synchronized)."""
        h = max(raw_bgr.shape[0], tracked_bgr.shape[0])
        w = max(raw_bgr.shape[1], tracked_bgr.shape[1])
        left = self._match_frame_size(raw_bgr, h, w)
        right = self._match_frame_size(tracked_bgr, h, w)
        return np.hstack([left, right])

    def _write_video(self, path: str, frames: list, fps: float) -> str:
        if not frames:
            raise ValueError("No frames to write")
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(path, fourcc, max(1.0, float(fps)), (w, h))
        if not vw.isOpened():
            path = os.path.splitext(path)[0] + ".avi"
            vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"XVID"), max(1.0, float(fps)), (w, h))
        for fr in frames:
            if fr.shape[0] != h or fr.shape[1] != w:
                fr = cv2.resize(fr, (w, h))
            vw.write(fr)
        vw.release()
        return path

    def _build_export_frame(self, img_bgr, boxes, masks, det_i: int) -> np.ndarray:
        label = self._tracking_instance_name or self._instance_display_name()
        conf = None
        if det_i is not None and boxes is not None and det_i < len(boxes):
            conf = float(boxes[det_i].conf[0])

        full = self._draw_tracked_overlay(img_bgr, boxes, masks, det_i, label, conf)
        if not self.export_crop_mode.get() or det_i is None or boxes is None or det_i >= len(boxes):
            return full

        h, w = img_bgr.shape[:2]
        box = boxes[det_i]
        x, y, bw, bh = box.xywh[0].tolist()
        x1 = int(max(0, x - bw / 2))
        y1 = int(max(0, y - bh / 2))
        x2 = int(min(w, x + bw / 2))
        y2 = int(min(h, y + bh / 2))
        x1, y1, x2, y2 = self._clip_xyxy_to_y_limit(x1, y1, x2, y2, h, w)
        if y2 <= y1:
            return full
        pad = int(0.1 * max(x2 - x1, y2 - y1, 1))
        cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
        cx2, cy2 = min(w, x2 + pad), min(h, y2 + pad)
        return full[cy1:cy2, cx1:cx2].copy()

    def start_tracking(self):
        if self.is_tracking:
            return
        label = self._instance_display_name()
        if not label:
            messagebox.showwarning(
                "Instance name",
                "Enter a name for what you are tracking (e.g. Tool pin, Keyhole).\n\n"
                "This is shown on exports — separate from the class name inside best.pt.",
                parent=self.window,
            )
            return

        self._tracking_instance_name = label
        self._update_window_title()

        self.csv_path = filedialog.asksaveasfilename(
            parent=self.window,
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
            initialfile=f"{self._sanitize_export_name()}_tracking.csv",
        )
        if not self.csv_path:
            return
        try:
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "Frame_ID",
                        "Instance_Name",
                        "Track_ID",
                        "YOLO_Class_ID",
                        "YOLO_Model_Name",
                        "Center_X",
                        "Center_Y",
                        "Width",
                        "Height",
                        "Confidence",
                        "Polygon",
                    ]
                )
        except OSError as e:
            messagebox.showerror("CSV error", str(e))
            return

        self.is_tracking = True
        self.current_idx = 0
        self.locked_track_id = None
        self.export_frames = []
        self.export_full_frames = []
        self.export_raw_frames = []
        self._export_buttons_state()
        self.btn_track.config(state="disabled", text="Tracking…")
        self.process_next_frame()

    def process_next_frame(self):
        if not self.is_tracking or self.current_idx >= len(self.image_files):
            self.is_tracking = False
            self.btn_track.config(state="normal", text="3. Start tracking")
            self._export_buttons_state()
            inst = self._tracking_instance_name or self._instance_display_name()
            YoloTrackingValidationWindow(
                self.window,
                image_files=self.image_files,
                csv_path=self.csv_path,
                instance_name=inst or "",
            )
            msg = f"Tracking finished for: {inst or '(unnamed)'}\n\nCSV: {self.csv_path}"
            if self.export_frames:
                msg += f"\n{len(self.export_frames)} frame(s) ready — use Save video / Save image stack."
            msg += "\n\nReview and correct segmentations in the validation window."
            messagebox.showinfo("Tracking complete", msg, parent=self.window)
            return

        try:
            img_path = self.image_files[self.current_idx]
            results = self.model.track(
                img_path,
                persist=True,
                tracker=self.tracker_name,
                verbose=False,
                conf=self._track_confidence(),
                imgsz=640,
            )
            result = results[0]

            boxes = result.boxes
            masks = result.masks
            names = result.names or {}
            obj_label = self._tracking_instance_name or self.object_label.get().strip()

            det_i = self._select_detection_index(boxes, names)
            best_row = None
            conf = None

            if det_i is not None:
                box = boxes[det_i]
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                x, y, bw, bh = box.xywh[0].tolist()
                # Clip reported geometry to Y limit for CSV / analysis
                if self.y_limit is not None:
                    yl = float(self.y_limit)
                    y1 = y - bh / 2.0
                    y2 = y + bh / 2.0
                    y1 = max(y1, yl)
                    if y2 > y1:
                        y = 0.5 * (y1 + y2)
                        bh = y2 - y1
                    else:
                        det_i = None
                        conf = None
                track_id = int(box.id[0]) if box.id is not None else -1
                cls_name = names.get(cls_id, str(cls_id))
                poly_str = ""
                if det_i is not None and masks is not None and det_i < len(masks.xy):
                    poly = self._clip_polygon_to_y_limit(masks.xy[det_i])
                    poly_str = str(np.asarray(poly).tolist())
                if det_i is not None:
                    best_row = [
                        self.current_idx,
                        obj_label,
                        track_id,
                        cls_id,
                        cls_name,
                        x,
                        y,
                        bw,
                        bh,
                        conf,
                        poly_str,
                    ]

            with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if best_row is not None:
                    writer.writerow(best_row)
                else:
                    writer.writerow([self.current_idx, obj_label, -1, -1, "", "NaN", "NaN", "NaN", "NaN", 0, ""])

            raw = imread_safe(img_path)
            display_img = raw
            if raw is not None:
                display_img = self._draw_tracked_overlay(raw, boxes, masks, det_i, obj_label, conf)

            if self.record_for_export.get() and raw is not None:
                self.export_raw_frames.append(raw.copy())
                self.export_frames.append(self._build_export_frame(raw, boxes, masks, det_i))
                self.export_full_frames.append(display_img)

            self.update_canvas(display_img)
            prefix = f"[{obj_label}] " if obj_label else ""
            self.lbl_info.config(
                text=f"{prefix}Frame {self.current_idx + 1}/{len(self.image_files)} | {os.path.basename(img_path)}",
                fg="#A9DFBF",
            )
            self.current_idx += 1
            self.window.after(1, self.process_next_frame)

        except Exception as e:
            self.is_tracking = False
            self.btn_track.config(state="normal", text="3. Start tracking")
            messagebox.showerror("Tracking error", f"Frame {self.current_idx}:\n{e}")

    def export_tracked_media(self, video=True, stack=True):
        if not self.export_frames:
            messagebox.showwarning(
                "Export",
                "No recorded frames.\nEnable 'Record frames while tracking' and run tracking again.",
            )
            return

        dlg = tk.Toplevel(self.window)
        inst = self._tracking_instance_name or self._instance_display_name()
        dlg.title(f"Export — {inst}" if inst else "Export tracked instance")
        dlg.geometry("420x300")
        dlg.transient(self.window)

        out_var = tk.StringVar()
        fps_var = tk.DoubleVar(value=10.0)
        video_var = tk.BooleanVar(value=video)
        stack_var = tk.BooleanVar(value=stack)
        use_full_var = tk.BooleanVar(value=True)
        side_by_side_var = tk.BooleanVar(value=False)

        tk.Label(dlg, text="Output folder", font=("Arial", 10, "bold")).pack(anchor="w", padx=12, pady=(10, 4))
        row = tk.Frame(dlg)
        row.pack(fill="x", padx=12)
        tk.Entry(row, textvariable=out_var, width=32).pack(side="left", fill="x", expand=True)
        tk.Button(row, text="Browse…", command=lambda: out_var.set(filedialog.askdirectory() or out_var.get())).pack(
            side="left", padx=4
        )

        opts = tk.Frame(dlg, padx=12, pady=8)
        opts.pack(fill="x")
        tk.Checkbutton(opts, text="MP4 video", variable=video_var).pack(anchor="w")
        tk.Checkbutton(opts, text="PNG image stack", variable=stack_var).pack(anchor="w")
        tk.Checkbutton(
            opts,
            text="Use full-frame view (with your label) instead of cropped export",
            variable=use_full_var,
        ).pack(anchor="w", pady=(4, 0))
        tk.Checkbutton(
            opts,
            text="Side-by-side video (raw left | tracked right with your label + confidence)",
            variable=side_by_side_var,
        ).pack(anchor="w", pady=(2, 0))
        tk.Label(opts, text="FPS:").pack(anchor="w", pady=(6, 0))
        tk.Spinbox(opts, from_=1, to=60, increment=1, textvariable=fps_var, width=6).pack(anchor="w")

        def _run():
            folder = out_var.get().strip()
            if not folder:
                messagebox.showwarning("Export", "Choose an output folder.", parent=dlg)
                return
            if not video_var.get() and not stack_var.get() and not side_by_side_var.get():
                messagebox.showwarning("Export", "Enable at least one export option.", parent=dlg)
                return
            dlg.destroy()
            video_frames = self.export_full_frames if use_full_var.get() else self.export_frames
            stack_frames = self.export_full_frames or self.export_raw_frames or self.export_frames
            tag = self._sanitize_export_name()
            os.makedirs(folder, exist_ok=True)
            stack_dir = os.path.join(folder, f"{tag}_frames")
            saved = []
            try:
                fps = max(1.0, float(fps_var.get()))
                if stack_var.get():
                    os.makedirs(stack_dir, exist_ok=True)
                    for i, fr in enumerate(stack_frames):
                        imwrite_safe(os.path.join(stack_dir, f"{tag}_{i:05d}.png"), fr)
                    saved.append(f"Stack: {stack_dir} ({len(stack_frames)} full-frame image(s))")
                if video_var.get():
                    mp4 = self._write_video(os.path.join(folder, f"{tag}_tracking.mp4"), video_frames, fps)
                    saved.append(f"Video: {mp4}")
                if side_by_side_var.get():
                    if not self.export_raw_frames or not self.export_full_frames:
                        raise ValueError("Side-by-side export requires recorded raw + tracked frames.")
                    n = min(len(self.export_raw_frames), len(self.export_full_frames))
                    dual = [
                        self._compose_side_by_side(self.export_raw_frames[i], self.export_full_frames[i])
                        for i in range(n)
                    ]
                    mp4 = self._write_video(os.path.join(folder, f"{tag}_raw_tracked_side_by_side.mp4"), dual, fps)
                    saved.append(f"Side-by-side: {mp4}")
                messagebox.showinfo("Export complete", "Saved to:\n" + folder + "\n\n" + "\n".join(saved))
            except Exception as e:
                messagebox.showerror("Export failed", str(e))

        tk.Button(dlg, text="Export", command=_run, bg="#16A085", fg="white", font=("Arial", 10, "bold")).pack(
            fill="x", padx=12, pady=10
        )

    def update_canvas(self, cv_img):
        if cv_img is None:
            return
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        self._img_h, self._img_w = h, w
        disp_h = min(720, max(400, self.canvas.winfo_height() or 700))
        scale = disp_h / max(h, 1)
        disp_w = int(w * scale)
        self._disp_scale = scale
        self._disp_w = disp_w
        self._disp_h = disp_h
        resized = cv2.resize(rgb, (disp_w, disp_h))
        photo = ImageTk.PhotoImage(Image.fromarray(resized))
        self.canvas.config(image=photo)
        self.canvas.image = photo


# ==============================================================================
# YOLO TRACKING VALIDATION (review / SAM / polygon correction after track)
# ==============================================================================

def _sam_discover_checkpoints() -> list:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_folder = os.path.join(os.path.dirname(script_dir), "models")
    search_paths = [model_folder, os.path.join(os.getcwd(), "models"), os.getcwd()]
    seen, out = set(), []
    for d in search_paths:
        if not os.path.isdir(d):
            continue
        try:
            for f in os.listdir(d):
                fl = f.lower()
                if fl.endswith((".pth", ".pt")) and "yolo" not in fl and f not in seen:
                    seen.add(f)
                    out.append(os.path.join(d, f))
        except OSError:
            pass
    out.sort(key=lambda p: os.path.basename(p).lower())
    return out


def _sam_init_predictor(checkpoint: str, device: str):
    ckpt_name = os.path.basename(checkpoint).lower()
    if "sam2" in ckpt_name:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        from hydra.core.global_hydra import GlobalHydra
        from hydra import initialize_config_dir

        model_dir = os.path.abspath(os.path.dirname(checkpoint))
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
        if "tiny" in ckpt_name or " t " in ckpt_name or "hiera_t" in ckpt_name:
            cfg = "sam2.1_hiera_t.yaml"
        elif "small" in ckpt_name or " s " in ckpt_name or "hiera_s" in ckpt_name:
            cfg = "sam2.1_hiera_s.yaml"
        elif "large" in ckpt_name or " l " in ckpt_name or "hiera_l" in ckpt_name:
            cfg = "sam2.1_hiera_l.yaml"
        else:
            cfg = "sam2.1_hiera_b+.yaml"
        with initialize_config_dir(config_dir=model_dir, version_base="1.2"):
            sam2_model = build_sam2(cfg, checkpoint, device=device)
        return SAM2ImagePredictor(sam2_model), True

    from segment_anything import sam_model_registry, SamPredictor

    if "vit_h" in ckpt_name:
        model_type = "vit_h"
    elif "vit_l" in ckpt_name:
        model_type = "vit_l"
    else:
        model_type = "vit_b"
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device=device)
    return SamPredictor(sam), False


def _parse_tracking_polygon(poly_str: str):
    if not poly_str or str(poly_str).strip() in ("", "[]", "None"):
        return []
    try:
        import ast

        data = ast.literal_eval(str(poly_str))
        if not data:
            return []
        arr = np.asarray(data, dtype=np.float64)
        if arr.ndim == 1 and len(arr) >= 2:
            return arr.reshape(-1, 2).tolist()
        if arr.ndim == 2 and arr.shape[1] >= 2:
            return arr[:, :2].tolist()
    except (ValueError, SyntaxError, TypeError):
        pass
    return []


def _polygon_to_csv_str(poly: list) -> str:
    if not poly:
        return ""
    return str([[float(x), float(y)] for x, y in poly])


def _mask_to_polygon(mask: np.ndarray) -> list:
    if mask is None:
        return []
    m = (mask.astype(np.uint8) * 255) if mask.dtype != np.uint8 else mask
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return []
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 4:
        return []
    return c.reshape(-1, 2).astype(float).tolist()


class YoloTrackingValidationWindow:
    """Post-tracking QA: browse frames, verify YOLO masks, fix with SAM or polygon."""

    def __init__(self, parent, image_files, csv_path, instance_name=""):
        utils.init_ml_dependencies()
        _sync_ml_flags_from_utils()
        import torch

        self.window = tk.Toplevel(parent)
        self.window.title(f"Tracking validation — {instance_name or 'object'}")
        self.window.geometry("1400x920")
        self.window.configure(bg="#2C3E50")
        try:
            self.window.state("zoomed")
        except tk.TclError:
            pass

        self.image_files = list(image_files or [])
        self.csv_path = csv_path
        self.instance_name = (instance_name or "").strip()
        self.device = "cuda" if (torch.cuda.is_available() and HAS_SAM) else "cpu"
        self.predictor = None
        self.is_sam2 = False
        self.is_ai_ready = False

        self.rows_by_frame: dict[int, dict] = {}
        self.original_polygons: dict[int, list] = {}
        self.frame_status: dict[int, str] = {}
        self.current_idx = 0
        self._filtered_indices: list[int] = []

        self.raw_bgr = None
        self.raw_rgb = None
        self.active_polygon: list = []
        self.preview_mask = None
        self.mode = tk.StringVar(value="view")
        self.search_var = tk.StringVar()
        self.input_points: list = []
        self.input_labels: list = []
        self.input_box = None
        self.drawing_box = False
        self.box_start = (0, 0)
        self.box_end = (0, 0)
        self.poly_points: list = []
        self.scale = 1.0
        self.off_x = 0
        self.off_y = 0
        self.zoom_level = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.tk_img = None
        self._embed_token = 0
        self._destroyed = False

        self._load_csv()
        self._build_ui()
        self.search_var.trace_add("write", lambda *_: self._refresh_file_list())
        self._refresh_file_list()
        self._load_frame(0)

        self.window.bind("<Left>", lambda e: self._step_frame(-1))
        self.window.bind("<Right>", lambda e: self._step_frame(1))
        self.window.protocol("WM_DELETE_WINDOW", self._on_close)

    def _on_close(self):
        self._destroyed = True
        try:
            self.window.destroy()
        except tk.TclError:
            pass

    def _ui_alive(self) -> bool:
        if self._destroyed:
            return False
        try:
            return bool(self.window.winfo_exists())
        except tk.TclError:
            return False

    def _set_status(self, text: str):
        if not self._ui_alive():
            return

        def _apply():
            if not self._ui_alive():
                return
            try:
                self.lbl_status.config(text=text)
            except tk.TclError:
                pass

        try:
            self.window.after(0, _apply)
        except tk.TclError:
            pass

    def _load_csv(self):
        if not self.csv_path or not os.path.isfile(self.csv_path):
            return
        with open(self.csv_path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                try:
                    fid = int(float(row.get("Frame_ID", -1)))
                except (TypeError, ValueError):
                    continue
                poly = _parse_tracking_polygon(row.get("Polygon", ""))
                self.rows_by_frame[fid] = dict(row)
                self.rows_by_frame[fid]["Polygon"] = _polygon_to_csv_str(poly)
                self.original_polygons[fid] = [list(p) for p in poly]
                self.frame_status.setdefault(fid, "pending")

    def _build_ui(self):
        top = tk.Frame(self.window, bg="#34495E", pady=6)
        top.pack(side="top", fill="x")
        tk.Label(
            top,
            text="Review YOLO segmentations — mark OK, or correct with SAM / polygon",
            bg="#34495E",
            fg="white",
            font=("Arial", 11, "bold"),
        ).pack(side="left", padx=10)
        tk.Button(top, text="Save CSV", command=self._save_csv, bg="#16A085", fg="white", font=("Arial", 10, "bold")).pack(
            side="right", padx=6
        )
        tk.Button(top, text="Load SAM model", command=self._pick_sam_model, bg="#8E44AD", fg="white").pack(side="right", padx=6)
        tk.Button(top, text="Run SAM embed", command=self._run_sam_embed, bg="#F1C40F", font=("Arial", 9, "bold")).pack(
            side="right", padx=4
        )

        body = tk.Frame(self.window, bg="#2C3E50")
        body.pack(fill="both", expand=True, padx=8, pady=6)

        left = tk.Frame(body, bg="#ECF0F1", width=280)
        left.pack(side="left", fill="y", padx=(0, 8))
        left.pack_propagate(False)

        tk.Label(left, text="Frames", bg="#ECF0F1", font=("Arial", 10, "bold")).pack(anchor="w", padx=8, pady=(8, 2))
        sf = tk.Frame(left, bg="#ECF0F1")
        sf.pack(fill="x", padx=8)
        tk.Entry(sf, textvariable=self.search_var).pack(side="left", fill="x", expand=True)
        tk.Button(sf, text="Clear", command=lambda: self.search_var.set(""), width=6).pack(side="left", padx=4)

        list_wrap = tk.Frame(left, bg="#ECF0F1")
        list_wrap.pack(fill="both", expand=True, padx=8, pady=6)
        sb = tk.Scrollbar(list_wrap)
        sb.pack(side="right", fill="y")
        self.file_listbox = tk.Listbox(list_wrap, yscrollcommand=sb.set, font=("Consolas", 9), selectmode="browse")
        self.file_listbox.pack(side="left", fill="both", expand=True)
        sb.config(command=self.file_listbox.yview)
        self.file_listbox.bind("<<ListboxSelect>>", self._on_file_select)

        nav = tk.Frame(left, bg="#ECF0F1")
        nav.pack(fill="x", padx=8, pady=4)
        tk.Button(nav, text="◀ Prev", command=lambda: self._step_frame(-1)).pack(side="left", fill="x", expand=True, padx=2)
        tk.Button(nav, text="Next ▶", command=lambda: self._step_frame(1)).pack(side="left", fill="x", expand=True, padx=2)

        tk.Label(left, text="Legend: [ ] pending  [✓] OK  [!] corrected", bg="#ECF0F1", fg="#566573", font=("Arial", 8)).pack(
            anchor="w", padx=8, pady=(0, 8)
        )

        center = tk.Frame(body, bg="#1C2833")
        center.pack(side="left", fill="both", expand=True)

        tools = tk.Frame(center, bg="#566573", pady=4)
        tools.pack(fill="x")
        tk.Label(tools, text="Tool:", bg="#566573", fg="white").pack(side="left", padx=6)
        for label, val in (("View", "view"), ("SAM point", "ai"), ("SAM box", "box"), ("Polygon", "polygon")):
            tk.Radiobutton(
                tools,
                text=label,
                variable=self.mode,
                value=val,
                command=self._clear_prompts,
                bg="#566573",
                fg="white",
                selectcolor="#34495E",
                activebackground="#566573",
                activeforeground="white",
            ).pack(side="left", padx=2)
        tk.Button(tools, text="Apply correction", command=self._apply_correction, bg="#27AE60", fg="white").pack(
            side="left", padx=8
        )
        tk.Button(tools, text="Mark OK", command=self._mark_ok, bg="#2980B9", fg="white").pack(side="left", padx=2)
        tk.Button(tools, text="Revert YOLO", command=self._revert_yolo, bg="#E67E22", fg="white").pack(side="left", padx=2)
        tk.Button(tools, text="Clear preview", command=self._clear_prompts, bg="#7F8C8D", fg="white").pack(side="left", padx=2)

        self.canvas = tk.Canvas(center, bg="black", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=4, pady=4)
        self.canvas.bind("<Button-1>", self._on_left_click)
        self.canvas.bind("<ButtonRelease-1>", self._on_left_release)
        self.canvas.bind("<B1-Motion>", self._on_left_drag)
        self.canvas.bind("<Button-3>", self._on_right_click)
        self.canvas.bind("<Configure>", lambda e: self._redraw())

        self.lbl_status = tk.Label(center, text="", bg="#1C2833", fg="#AED6F1", anchor="w", font=("Arial", 9))
        self.lbl_status.pack(fill="x", padx=8, pady=(0, 6))

    def _status_symbol(self, frame_id: int) -> str:
        st = self.frame_status.get(frame_id, "pending")
        return {"pending": "[ ]", "ok": "[✓]", "corrected": "[!]"}.get(st, "[ ]")

    def _refresh_file_list(self):
        query = (self.search_var.get() or "").strip().lower()
        self._filtered_indices = []
        self.file_listbox.delete(0, tk.END)
        for i, path in enumerate(self.image_files):
            name = os.path.basename(path).lower()
            if query and query not in name and query not in f"{i:05d}":
                continue
            self._filtered_indices.append(i)
            sym = self._status_symbol(i)
            self.file_listbox.insert(tk.END, f"{sym} {i:05d}  {os.path.basename(path)}")
        cur = self.current_idx
        if cur in self._filtered_indices:
            sel = self._filtered_indices.index(cur)
            self.file_listbox.selection_clear(0, tk.END)
            self.file_listbox.selection_set(sel)
            self.file_listbox.see(sel)

    def _on_file_select(self, _event=None):
        sel = self.file_listbox.curselection()
        if not sel:
            return
        idx = self._filtered_indices[sel[0]]
        self._load_frame(idx)

    def _step_frame(self, delta: int):
        if not self.image_files:
            return
        self._load_frame(max(0, min(len(self.image_files) - 1, self.current_idx + delta)))

    def _frame_polygon(self, frame_id: int) -> list:
        row = self.rows_by_frame.get(frame_id, {})
        return _parse_tracking_polygon(row.get("Polygon", ""))

    def _load_frame(self, idx: int):
        if not self.image_files or idx < 0 or idx >= len(self.image_files):
            return
        self.current_idx = idx
        self._clear_prompts()
        path = self.image_files[idx]
        self.raw_bgr = imread_safe(path)
        if self.raw_bgr is None:
            self._set_status(f"Could not load: {path}")
            return
        self.raw_rgb = cv2.cvtColor(self.raw_bgr, cv2.COLOR_BGR2RGB)
        self.active_polygon = self._frame_polygon(idx)
        self.is_ai_ready = False
        if self.predictor is not None:
            self._embed_token += 1
            token = self._embed_token
            threading.Thread(target=lambda: self._set_sam_embedding(token), daemon=True).start()
        self._redraw()
        self._refresh_file_list()
        row = self.rows_by_frame.get(idx, {})
        conf = row.get("Confidence", "")
        self._set_status(
            f"Frame {idx + 1}/{len(self.image_files)} — {os.path.basename(path)} | "
            f"Status: {self.frame_status.get(idx, 'pending')} | "
            f"Conf: {conf} | "
            f"Vertices: {len(self.active_polygon)}"
        )

    def _set_sam_embedding(self, token=None):
        if self.predictor is None or self.raw_rgb is None:
            return
        if token is not None and token != self._embed_token:
            return
        self._set_status("SAM embedding…")
        try:
            rgb = self.raw_rgb.copy()
            self.predictor.set_image(rgb)
            if token is not None and token != self._embed_token:
                return
            self.is_ai_ready = True
            self._set_status(f"SAM ready — frame {self.current_idx + 1}")
        except Exception as exc:
            if token is not None and token != self._embed_token:
                return
            self.is_ai_ready = False
            self._set_status(f"SAM embed failed: {exc}")

    def _canvas_to_image(self, cx, cy):
        if self.scale <= 0:
            return 0, 0
        ix = int((cx - self.off_x) / self.scale)
        iy = int((cy - self.off_y) / self.scale)
        if self.raw_bgr is not None:
            h, w = self.raw_bgr.shape[:2]
            ix = max(0, min(w - 1, ix))
            iy = max(0, min(h - 1, iy))
        return ix, iy

    def _clear_prompts(self):
        self.preview_mask = None
        self.input_points = []
        self.input_labels = []
        self.input_box = None
        self.drawing_box = False
        self.poly_points = []
        self._redraw()

    def _on_left_click(self, event):
        if self.raw_bgr is None:
            return
        ix, iy = self._canvas_to_image(event.x, event.y)
        mode = self.mode.get()
        if mode == "ai":
            if not self.is_ai_ready:
                messagebox.showwarning("SAM", "Load a SAM model and run embed first.", parent=self.window)
                return
            self.input_points.append([ix, iy])
            self.input_labels.append(1)
            self._run_sam_prediction()
        elif mode == "box":
            self.drawing_box = True
            self.box_start = (ix, iy)
            self.box_end = (ix, iy)
            self._redraw()
        elif mode == "polygon":
            self.poly_points.append((ix, iy))
            self._redraw()

    def _on_left_drag(self, event):
        if self.mode.get() == "box" and self.drawing_box:
            self.box_end = self._canvas_to_image(event.x, event.y)
            self._redraw()

    def _on_left_release(self, event):
        if self.mode.get() != "box" or not self.drawing_box:
            return
        self.drawing_box = False
        ix, iy = self._canvas_to_image(event.x, event.y)
        x1, x2 = min(self.box_start[0], ix), max(self.box_start[0], ix)
        y1, y2 = min(self.box_start[1], iy), max(self.box_start[1], iy)
        if (x2 - x1) > 5 and (y2 - y1) > 5:
            if not self.is_ai_ready:
                messagebox.showwarning("SAM", "Load a SAM model and run embed first.", parent=self.window)
                return
            self.input_box = [x1, y1, x2, y2]
            self._run_sam_prediction()
        self._redraw()

    def _on_right_click(self, event):
        if self.raw_bgr is None:
            return
        ix, iy = self._canvas_to_image(event.x, event.y)
        if self.mode.get() == "ai" and self.is_ai_ready:
            self.input_points.append([ix, iy])
            self.input_labels.append(0)
            self._run_sam_prediction()
        elif self.mode.get() == "polygon" and len(self.poly_points) > 2:
            h, w = self.raw_bgr.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [np.array(self.poly_points, np.int32)], 1)
            self.preview_mask = mask.astype(bool)
            self.poly_points = []
            self._redraw()

    def _run_sam_prediction(self):
        if self.predictor is None:
            return
        try:
            kwargs = {"multimask_output": False}
            if self.input_points:
                kwargs["point_coords"] = np.array(self.input_points)
                kwargs["point_labels"] = np.array(self.input_labels)
            if self.input_box is not None:
                kwargs["box"] = np.array(self.input_box, dtype=np.float32)
            if not self.input_points and self.input_box is None:
                return
            masks, _, _ = self.predictor.predict(**kwargs)
            self.preview_mask = masks[0].astype(bool)
            self._redraw()
        except Exception as exc:
            messagebox.showerror("SAM", str(exc), parent=self.window)

    def _compose_display_bgr(self):
        if self.raw_bgr is None:
            return None
        out = self.raw_bgr.copy()
        h, w = out.shape[:2]
        st = self.frame_status.get(self.current_idx, "pending")

        # Hide original YOLO overlay once a correction has been applied.
        if st != "corrected":
            yolo_poly = self.original_polygons.get(self.current_idx, [])
            if yolo_poly:
                pts = np.array(yolo_poly, dtype=np.int32)
                if len(pts) >= 3:
                    overlay = out.copy()
                    cv2.fillPoly(overlay, [pts], (180, 140, 40))
                    cv2.addWeighted(overlay, 0.25, out, 0.75, 0, out)
                    cv2.polylines(out, [pts], True, (200, 160, 60), 1, cv2.LINE_AA)

        if self.preview_mask is not None:
            overlay = out.copy()
            overlay[self.preview_mask] = (255, 120, 0)
            cv2.addWeighted(overlay, 0.45, out, 0.55, 0, out)

        poly = self.active_polygon
        if poly:
            pts = np.array(poly, dtype=np.int32)
            if len(pts) >= 3:
                overlay = out.copy()
                cv2.fillPoly(overlay, [pts], (0, 200, 80))
                cv2.addWeighted(overlay, 0.35, out, 0.65, 0, out)
                cv2.polylines(out, [pts], True, (0, 255, 120), 2, cv2.LINE_AA)

        if self.mode.get() == "polygon" and self.poly_points:
            for i, (px, py) in enumerate(self.poly_points):
                cv2.circle(out, (int(px), int(py)), 4, (0, 255, 255), -1)
                if i > 0:
                    p0 = self.poly_points[i - 1]
                    cv2.line(out, (int(p0[0]), int(p0[1])), (int(px), int(py)), (0, 255, 255), 2)

        if self.mode.get() == "box":
            if self.input_box is not None:
                x1, y1, x2, y2 = [int(v) for v in self.input_box]
                cv2.rectangle(out, (x1, y1), (x2, y2), (243, 156, 18), 2)
            if self.drawing_box:
                x1, x2 = min(self.box_start[0], self.box_end[0]), max(self.box_start[0], self.box_end[0])
                y1, y2 = min(self.box_start[1], self.box_end[1]), max(self.box_start[1], self.box_end[1])
                cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 2)

        if self.mode.get() == "ai" and self.input_points:
            for i, (px, py) in enumerate(self.input_points):
                col = (0, 255, 0) if self.input_labels[i] == 1 else (0, 0, 255)
                cv2.circle(out, (int(px), int(py)), 5, col, -1)

        tag = self.instance_name or "tracked"
        legend = (
            f"{tag} | corrected (YOLO hidden)"
            if st == "corrected"
            else f"{tag} | {st} | YOLO=dim  Active=green  Preview=orange"
        )
        cv2.putText(
            out,
            legend,
            (10, max(24, h // 40)),
            cv2.FONT_HERSHEY_SIMPLEX,
            max(0.5, min(h, w) / 900.0),
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return out

    def _redraw(self):
        disp_bgr = self._compose_display_bgr()
        if disp_bgr is None:
            return
        rgb = cv2.cvtColor(disp_bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        cw = max(self.canvas.winfo_width(), 400)
        ch = max(self.canvas.winfo_height(), 300)
        self.scale = min(cw / w, ch / h) * self.zoom_level
        nw, nh = max(1, int(w * self.scale)), max(1, int(h * self.scale))
        pil = Image.fromarray(rgb).resize((nw, nh), Image.Resampling.LANCZOS)
        self.off_x = (cw - nw) // 2 + self.pan_x
        self.off_y = (ch - nh) // 2 + self.pan_y
        self.tk_img = ImageTk.PhotoImage(pil)
        self.canvas.delete("all")
        self.canvas.create_image(self.off_x, self.off_y, anchor="nw", image=self.tk_img)

    def _apply_correction(self):
        poly = []
        if self.preview_mask is not None:
            poly = _mask_to_polygon(self.preview_mask)
        elif len(self.poly_points) >= 3:
            poly = [[float(x), float(y)] for x, y in self.poly_points]
        if len(poly) < 3:
            messagebox.showwarning(
                "Correction",
                "Draw a SAM mask or close a polygon (right-click), then Apply.",
                parent=self.window,
            )
            return
        fid = self.current_idx
        if fid not in self.rows_by_frame:
            self.rows_by_frame[fid] = {"Frame_ID": str(fid), "Instance_Name": self.instance_name}
        self.rows_by_frame[fid]["Polygon"] = _polygon_to_csv_str(poly)
        self.active_polygon = poly
        self.frame_status[fid] = "corrected"
        self._clear_prompts()
        self._refresh_file_list()
        self._redraw()
        self._set_status(f"Applied correction on frame {fid + 1} ({len(poly)} vertices) — YOLO mask hidden")

    def _mark_ok(self):
        fid = self.current_idx
        self.frame_status[fid] = "ok"
        self._refresh_file_list()
        self._set_status(f"Marked frame {fid + 1} as OK")

    def _revert_yolo(self):
        fid = self.current_idx
        orig = self.original_polygons.get(fid, [])
        if fid in self.rows_by_frame:
            self.rows_by_frame[fid]["Polygon"] = _polygon_to_csv_str(orig)
        self.active_polygon = [list(p) for p in orig]
        self.frame_status[fid] = "pending"
        self._clear_prompts()
        self._refresh_file_list()
        self._redraw()

    def _save_csv(self):
        if not self.csv_path:
            return
        fieldnames = [
            "Frame_ID",
            "Instance_Name",
            "Track_ID",
            "YOLO_Class_ID",
            "YOLO_Model_Name",
            "Center_X",
            "Center_Y",
            "Width",
            "Height",
            "Confidence",
            "Polygon",
        ]
        try:
            with open(self.csv_path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                for i in range(len(self.image_files)):
                    row = dict(self.rows_by_frame.get(i, {}))
                    row["Frame_ID"] = i
                    if "Instance_Name" not in row or not row["Instance_Name"]:
                        row["Instance_Name"] = self.instance_name
                    if "Polygon" not in row:
                        row["Polygon"] = ""
                    writer.writerow(row)
            messagebox.showinfo("Saved", f"Updated tracking CSV:\n{self.csv_path}", parent=self.window)
        except OSError as exc:
            messagebox.showerror("Save failed", str(exc), parent=self.window)

    def _pick_sam_model(self):
        if not HAS_SAM:
            messagebox.showerror(
                "SAM",
                "SAM not installed.\npip install torch segment-anything sam2",
                parent=self.window,
            )
            return
        popup = tk.Toplevel(self.window)
        popup.title("Select SAM model")
        popup.geometry("480x380")
        popup.transient(self.window)
        tk.Label(popup, text="Same models folder as the segmenter (../models)", font=("Arial", 10, "bold")).pack(
            pady=8
        )
        lf = tk.Frame(popup)
        lf.pack(fill="both", expand=True, padx=12, pady=4)
        sb = tk.Scrollbar(lf)
        sb.pack(side="right", fill="y")
        lb = tk.Listbox(lf, yscrollcommand=sb.set, font=("Consolas", 10))
        lb.pack(side="left", fill="both", expand=True)
        sb.config(command=lb.yview)
        candidates = _sam_discover_checkpoints()
        for p in candidates:
            lb.insert(tk.END, os.path.basename(p))
            bname = os.path.basename(p).lower()
            if "sam2" in bname:
                lb.itemconfig(tk.END, {"fg": "blue"})
            elif "vit" in bname:
                lb.itemconfig(tk.END, {"fg": "darkgreen"})

        def _load_selected(path=None):
            if path is None:
                sel = lb.curselection()
                if not sel:
                    return
                fname = lb.get(sel[0])
                path = next((p for p in candidates if os.path.basename(p) == fname), None)
            if not path:
                return
            popup.destroy()
            self._set_status(f"Loading {os.path.basename(path)}…")

            def _worker():
                try:
                    predictor, is_sam2 = _sam_init_predictor(path, self.device)
                    if not self._ui_alive():
                        return
                    self.predictor = predictor
                    self.is_sam2 = is_sam2
                    self.is_ai_ready = False
                    if self.raw_rgb is not None:
                        self.predictor.set_image(self.raw_rgb.copy())
                        self.is_ai_ready = True
                    name = os.path.basename(path)
                    frame_no = self.current_idx + 1
                    self._set_status(f"SAM loaded: {name} — frame {frame_no}")
                except Exception as exc:
                    err = str(exc)

                    def _show_err(msg=err):
                        if self._ui_alive():
                            messagebox.showerror("SAM load failed", msg, parent=self.window)

                    if self._ui_alive():
                        self.window.after(0, _show_err)

            threading.Thread(target=_worker, daemon=True).start()

        def _browse():
            script_dir = os.path.dirname(os.path.abspath(__file__))
            start = os.path.join(os.path.dirname(script_dir), "models")
            path = filedialog.askopenfilename(
                parent=popup,
                initialdir=start if os.path.isdir(start) else os.getcwd(),
                filetypes=[("Model", "*.pth *.pt"), ("All", "*.*")],
            )
            if path:
                _load_selected(path)

        bf = tk.Frame(popup)
        bf.pack(fill="x", padx=12, pady=10)
        tk.Button(bf, text="Browse…", command=_browse).pack(side="left")
        tk.Button(bf, text="Load selected", command=lambda: _load_selected(), bg="#27AE60", fg="white").pack(side="right")

    def _run_sam_embed(self):
        if self.predictor is None:
            messagebox.showwarning("SAM", "Load a SAM model first.", parent=self.window)
            return
        self._embed_token += 1
        token = self._embed_token
        threading.Thread(target=lambda: self._set_sam_embedding(token), daemon=True).start()


# ==============================================================================
# YOLO TRAINING SUMMARY (metrics graphs only — no dataset preview images)
# ==============================================================================
_YOLO_TRAIN_CLASS_DEFAULT = "Keyhole"

_YOLO_SUMMARY_PLOT_ORDER = [
    ("Training curves (loss & mAP vs epoch)", "results.png"),
    ("Confusion matrix (normalized)", "confusion_matrix_normalized.png"),
    ("Confusion matrix", "confusion_matrix.png"),
    ("Mask precision–recall", "MaskPR_curve.png"),
]

_YOLO_SUMMARY_CHART_NAMES = frozenset(
    n.lower()
    for n in (
        "results.png",
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "MaskPR_curve.png",
    )
)

_YOLO_SUMMARY_PLOT_MAX_WIDTH = {
    "results.png": 1200,
    "confusion_matrix_normalized.png": 900,
    "confusion_matrix.png": 900,
    "maskpr_curve.png": 800,
}


def _yolo_summary_plot_max_width(filename: str) -> int:
    return _YOLO_SUMMARY_PLOT_MAX_WIDTH.get((filename or "").lower(), 800)


def _yolo_summary_chart_allowed(filename: str) -> bool:
    """Only metric/chart PNGs — never train/val batch preview JPEGs."""
    n = (filename or "").lower()
    if not n.endswith(".png"):
        return False
    return n in _YOLO_SUMMARY_CHART_NAMES


_YOLO_SUMMARY_METRIC_COLS = (
    "metrics/mAP50(M)",
    "metrics/mAP50-95(M)",
    "metrics/precision(M)",
    "metrics/recall(M)",
)


def _yolo_collect_training_plots(results_dir) -> list:
    root = Path(results_dir)
    found, seen = [], set()
    for caption, name in _YOLO_SUMMARY_PLOT_ORDER:
        if not _yolo_summary_chart_allowed(name):
            continue
        path = root / name
        if path.is_file():
            found.append((caption, path))
            seen.add(name.lower())
    return found


def _yolo_read_train_args(results_dir: Path) -> dict:
    args_path = results_dir / "args.yaml"
    if not args_path.is_file():
        return {}
    try:
        with open(args_path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _yolo_format_class_names(names) -> str:
    if isinstance(names, dict):
        parts = [f"{k}: {v}" for k, v in sorted(names.items(), key=lambda x: int(x[0]) if str(x[0]).isdigit() else str(x[0]))]
        return ", ".join(parts) if parts else str(names)
    if isinstance(names, (list, tuple)):
        return ", ".join(str(n) for n in names)
    return str(names) if names else "—"


def _yolo_primary_class_label(names) -> str:
    if isinstance(names, dict) and names:
        if len(names) == 1:
            return str(next(iter(names.values())))
        return _yolo_format_class_names(names)
    if isinstance(names, (list, tuple)) and names:
        return str(names[0])
    return str(names).strip() if names else _YOLO_TRAIN_CLASS_DEFAULT


def _yolo_build_training_summary(results_dir, best_pt, config: dict | None = None) -> str:
    config = config or {}
    results_dir = Path(results_dir)
    best_pt = Path(best_pt) if best_pt else results_dir / "weights" / "best.pt"
    args = _yolo_read_train_args(results_dir)
    class_names = config.get("class_names") or args.get("names") or {0: _YOLO_TRAIN_CLASS_DEFAULT}
    class_label = _yolo_primary_class_label(class_names)
    lines = [f"═══ YOLO Training Summary — {class_label} ═══", ""]

    lines.append("— Model weights —")
    if best_pt.is_file():
        stat = best_pt.stat()
        lines.append(f"  best.pt:     {best_pt}")
        lines.append(f"  File size:   {stat.st_size / 1e6:.2f} MB")
        try:
            from datetime import datetime
            lines.append(f"  Modified:    {datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M')}")
        except Exception:
            pass
    else:
        lines.append("  best.pt:     (not found)")

    lines.extend(["", "— Training configuration —"])
    ver_label = config.get("yolo_version_label") or args.get("model", "—")
    lines.append(f"  YOLO version:  {ver_label}")
    lines.append(f"  Backbone:      {config.get('backbone') or args.get('model', '—')}")
    lines.append(f"  Task:          {config.get('task') or args.get('task', 'segment')}")
    lines.append(f"  Epochs:        {config.get('epochs') or args.get('epochs', '—')}")
    lines.append(f"  Batch size:    {config.get('batch') or args.get('batch', '—')}")
    lines.append(f"  Image size:    {config.get('imgsz') or args.get('imgsz', '—')}")
    if config.get("training_time_sec") is not None:
        sec = float(config["training_time_sec"])
        lines.append(f"  Training time: {sec:.1f} s ({sec / 60:.1f} min)")
    if config.get("output_dir"):
        lines.append(f"  Results dir:   {config['output_dir']}")

    lines.extend(["", "— Dataset —"])
    lines.append(f"  Train images:  {config.get('train_count', '—')}")
    lines.append(f"  Val images:    {config.get('val_count', '—')}")
    lines.append(f"  Trained class: {_yolo_format_class_names(class_names)}")
    lines.append("  Note: Background is implicit (unlabeled pixels). No background class is required.")

    csv_path = results_dir / "results.csv"
    if csv_path.is_file():
        try:
            import pandas as pd
            df = pd.read_csv(csv_path)
            df.columns = df.columns.str.strip()
            row = df.iloc[-1]
            epoch_no = int(row["epoch"]) if "epoch" in row.index else len(df)
            lines.extend(["", f"— Final metrics (epoch {epoch_no}) —"])
            for col in _YOLO_SUMMARY_METRIC_COLS:
                if col in row.index:
                    try:
                        lines.append(f"  {col}: {float(row[col]):.4f}")
                    except (TypeError, ValueError):
                        pass
        except Exception as exc:
            lines.append(f"  (Could not read results.csv: {exc})")
    else:
        lines.extend(["", "— Metrics —", "  results.csv not found in this folder."])

    lines.extend([
        "",
        "— Graphs included —",
        "  • results.png — train/val loss and mask mAP vs training epoch",
        "  • confusion_matrix* — classification errors on the validation split",
        "  • MaskPR_curve — combined precision/recall vs confidence (P & R curves omitted as redundant)",
        "",
        "No train/val batch preview images are shown in this summary.",
        f"Record a demo video in section 5 (Inference & Tracking) with best.pt ({class_label}).",
    ])
    return "\n".join(lines)


class YoloTrainingSummaryWindow:
    """Scrollable report: text metrics + Ultralytics training plot images."""

    def __init__(self, parent, results_dir, best_pt, summary_text, plot_paths, class_label=""):
        self.win = tk.Toplevel(parent)
        class_label = (class_label or "").strip()
        self.win.title(f"Training Summary — {class_label}" if class_label else "YOLO Training Summary")
        self.win.geometry("1320x920")
        self.win.configure(bg="#F4F6F6")
        self._photos = []

        top = tk.Frame(self.win, bg="#1F618D", pady=8)
        top.pack(fill="x")
        header = (
            f"Training summary — {class_label} (metrics & graphs)"
            if class_label
            else "Training summary — metrics & graphs"
        )
        tk.Label(
            top,
            text=header,
            bg="#1F618D",
            fg="white",
            font=("Arial", 13, "bold"),
        ).pack(side="left", padx=14)
        tk.Button(top, text="Close", command=self.win.destroy, bg="#154360", fg="white").pack(side="right", padx=12)

        outer = tk.Frame(self.win)
        outer.pack(fill="both", expand=True, padx=10, pady=8)
        canvas = tk.Canvas(outer, highlightthickness=0, bg="#F4F6F6")
        scrollbar = tk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        inner = tk.Frame(canvas, bg="#F4F6F6")
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        win_id = canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(win_id, width=e.width))
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        def _wheel(event):
            if event.num == 4 or getattr(event, "delta", 0) > 0:
                canvas.yview_scroll(-1, "units")
            elif event.num == 5 or getattr(event, "delta", 0) < 0:
                canvas.yview_scroll(1, "units")

        canvas.bind_all("<MouseWheel>", _wheel, add="+")
        canvas.bind_all("<Button-4>", _wheel, add="+")
        canvas.bind_all("<Button-5>", _wheel, add="+")

        def _close_summary():
            try:
                canvas.unbind_all("<MouseWheel>")
                canvas.unbind_all("<Button-4>")
                canvas.unbind_all("<Button-5>")
            except tk.TclError:
                pass
            self.win.destroy()

        self.win.protocol("WM_DELETE_WINDOW", _close_summary)

        info = tk.LabelFrame(inner, text="Overview", font=("Arial", 11, "bold"), bg="#F4F6F6")
        info.pack(fill="x", padx=4, pady=4)
        txt = scrolledtext.ScrolledText(info, height=16, font=("Consolas", 9), wrap="word", bg="#FBFCFC")
        txt.pack(fill="x", padx=8, pady=8)
        txt.insert("1.0", summary_text)
        txt.config(state="disabled")

        plots_frame = tk.LabelFrame(inner, text="Training quality graphs", font=("Arial", 11, "bold"), bg="#F4F6F6")
        plots_frame.pack(fill="x", padx=4, pady=8)

        if not plot_paths:
            tk.Label(
                plots_frame,
                text="No plot images found in the results folder.",
                bg="#F4F6F6",
                fg="#7B7D7D",
                font=("Arial", 10, "italic"),
            ).pack(padx=12, pady=12)
        else:
            for caption, path in plot_paths:
                block = tk.Frame(plots_frame, bg="#F4F6F6")
                block.pack(fill="x", padx=8, pady=10)
                tk.Label(block, text=caption, bg="#F4F6F6", font=("Arial", 10, "bold"), anchor="w").pack(anchor="w")
                tk.Label(block, text=str(path.name), bg="#F4F6F6", fg="gray", font=("Arial", 8)).pack(anchor="w")
                try:
                    pil = Image.open(path)
                    w, h = pil.size
                    max_w = _yolo_summary_plot_max_width(path.name)
                    if path.name.lower() == "results.png" and w < max_w:
                        scale = min(max_w / w, 1.35)
                        if scale > 1.0:
                            pil = pil.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
                    elif w > max_w:
                        scale = max_w / w
                        pil = pil.resize((max_w, int(h * scale)), Image.Resampling.LANCZOS)
                    photo = ImageTk.PhotoImage(pil)
                    self._photos.append(photo)
                    tk.Label(block, image=photo, bg="#F4F6F6", bd=1, relief="solid").pack(pady=4, anchor="w")
                except Exception as exc:
                    tk.Label(block, text=f"(Could not load image: {exc})", fg="#C0392B", bg="#F4F6F6").pack(anchor="w")


def _yolo_trainer_scrollable(parent):
    """Canvas + vertical scrollbar for YoloTrainerApp main content."""
    try:
        bg = parent.cget("bg")
    except tk.TclError:
        bg = "#F5EEF8"
    canvas = tk.Canvas(parent, highlightthickness=0, bg=bg)
    scrollbar = tk.Scrollbar(parent, orient="vertical", command=canvas.yview)
    inner = tk.Frame(canvas, bg=bg)
    inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    win_id = canvas.create_window((0, 0), window=inner, anchor="nw")

    def _on_canvas_configure(event):
        canvas.itemconfig(win_id, width=event.width)

    canvas.bind("<Configure>", _on_canvas_configure)
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    def _on_mousewheel(event):
        if event.num == 4 or getattr(event, "delta", 0) > 0:
            canvas.yview_scroll(-1, "units")
        elif event.num == 5 or getattr(event, "delta", 0) < 0:
            canvas.yview_scroll(1, "units")

    def _bind(_e):
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        canvas.bind_all("<Button-4>", _on_mousewheel)
        canvas.bind_all("<Button-5>", _on_mousewheel)

    def _unbind(_e):
        canvas.unbind_all("<MouseWheel>")
        canvas.unbind_all("<Button-4>")
        canvas.unbind_all("<Button-5>")

    canvas.bind("<Enter>", _bind)
    canvas.bind("<Leave>", _unbind)
    return inner, canvas


# ==============================================================================
# YOLO TRAINER APP
# ==============================================================================
class YoloTrainerApp:
    def __init__(self, parent_frame, on_back):
        utils.init_ml_dependencies()
        _sync_ml_flags_from_utils()

        self.frame = tk.Frame(parent_frame)
        self.frame.pack(fill="both", expand=True)
        self.on_back = on_back
        
        self.source_folders = [] 
        self.out_path = tk.StringVar()
        self.epochs = tk.IntVar(value=50)
        self.batch = tk.IntVar(value=16)
        self.img_size = tk.IntVar(value=640)
        self.yolo_version = tk.StringVar(value="v8")
        self.train_class_name = tk.StringVar(value=_YOLO_TRAIN_CLASS_DEFAULT)
        
        self.is_training = False
        self.training_results_dir = None
        self.last_best_pt = None
        self.training_summary_text = None
        self.training_plot_paths = []
        self._summary_window = None
        self.training_class_label = ""
        self.setup_ui()
        if not HAS_YOLO:
            messagebox.showerror("Error", "Ultralytics YOLO not found.\nRun: pip install ultralytics")

    def setup_ui(self):
        top = tk.Frame(self.frame, bg="#F5EEF8", height=60)
        top.pack(fill="x")
        tk.Button(top, text="← Back", command=self.go_back, bg="#95A5A6", fg="white", font=("Arial", 11)).pack(side="left", padx=20)
        tk.Label(top, text="Tool: YOLOv8 / YOLOv12 Trainer & Tracker", bg="#F5EEF8", fg="#8E44AD", font=("Arial", 18, "bold")).pack(side="left", padx=20)
        tk.Button(
            top,
            text="YOLO demo →",
            command=lambda: launch_yolo_demonstration(self.frame),
            bg="#2E86C1",
            fg="white",
            font=("Arial", 10, "bold"),
            cursor="hand2",
        ).pack(side="right", padx=20, pady=10)

        scroll_host = tk.Frame(self.frame, bg="#F5EEF8")
        scroll_host.pack(fill="both", expand=True)
        scroll_inner, self._scroll_canvas = _yolo_trainer_scrollable(scroll_host)
        content = tk.Frame(scroll_inner, bg=scroll_inner.cget("bg"))
        content.pack(fill="both", expand=True, padx=50, pady=20)

        # 1. Dataset Selection
        f_data = tk.LabelFrame(content, text="1. Source Datasets (Add 'Input' folders)", font=("Arial", 11, "bold"), fg="#2C3E50")
        f_data.pack(fill="x", pady=5)
        
        self.folder_listbox = tk.Listbox(f_data, height=3)
        self.folder_listbox.pack(side="left", fill="x", expand=True, padx=10, pady=5)
        self._section_min_h = max(self.folder_listbox.winfo_reqheight() + 16, 88)
        
        btn_frame = tk.Frame(f_data)
        btn_frame.pack(side="right", padx=10)
        tk.Button(btn_frame, text="Add Folder...", command=self.add_folder).pack(fill="x", pady=2)
        tk.Button(btn_frame, text="Clear List", command=self.clear_folders).pack(fill="x", pady=2)

        # 2. Output & Params
        f_param = tk.LabelFrame(content, text="2. Configuration", font=("Arial", 11, "bold"), fg="#2C3E50")
        f_param.pack(fill="x", pady=5)
        
        grid = {'padx': 10, 'pady': 5, 'sticky': 'w'}
        tk.Label(f_param, text="Output Directory:").grid(row=0, column=0, **grid)
        tk.Entry(f_param, textvariable=self.out_path, width=50).grid(row=0, column=1, **grid)
        tk.Button(f_param, text="Browse", command=lambda: self.browse(self.out_path)).grid(row=0, column=2, **grid)
        
        h_frame = tk.Frame(f_param)
        h_frame.grid(row=1, column=0, columnspan=3, sticky="w", pady=5)
        
        tk.Label(h_frame, text="Epochs:").pack(side="left", padx=5)
        tk.Entry(h_frame, textvariable=self.epochs, width=5).pack(side="left")
        
        tk.Label(h_frame, text="Batch:").pack(side="left", padx=15)
        tk.Entry(h_frame, textvariable=self.batch, width=5).pack(side="left")
        
        tk.Label(h_frame, text="Img Size:").pack(side="left", padx=15)
        tk.Entry(h_frame, textvariable=self.img_size, width=6).pack(side="left")
        tk.Label(h_frame, text="(Try 1024+ for small pores)", fg="gray", font=("Arial", 8)).pack(side="left", padx=5)

        yolo_row = tk.Frame(f_param)
        yolo_row.grid(row=2, column=0, columnspan=3, sticky="w", pady=(8, 2))
        tk.Label(yolo_row, text="YOLO version:", font=("Arial", 10, "bold")).pack(side="left", padx=5)
        tk.Radiobutton(
            yolo_row,
            text="YOLOv8 (stable)",
            variable=self.yolo_version,
            value="v8",
            font=("Arial", 10),
        ).pack(side="left", padx=8)
        tk.Radiobutton(
            yolo_row,
            text="YOLOv12 (newer)",
            variable=self.yolo_version,
            value="v12",
            font=("Arial", 10),
        ).pack(side="left", padx=8)
        tk.Label(
            yolo_row,
            text="Training backbone: yolov8n-seg.pt (v8) or yolo12n-seg.yaml (v12). For v12 tracking, load trained best.pt.",
            fg="gray",
            font=("Arial", 8),
        ).pack(side="left", padx=6)

        class_row = tk.Frame(f_param)
        class_row.grid(row=3, column=0, columnspan=3, sticky="w", pady=(6, 2))
        tk.Label(class_row, text="Object name (class 0):", font=("Arial", 10, "bold")).pack(side="left", padx=5)
        tk.Entry(class_row, textvariable=self.train_class_name, width=18, font=("Arial", 10)).pack(side="left", padx=4)
        for preset in ("Keyhole", "Tool pin", "KH-Pore", "Gas pore"):
            tk.Button(
                class_row,
                text=preset,
                command=lambda p=preset: self.train_class_name.set(p),
                bg="#D5DBDB",
                font=("Arial", 8),
                padx=3,
            ).pack(side="left", padx=2)
        tk.Label(
            class_row,
            text="Used in data.yaml, best.pt labels, and section 4 summary.",
            fg="gray",
            font=("Arial", 8),
        ).pack(side="left", padx=(8, 0))

        # 3. Training
        f_train = tk.LabelFrame(content, text="3. Training", font=("Arial", 11, "bold"), fg="#2C3E50")
        f_train.pack(fill="both", expand=True, pady=5)

        self.btn_train = tk.Button(
            f_train,
            text="MERGE DATASETS & START TRAINING",
            bg="#8E44AD",
            fg="white",
            font=("Arial", 14, "bold"),
            height=2,
            command=self.start_training,
        )
        self.btn_train.pack(fill="x", padx=10, pady=10)

        self.log_area = scrolledtext.ScrolledText(f_train, height=6, bg="#F4F6F6")
        self.log_area.pack(fill="both", expand=True, padx=10, pady=(0, 6))

        self.lbl_stage = tk.Label(
            f_train,
            text="Status: Idle",
            bg="#D5D8DC",
            fg="#2C3E50",
            font=("Arial", 14, "bold"),
            height=2,
        )
        self.lbl_stage.pack(fill="x", padx=10, pady=(0, 10))

        # 4. Training summary (best.pt report + Ultralytics plots)
        f_summary = tk.LabelFrame(
            content,
            text="4. Training Summary (graphs & metrics only)",
            font=("Arial", 11, "bold"),
            fg="#9A7D0A",
            bg="#FEF9E7",
        )
        f_summary.pack(fill="x", pady=5)
        summary_body = tk.Frame(f_summary, bg="#FEF9E7", height=self._section_min_h)
        summary_body.pack(fill="x", padx=10, pady=5)
        summary_body.pack_propagate(False)
        self.lbl_summary_status = tk.Label(
            summary_body,
            text="After training, view metrics and graphs for the object name you set above.",
            bg="#FEF9E7",
            fg="#7D6608",
            font=("Arial", 9),
            wraplength=720,
            justify="left",
        )
        self.lbl_summary_status.pack(anchor="w", padx=4, pady=(10, 6))
        summary_btns = tk.Frame(summary_body, bg="#FEF9E7")
        summary_btns.pack(anchor="w", padx=4, pady=(0, 10))
        self.btn_view_summary = tk.Button(
            summary_btns,
            text="View Training Summary",
            command=self.open_training_summary,
            bg="#D4AC0D",
            fg="white",
            font=("Arial", 11, "bold"),
            state="disabled",
        )
        self.btn_view_summary.pack(side="left", padx=(0, 8))
        tk.Button(
            summary_btns,
            text="Load results folder…",
            command=self.load_training_results_folder,
            bg="#F7DC6F",
            fg="#4A4020",
            font=("Arial", 10),
        ).pack(side="left")

        # 5. Inference & tracking (below summary; same vertical space as section 1)
        f_infer = tk.LabelFrame(
            content,
            text="5. Inference & Tracking (YOLOv8 / YOLOv12)",
            font=("Arial", 11, "bold"),
            fg="#2E86C1",
            bg="#EBF5FB",
        )
        f_infer.pack(fill="x", pady=5)
        infer_body = tk.Frame(f_infer, bg="#EBF5FB", height=self._section_min_h)
        infer_body.pack(fill="x", padx=10, pady=5)
        infer_body.pack_propagate(False)
        tk.Label(
            infer_body,
            text="Run YOLO track() on an image sequence after training (load best.pt in the tracking tool).",
            bg="#EBF5FB",
            fg="#1A5276",
            font=("Arial", 9),
            wraplength=720,
            justify="left",
        ).pack(anchor="w", padx=4, pady=(10, 6))
        tk.Button(
            infer_body,
            text="Start Tracking Tool",
            command=self.open_tracking_tool,
            bg="#2E86C1",
            fg="white",
            font=("Arial", 12, "bold"),
        ).pack(pady=(0, 10))

        content.after_idle(lambda: self._scroll_canvas.configure(scrollregion=self._scroll_canvas.bbox("all")))

    def go_back(self):
        try:
            if hasattr(self, "_scroll_canvas") and self._scroll_canvas.winfo_exists():
                self._scroll_canvas.unbind_all("<MouseWheel>")
                self._scroll_canvas.unbind_all("<Button-4>")
                self._scroll_canvas.unbind_all("<Button-5>")
        except tk.TclError:
            pass
        self.frame.destroy()
        self.on_back()

    def browse(self, var):
        path = filedialog.askdirectory()
        if path: var.set(path)
        
    def add_folder(self):
        path = filedialog.askdirectory(title="Select Folder containing images & labels")
        if path:
            self.source_folders.append(path)
            self.folder_listbox.insert(tk.END, path)

    def clear_folders(self):
        self.source_folders = []
        self.folder_listbox.delete(0, tk.END)

    def _trainer_class_name(self) -> str:
        name = (self.train_class_name.get() or _YOLO_TRAIN_CLASS_DEFAULT).strip()
        return name or _YOLO_TRAIN_CLASS_DEFAULT

    def log(self, msg):
        self.log_area.insert(tk.END, msg + "\n")
        self.log_area.see(tk.END)

    def _apply_training_summary(self, results_dir, best_pt, summary_text, plot_paths, class_label="", auto_open=False):
        self.training_results_dir = str(results_dir)
        self.last_best_pt = str(best_pt) if best_pt else ""
        self.training_summary_text = summary_text
        self.training_plot_paths = plot_paths
        self.training_class_label = (class_label or self._trainer_class_name()).strip()
        best_name = Path(best_pt).name if best_pt and Path(best_pt).is_file() else "results"
        self.lbl_summary_status.config(
            text=f"Summary ready — {self.training_class_label} | {best_name} | {len(plot_paths)} graph(s).",
            fg="#196F3D",
        )
        self.btn_view_summary.config(state="normal")
        try:
            Path(results_dir).joinpath("training_summary.txt").write_text(summary_text, encoding="utf-8")
        except OSError:
            pass
        if auto_open:
            self.open_training_summary()

    def load_training_results_folder(self):
        start = self.out_path.get() or os.getcwd()
        default = os.path.join(start, "training_results")
        folder = filedialog.askdirectory(
            title="Select Ultralytics training results folder",
            initialdir=default if os.path.isdir(default) else start,
        )
        if not folder:
            return
        results_dir = Path(folder)
        best_pt = results_dir / "weights" / "best.pt"
        if not best_pt.is_file():
            messagebox.showwarning(
                "Missing weights",
                "This folder has no weights/best.pt.\n\n"
                "Pick the folder that contains weights/, results.csv, and results.png "
                "(usually …/training_results).",
                parent=self.frame,
            )
            return
        summary_path = results_dir / "training_summary.txt"
        if summary_path.is_file():
            summary_text = summary_path.read_text(encoding="utf-8")
        else:
            summary_text = _yolo_build_training_summary(results_dir, best_pt, {"output_dir": str(results_dir)})
        plots = _yolo_collect_training_plots(results_dir)
        class_label = _yolo_primary_class_label(_yolo_read_train_args(results_dir).get("names"))
        self._apply_training_summary(results_dir, best_pt, summary_text, plots, class_label=class_label, auto_open=True)

    def open_training_summary(self):
        if not self.training_summary_text:
            messagebox.showinfo(
                "No summary",
                "Train a model in section 3 first, or use Load results folder…",
                parent=self.frame,
            )
            return
        if self._summary_window is not None:
            try:
                if self._summary_window.win.winfo_exists():
                    self._summary_window.win.lift()
                    self._summary_window.win.focus_force()
                    return
            except tk.TclError:
                pass
        root = self.frame.winfo_toplevel()
        self._summary_window = YoloTrainingSummaryWindow(
            root,
            self.training_results_dir,
            self.last_best_pt,
            self.training_summary_text,
            self.training_plot_paths,
            class_label=getattr(self, "training_class_label", "") or self._trainer_class_name(),
        )

    def start_training(self):
        if self.is_training: return
        if not self.source_folders:
            messagebox.showwarning("Missing Data", "Please add at least one source dataset folder.")
            return
        if not self.out_path.get():
            messagebox.showwarning("Missing Output", "Please select an output directory.")
            return
        if not self._trainer_class_name():
            messagebox.showwarning("Object name", "Enter the object name for class 0 (e.g. Tool pin, Keyhole).")
            return

        self.is_training = True
        self.btn_train.config(state='disabled', text="Training in Progress...", bg="gray")
        self.lbl_stage.config(text="Status: Preparing Workspace...", bg="#F39C12") 
        threading.Thread(target=self.run_training_logic, daemon=True).start()

    def on_train_epoch_end(self, trainer):
        current_epoch = trainer.epoch + 1
        total_epochs = trainer.epochs
        label = getattr(self, "_active_train_class", self._trainer_class_name())
        status_msg = f"Status: Training {label} — Epoch {current_epoch}/{total_epochs}"
        self.frame.after(0, lambda: self.lbl_stage.config(text=status_msg, bg="#5DADE2"))

    def run_training_logic(self):
        temp_dir = None
        ver = self.yolo_version.get().strip().lower()
        if ver not in ("v8", "v12"):
            ver = "v8"
        backbone = _yolo_trainer_default_seg_checkpoint(ver)
        ver_name = _yolo_trainer_version_label(ver)
        class_name = self._trainer_class_name()
        self._active_train_class = class_name
        try:
            self.log(f"--- YOLO training: {ver_name} (backbone {backbone}) | class 0: {class_name} ---")
            self.log("--- Step 1: Creating Safe Workspace ---")
            
            # 1. CREATE SAFE TEMP DIRECTORY
            temp_dir = Path.home() / "yolo_temp_fix"
            
            if temp_dir.exists():
                try: shutil.rmtree(temp_dir)
                except: pass
                time.sleep(0.5)
                if temp_dir.exists(): 
                     temp_dir = Path.home() / f"yolo_temp_fix_{random.randint(1000,9999)}"
            
            temp_dir.mkdir(parents=True, exist_ok=True)
            self.log(f"Workspace created at: {temp_dir}")

            # Subfolders
            img_train = temp_dir / 'images' / 'train'
            img_val   = temp_dir / 'images' / 'val'
            lbl_train = temp_dir / 'labels' / 'train'
            lbl_val   = temp_dir / 'labels' / 'val'
            
            for d in [img_train, img_val, lbl_train, lbl_val]:
                d.mkdir(parents=True, exist_ok=True)

            # 2. SCAN & COLLECT DATA
            all_pairs = [] 
            
            for folder in self.source_folders:
                self.log(f"Scanning: {folder}")
                folder_path = Path(folder)
                
                images = []
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff']:
                    images.extend(list(folder_path.glob(ext)))
                    images.extend(list(folder_path.glob(ext.upper())))
                
                found_lbl_count = 0
                for img_path in images:
                    base = img_path.stem
                    parent = img_path.parent
                    
                    p_labels = [
                        parent / "labels_txt" / (base + ".txt"),
                        parent / "labels" / (base + ".txt"),
                        parent / (base + ".txt")
                    ]
                    
                    label_path = None
                    for p in p_labels:
                        if p.exists():
                            label_path = p
                            break
                    
                    if label_path:
                        all_pairs.append((img_path, label_path))
                        found_lbl_count += 1
                
                self.log(f"  > Found {len(images)} images, Matched {found_lbl_count} labels.")

            if not all_pairs:
                raise Exception("No valid image-label pairs found!")

            # 3. COPY DATA
            random.shuffle(all_pairs)
            split_idx = int(len(all_pairs) * 0.8)
            if split_idx == 0: split_idx = len(all_pairs)
            
            train_set = all_pairs[:split_idx]
            val_set = all_pairs[split_idx:]
            if not val_set: val_set = train_set

            def copy_batch(pairs, i_dst, l_dst):
                count = 0
                for i_src, l_src in pairs:
                    try:
                        shutil.copy(i_src, i_dst / i_src.name)
                        dst_lbl = l_dst / l_src.name
                        with open(l_src, 'r') as fin, open(dst_lbl, 'w') as fout:
                            for line in fin:
                                parts = line.split()
                                if parts:
                                    fout.write(line)
                        count += 1
                    except Exception as e:
                        print(f"Copy fail: {e}")
                return count

            self.log("Copying data to safe workspace...")
            copy_batch(train_set, img_train, lbl_train)
            copy_batch(val_set, img_val, lbl_val)
            
            if len(list(img_train.glob("*"))) == 0:
                raise Exception("Copy failed! Training folder is empty.")

            # 4. GENERATE YAML
            yaml_path = temp_dir / 'data.yaml'
            names_dict = {0: class_name}

            yaml_content = {
                'path':  str(temp_dir.absolute()).replace('\\', '/'),
                'train': 'images/train',
                'val':   'images/val',
                'names': names_dict
            }
            
            with open(yaml_path, 'w') as f:
                yaml.dump(yaml_content, f, sort_keys=False)

            # 5. RUN YOLO (SEGMENTATION MODEL)
            self.frame.after(0, lambda: self.lbl_stage.config(
                text=f"Status: Training {class_name} ({ver_name})...", bg="#5DADE2"
            ))
            self.log(f"\n--- Step 2: Running {ver_name} segmentation ({backbone}) ---")
            
            from ultralytics import YOLO
            model = YOLO(backbone) 
            model.add_callback("on_train_epoch_end", self.on_train_epoch_end)

            t_train0 = time.perf_counter()
            results = model.train(
                data=str(yaml_path),
                epochs=self.epochs.get(),
                imgsz=self.img_size.get(),
                batch=self.batch.get(),
                task="segment",
                device=0,
                project=str(temp_dir / "runs").replace('\\', '/'),
                name="train_result",
                workers=0
            )
            training_sec = time.perf_counter() - t_train0
            
            self.log("\n✅ TRAINING COMPLETE!")
            weights_dir = Path(results.save_dir) / "weights"
            best_pt = weights_dir / "best.pt"
            if best_pt.exists():
                self.log(f"Best weights: {best_pt}")
                self.log(f"Use these with {ver_name} in the tracking tool (Load model → Browse).")

            # 6. SAVE RESULTS
            self.log("Copying results to Output Folder...")
            final_out = Path(self.out_path.get()) / "training_results"
            
            if final_out.exists(): shutil.rmtree(final_out, ignore_errors=True)
            
            source_res = Path(results.save_dir)
            shutil.copytree(source_res, final_out)

            summary_config = {
                "yolo_version": ver,
                "yolo_version_label": ver_name,
                "backbone": backbone,
                "epochs": self.epochs.get(),
                "batch": self.batch.get(),
                "imgsz": self.img_size.get(),
                "task": "segment",
                "train_count": len(train_set),
                "val_count": len(val_set),
                "class_names": names_dict,
                "training_time_sec": round(training_sec, 1),
                "output_dir": str(final_out),
            }
            summary_text = _yolo_build_training_summary(final_out, best_pt, summary_config)
            plot_paths = _yolo_collect_training_plots(final_out)

            def _on_train_done():
                self.lbl_stage.config(text=f"Status: Finished — {class_name}!", bg="#58D68D")
                self._apply_training_summary(
                    final_out, best_pt, summary_text, plot_paths,
                    class_label=class_name, auto_open=True,
                )
                messagebox.showinfo(
                    "Success",
                    f"Training finished for: {class_name}\n\nResults saved to:\n{final_out}\n\n"
                    "Section 4 shows metrics and graphs for this object.",
                    parent=self.frame,
                )

            self.frame.after(0, _on_train_done)

        except Exception as e:
            self.log(f"\nCRITICAL ERROR: {e}")
            self.frame.after(0, lambda: self.lbl_stage.config(text="Status: Failed", bg="#E74C3C"))
            import traceback; traceback.print_exc()
        finally:
            self.is_training = False
            self.frame.after(0, lambda: self.btn_train.config(state='normal', text="MERGE DATASETS & START TRAINING", bg="#8E44AD"))            

    # --- TRACKING TOOL ---
    def open_tracking_tool(self):
        viewer = getattr(self, "_tracking_viewer", None)
        if viewer is not None:
            try:
                if viewer.window.winfo_exists():
                    viewer.window.lift()
                    viewer.window.focus_force()
                    return
            except tk.TclError:
                pass
        root = self.frame.winfo_toplevel()
        self._tracking_viewer = YoloTrackingViewer(
            root,
            yolo_version=self.yolo_version.get(),
            initial_object_name=self._trainer_class_name(),
        )