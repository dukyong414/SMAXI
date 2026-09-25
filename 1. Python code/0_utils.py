import os
import glob
import time
import numpy as np
import cv2
import threading
import shutil
import random
import yaml
import copy
import csv
import multiprocessing
import platform
import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox, Toplevel, simpledialog, Menu
from natsort import natsorted
from PIL import Image, ImageTk, ImageDraw
import scipy.ndimage
from scipy import stats
import math

from pathlib import Path
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# --- DEPENDENCY FLAGS (loaded on demand for faster app startup) ---
HAS_OLLAMA = False
HAS_GPU_NORMALIZER = False
GPU_NAME = "None (CPU Mode)"
HAS_SAM = False
HAS_YOLO = False
HAS_NIBABEL = False
HAS_PYVISTA = False

_gpu_deps_initialized = False
_ml_deps_initialized = False
_viz_deps_initialized = False


def init_gpu_normalizer(verbose=False):
    """Load CuPy and detect GPU (call before normalizer or hardware display)."""
    global HAS_GPU_NORMALIZER, GPU_NAME, _gpu_deps_initialized
    if _gpu_deps_initialized:
        return HAS_GPU_NORMALIZER
    _gpu_deps_initialized = True
    try:
        import cupy as cp
        import cupyx.scipy.ndimage  # noqa: F401
        if cp.cuda.runtime.getDeviceCount() > 0:
            HAS_GPU_NORMALIZER = True
            try:
                GPU_NAME = cp.cuda.runtime.getDeviceProperties(0)["name"].decode("utf-8")
            except Exception:
                GPU_NAME = "NVIDIA GPU Detected"
        else:
            HAS_GPU_NORMALIZER = False
            GPU_NAME = "None (CPU Mode)"
    except ImportError:
        HAS_GPU_NORMALIZER = False
        GPU_NAME = "None (CPU Mode)"
        if verbose:
            print("Warning: 'cupy' not found. GPU normalizer will use CPU.")
    return HAS_GPU_NORMALIZER


def init_ml_dependencies(verbose=False):
    """Load PyTorch, SAM (v1 and/or v2), and Ultralytics YOLO."""
    global HAS_SAM, HAS_YOLO, _ml_deps_initialized
    if _ml_deps_initialized:
        return HAS_SAM, HAS_YOLO
    _ml_deps_initialized = True
    try:
        import torch  # noqa: F401
    except ImportError:
        if verbose:
            print("Warning: 'torch' not found. SAM/YOLO tools disabled.")
    try:
        from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator  # noqa: F401
        HAS_SAM = True
    except ImportError:
        try:
            import sam2  # noqa: F401
            HAS_SAM = True
        except ImportError:
            HAS_SAM = False
            if verbose:
                print("Warning: SAM not available (install segment-anything and/or sam2).")
    try:
        from ultralytics import YOLO  # noqa: F401
        HAS_YOLO = True
    except ImportError:
        HAS_YOLO = False
        if verbose:
            print("Warning: 'ultralytics' not found. YOLO tools disabled.")
    return HAS_SAM, HAS_YOLO


def init_viz_dependencies(verbose=False):
    """Load nibabel / pyvista for CT and 3D views."""
    global HAS_NIBABEL, HAS_PYVISTA, nib, pv, _viz_deps_initialized
    if _viz_deps_initialized:
        return HAS_NIBABEL, HAS_PYVISTA
    _viz_deps_initialized = True
    nib = None
    pv = None
    try:
        import nibabel as nib
        HAS_NIBABEL = True
    except ImportError:
        HAS_NIBABEL = False
        if verbose:
            print("Warning: 'nibabel' not found. CT support disabled.")
    try:
        import pyvista as pv
        HAS_PYVISTA = True
    except ImportError:
        HAS_PYVISTA = False
        if verbose:
            print("Warning: 'pyvista' not found. 3D visualization disabled.")
    return HAS_NIBABEL, HAS_PYVISTA


# Optional: used by analysis chat (not needed at main-menu startup)
try:
    import ollama
    HAS_OLLAMA = True
except ImportError:
    HAS_OLLAMA = False

pims = None
try:
    import pims
except ImportError:
    print("Warning: 'pims' library not found. .cine support will be disabled.")

# --- GLOBAL CONFIG VARIABLES ---
BASE_DIR = ""
LOGO_FILE_PATH = ""
SAM_CHECKPOINT_DEFAULT = ""
BG_IMAGE_PATH = ""

# --- HELPER FUNCTIONS ---
def find_smaxi_install_root(start_path=None):
    """Locate the SMAXI installation package folder (works across Windows usernames)."""
    candidates = []
    if start_path:
        candidates.append(os.path.abspath(start_path))
    try:
        candidates.append(os.path.dirname(os.path.abspath(__file__)))
    except NameError:
        pass
    candidates.append(os.getcwd())

    seen = set()
    for start in candidates:
        path = os.path.abspath(start)
        while path and path not in seen:
            seen.add(path)
            if os.path.basename(path) == "SMAXI installation package":
                return path
            parent = os.path.dirname(path)
            if parent == path:
                break
            path = parent

    script_guess = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
    )
    if os.path.basename(script_guess) == "SMAXI installation package":
        return script_guess
    return ""


def find_file_under_smaxi(filename, root=None):
    """Search recursively under the SMAXI installation package for a file by name."""
    root = root or find_smaxi_install_root()
    if not root or not os.path.isdir(root):
        return ""
    for dirpath, _, filenames in os.walk(root):
        if filename in filenames:
            return os.path.join(dirpath, filename)
    return ""


def imread_safe(path):
    try:
        stream = np.fromfile(path, np.uint8)
        return cv2.imdecode(stream, cv2.IMREAD_COLOR) 
    except: return None

def imread_gray_safe(path):
    try:
        stream = np.fromfile(path, np.uint8)
        return cv2.imdecode(stream, cv2.IMREAD_GRAYSCALE)
    except: return None

def imwrite_safe(path, img):
    try:
        ext = os.path.splitext(path)[1]
        result, n = cv2.imencode(ext, img)
        if result:
            with open(path, mode='wb') as f: n.tofile(f)
            return True
        return False
    except: return False

def resize_aspect_fill(pil_img, target_w, target_h):
    img_w, img_h = pil_img.size
    ratio = max(target_w / img_w, target_h / img_h)
    new_w = int(img_w * ratio)
    new_h = int(img_h * ratio)
    pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    left = (new_w - target_w) / 2
    top = (new_h - target_h) / 2
    right = (new_w + target_w) / 2
    bottom = (new_h + target_h) / 2
    return pil_img.crop((left, top, right, bottom))

def get_system_specs():
    init_gpu_normalizer()
    spec_list = []
    spec_list.append(f"OS: {platform.system()} {platform.release()}")
    try: spec_list.append(f"CPU: {platform.processor()}")
    except: pass
    try:
        import psutil
        ram_gb = round(psutil.virtual_memory().total / (1024**3), 1)
        spec_list.append(f"RAM: {ram_gb} GB")
    except: pass
    spec_list.append(f"GPU: {GPU_NAME}")
    return " | ".join(spec_list)