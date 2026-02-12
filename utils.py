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


# --- DEPENDENCY CHECKS ---
# OLLAMA
try:
    import ollama
    HAS_OLLAMA = True
except ImportError:
    HAS_OLLAMA = False
    print("Warning: 'ollama' library not found. AI Chat will be disabled.")

# PIMS
try:
    import pims
except ImportError:
    print("Warning: 'pims' library not found. .cine support will be disabled.")

# GPU / CUPY
HAS_GPU_NORMALIZER = False
GPU_NAME = "None (CPU Mode)"
try:
    import cupy as cp
    import cupyx.scipy.ndimage
    if cp.cuda.runtime.getDeviceCount() > 0:
        HAS_GPU_NORMALIZER = True
        try: GPU_NAME = cp.cuda.runtime.getDeviceProperties(0)['name'].decode('utf-8')
        except: GPU_NAME = "NVIDIA GPU Detected"
except:
    HAS_GPU_NORMALIZER = False
    GPU_NAME = "None (CPU Mode)"

# SAM
try:
    import torch
    from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
    HAS_SAM = True
except ImportError:
    HAS_SAM = False

# YOLO
try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False

# NIBABEL / PYVISTA
try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False

try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False

# --- GLOBAL CONFIG VARIABLES ---
BASE_DIR = ""
LOGO_FILE_PATH = ""
SAM_CHECKPOINT_DEFAULT = ""
BG_IMAGE_PATH = ""

# --- HELPER FUNCTIONS ---
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