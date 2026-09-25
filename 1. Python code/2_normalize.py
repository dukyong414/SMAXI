import utils
from utils import *
import xml.etree.ElementTree as ET

_IMAGE_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}


def collect_image_paths(source: str, recursive: bool = True) -> list:
    """Collect image files from a folder or a single image path."""
    if not source:
        return []
    source = os.path.abspath(source)
    if os.path.isfile(source):
        if os.path.splitext(source)[1].lower() in _IMAGE_EXTENSIONS:
            return [source]
        return []
    if not os.path.isdir(source):
        return []

    found = set()
    if recursive:
        for root, _, names in os.walk(source):
            for name in names:
                if os.path.splitext(name)[1].lower() in _IMAGE_EXTENSIONS:
                    found.add(os.path.join(root, name))
    else:
        for name in os.listdir(source):
            full = os.path.join(source, name)
            if os.path.isfile(full) and os.path.splitext(name)[1].lower() in _IMAGE_EXTENSIONS:
                found.add(full)
    return natsorted(found)


def apply_clahe_gray(img_u8: np.ndarray, clip_limit: float, tile_size: int) -> np.ndarray:
    """Contrast Limited Adaptive Histogram Equalization (grayscale uint8)."""
    if img_u8.ndim == 3:
        gray = cv2.cvtColor(img_u8, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_u8
    tile = max(2, int(tile_size))
    clahe = cv2.createCLAHE(clipLimit=max(0.1, float(clip_limit)), tileGridSize=(tile, tile))
    return clahe.apply(gray)


def apply_clahe_float(img: np.ndarray, clip_limit: float, tile_size: int) -> np.ndarray:
    """CLAHE for float images. 12-bit-style (>255) maps via /16 then *16 like ver05."""
    d = img.astype(np.float32)
    mx = float(np.nanmax(d)) if d.size else 0.0
    if mx > 255.0:
        u8 = (np.clip(d, 0.0, 4095.0) / 16.0).astype(np.uint8)
        return apply_clahe_gray(u8, clip_limit, tile_size).astype(np.float32) * 16.0
    u8 = np.clip(d, 0.0, 255.0).astype(np.uint8)
    return apply_clahe_gray(u8, clip_limit, tile_size).astype(np.float32)


def apply_local_contrast(img: np.ndarray, strength: float, radius: int) -> np.ndarray:
    """Unsharp-style local contrast: d + strength * (d - box_blur)."""
    d = img.astype(np.float32)
    r = max(1, int(radius))
    k = 2 * r + 1
    blur = cv2.boxFilter(d, -1, (k, k))
    return d + float(strength) * (d - blur)


def apply_flat_division(frame: np.ndarray, flat: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Flat-field division with p1–p99 clip and mean rescale (ver05 formula)."""
    d = frame.astype(np.float32)
    f = np.asarray(flat, dtype=np.float32)
    if d.shape != f.shape:
        raise ValueError(f"Flat shape {f.shape} does not match frame shape {d.shape}")
    ratio = d / np.maximum(f, 1.0)
    p1, p99 = np.percentile(ratio, (1, 99))
    if p99 > p1:
        ratio = np.clip(ratio, p1, p99)
    return ratio * (float(np.mean(f)) * float(strength))


# Optional 7BM batch paths (used only if folders exist on this PC)
DEFAULT_7BM_BATCH_ROOT = r"H:\7BM imaging experiment (April 2026) 001-100"
DEFAULT_7BM_FLATFIELD_FOLDER = os.path.join(DEFAULT_7BM_BATCH_ROOT, "No087_flatfield_S001")
DEFAULT_7BM_EXPERIMENTS = (
    ("No099 — STS316H", os.path.join(
        DEFAULT_7BM_BATCH_ROOT,
        "No099_IN625_STS316H_n10X_600W_40E_200S_100msDwell_S001",
    )),
    ("No100 — Layer2", os.path.join(
        DEFAULT_7BM_BATCH_ROOT,
        "No100_IN625_STS316H_Layer2_n10X_600W_40E_200S_100msDwell_S001",
    )),
)


def find_mraw_in_folder(folder: str) -> str:
    """Return the .mraw file inside a Phantom experiment folder."""
    folder_p = Path(folder)
    if folder_p.is_file() and folder_p.suffix.lower() == ".mraw":
        return str(folder_p.resolve())
    if not folder_p.is_dir():
        raise FileNotFoundError(f"Not a folder or MRAW file:\n{folder}")
    mraws = sorted(folder_p.glob("*.mraw"), key=lambda p: p.stat().st_size, reverse=True)
    if not mraws:
        raise FileNotFoundError(f"No .mraw file found in:\n{folder}")
    return str(mraws[0].resolve())


def find_sibling_flatfield_folder(experiment_path: str) -> str:
    """Find a *flatfield* sibling folder under the same batch directory."""
    p = Path(experiment_path)
    search_root = p.parent if p.is_file() else p.parent
    if not search_root.is_dir():
        return ""
    candidates = []
    for child in search_root.iterdir():
        if child.is_dir() and "flatfield" in child.name.lower():
            try:
                find_mraw_in_folder(str(child))
                candidates.append(child)
            except FileNotFoundError:
                continue
    if not candidates:
        return ""
    candidates.sort(key=lambda c: c.name.lower())
    return str(candidates[0].resolve())


def resolve_input_to_mraw(path: str) -> tuple:
    """
    Resolve user path to (mraw_file, display_label).
    Accepts .mraw file or experiment folder containing one.
    """
    path = os.path.abspath(path.strip())
    if os.path.isdir(path):
        mraw = find_mraw_in_folder(path)
        return mraw, os.path.basename(path)
    if path.lower().endswith(".mraw"):
        return path, os.path.basename(path)
    return path, os.path.basename(path)


def find_mraw_metadata_file(mraw_path: str):
    """Locate sibling .cihx/.cih for a Phantom MRAW file."""
    mraw_p = Path(mraw_path)
    directory = mraw_p.parent
    exact_cihx = directory / (mraw_p.stem + ".cihx")
    exact_cih = directory / (mraw_p.stem + ".cih")
    if exact_cihx.exists():
        return exact_cihx
    if exact_cih.exists():
        return exact_cih
    for file in directory.iterdir():
        if file.is_file() and file.suffix.lower() in (".cih", ".cihx"):
            return file
    return None


def clean_cih(data_cih):
    """Extract parseable CIH XML from Phantom .cih / binary .cihx metadata."""
    data_cih = Path(data_cih)
    cleaned_cih = data_cih.parent.joinpath(data_cih.stem + ".xml")
    raw = data_cih.read_bytes()
    start = raw.find(b"<?xml")
    if start >= 0:
        end = raw.find(b"</cih>", start)
        if end < 0:
            raise ValueError(f"No </cih> block found in {data_cih.name}")
        xml_text = raw[start:end + len(b"</cih>")].decode("utf-8", errors="replace")
        cleaned_cih.write_text(xml_text, encoding="utf-8", newline="\n")
        return cleaned_cih

    with open(data_cih, "r", errors="replace") as meta_file:
        with open(cleaned_cih, "w", encoding="utf-8", newline="\n") as meta_xml:
            first_line = meta_file.readline()
            parts = first_line.split("<")
            meta_xml.write("<" + parts[-1] + "\n")
            while True:
                line_data = meta_file.readline()
                if not line_data:
                    break
                meta_xml.write(line_data)
                if line_data.strip().startswith("</cih>"):
                    break
    return cleaned_cih


def parse_cih_xml(cleaned_cih):
    xml_data = ET.parse(cleaned_cih)
    xml_root = xml_data.getroot()
    recorded_frames = int(xml_root.find("frameInfo").find("recordedFrame").text)
    rows = int(xml_root.find("imageFileInfo").find("resolution").find("height").text)
    columns = int(xml_root.find("imageFileInfo").find("resolution").find("width").text)
    fps = int(xml_root.find("recordInfo").find("recordRate").text)
    bits = int(xml_root.find("imageDataInfo").find("colorInfo").find("bit").text)
    return rows, columns, recorded_frames, bits, fps


def open_mraw_info(mraw_path: str):
    """Return (rows, cols, n_frames, bits, fps, frame_bytes) for an MRAW."""
    meta = find_mraw_metadata_file(mraw_path)
    if meta is None:
        raise FileNotFoundError(
            f"No .cih/.cihx metadata found beside:\n{mraw_path}"
        )
    cleaned = clean_cih(meta)
    rows, columns, recorded_frames, bits, fps = parse_cih_xml(cleaned)
    frame_bytes = int(rows * columns * 3 / 2)  # 12-bit packed
    return rows, columns, recorded_frames, bits, fps, frame_bytes


def unpack_12bit_frames(raw_bytes: np.ndarray, frames: int, rows: int, columns: int, frame_bytes: int):
    reshaped = raw_bytes.reshape(frames, int(np.ceil(frame_bytes / 3)), 3)
    final_data = np.zeros((frames, reshaped.shape[1], 2), dtype=np.uint16)
    final_data[:, :, 0] = reshaped[:, :, 0].astype(np.uint16) * 16 + (reshaped[:, :, 1] & 240) // 16
    final_data[:, :, 1] = reshaped[:, :, 2].astype(np.uint16) + (reshaped[:, :, 1] & 15).astype(np.uint16) * 256
    return final_data.reshape(frames, rows, columns)


def read_mraw_frame(mraw_path: str, frame_idx: int, frame_bytes: int, rows: int, columns: int):
    raw_byte = np.fromfile(
        mraw_path,
        dtype=np.uint8,
        count=frame_bytes,
        offset=int(frame_idx) * frame_bytes,
    )
    if raw_byte.size != frame_bytes:
        return None
    return unpack_12bit_frames(raw_byte, 1, rows, columns, frame_bytes)[0].astype(np.float32)


def compute_flat_master_mraw(flat_path: str, frame_bytes: int, rows: int, columns: int, max_frames: int = 50):
    """Mean of first N frames from a flat-field MRAW (floored at 1.0)."""
    bg_frames = max(1, int(max_frames))
    raw_bg = np.fromfile(flat_path, dtype=np.uint8, count=bg_frames * frame_bytes)
    actual = raw_bg.size // frame_bytes
    if actual == 0:
        raise ValueError("The flat-field MRAW is empty or unreadable.")
    raw_bg = raw_bg[: actual * frame_bytes]
    frames = unpack_12bit_frames(raw_bg, actual, rows, columns, frame_bytes)
    flat = np.mean(frames, axis=0, dtype=np.float32)
    return np.maximum(flat, 1.0)


class DriftGraphPopup:
    def __init__(self, master, trajectory, init_y, current_frame=0):
        self.window = Toplevel(master)
        self.window.title("Detailed Drift Trajectory")
        self.window.geometry("800x400")
        self.window.configure(bg="#17202A")
        
        self.trajectory = trajectory
        self.init_y = init_y
        
        # Graph Container
        self.frame = tk.Frame(self.window, bg="#17202A")
        self.frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.setup_graph()
        self.update_marker(current_frame)

    def setup_graph(self):
        self.fig, self.ax = plt.subplots(figsize=(8, 4), dpi=100)
        self.fig.patch.set_facecolor('#17202A') 
        self.ax.set_facecolor('#2C3E50')
        
        frames = np.arange(len(self.trajectory))
        self.ax.plot(frames, self.trajectory, label='Surface Y-Pos', color='#3498DB', linewidth=2)
        self.ax.axhline(y=self.init_y, color='#2ECC71', linestyle='--', label=f'Target Y={self.init_y}')
        
        # Interactive Marker
        self.marker_line = self.ax.axvline(x=0, color='#E74C3C', linewidth=2, label='Current Frame')
        
        # Styling
        self.ax.set_title("Thermal Drift Analysis", color="white", fontsize=12)
        self.ax.set_xlabel("Frame Index", color="white")
        self.ax.set_ylabel("Pixel Position (Y)", color="white")
        self.ax.tick_params(axis='x', colors='white')
        self.ax.tick_params(axis='y', colors='white')
        self.ax.invert_yaxis() 
        self.ax.grid(True, alpha=0.2)
        self.ax.legend(loc='upper right', facecolor='#34495E', edgecolor='white', labelcolor='white')
        
        self.fig.tight_layout()
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def update_marker(self, frame_idx):
        if hasattr(self, 'marker_line'):
            self.marker_line.set_xdata([frame_idx])
            self.canvas.draw_idle()

    def close(self):
        plt.close(self.fig)
        self.window.destroy()

## ==============================================================================
# TRIPLE PLAYER 
# ==============================================================================

class TripleVideoPlayer:
    def __init__(self, master, orig_files, stab_folder, init_y, trajectory=None, fps=30):
        self.window = Toplevel(master)
        self.window.title("Thermal Drift Correction Results")
        
        # Maximize window
        w = self.window.winfo_screenwidth()
        h = self.window.winfo_screenheight()
        self.window.geometry(f"{w}x{h}+0+0")
        try: self.window.state('zoomed')
        except: self.window.attributes('-fullscreen', True)
        
        self.window.configure(bg="#17202A")
        
        self.fps = fps
        self.base_delay = int(1000.0 / self.fps)
        
        # Data Sources
        self.orig_files = orig_files
        self.stab_folder = stab_folder
        self.init_y = int(init_y)
        self.trajectory = trajectory 
        self.total_frames = len(orig_files)
        
        self.popup = None
        
        self.is_playing = True
        self.current_pos = 0

        # --- LAYOUT CONTAINERS ---
        # 1. Top Title
        title_frame = tk.Frame(self.window, bg="#17202A")
        title_frame.pack(side="top", fill="x", pady=5)
        
        font_style = ("Arial", 12, "bold")
        tk.Label(title_frame, text="(1) Original", fg="white", bg="#17202A", font=font_style).pack(side="left", expand=True)
        tk.Label(title_frame, text="(2) Calibrated (Visual)", fg="#F1C40F", bg="#17202A", font=font_style).pack(side="left", expand=True)
        tk.Label(title_frame, text="(3) Calibrated (Clean)", fg="#2ECC71", bg="#17202A", font=font_style).pack(side="left", expand=True)

        # 2. Video Area (Top Half)
        video_frame = tk.Frame(self.window, bg="black")
        video_frame.pack(side="top", expand=True, fill="both", padx=10, pady=5)
        
        self.lbl_1 = tk.Label(video_frame, bg="black")
        self.lbl_1.pack(side="left", expand=True, fill="both", padx=2)
        
        self.lbl_2 = tk.Label(video_frame, bg="black")
        self.lbl_2.pack(side="left", expand=True, fill="both", padx=2)
        
        self.lbl_3 = tk.Label(video_frame, bg="black")
        self.lbl_3.pack(side="left", expand=True, fill="both", padx=2)

        # 3. Graph Area (Bottom Half)
        self.graph_frame = tk.Frame(self.window, bg="#17202A", height=250)
        self.graph_frame.pack(side="top", fill="x", padx=10, pady=5)
        self.graph_frame.pack_propagate(False) # Force height

        # 4. Controls (Bottom)
        ctrl_frame = tk.Frame(self.window, bg="#34495E", pady=5)
        ctrl_frame.pack(side="bottom", fill="x")
        
        tk.Button(ctrl_frame, text="Replay / Loop", command=self.reset, bg="#F39C12", font=("Arial", 11, "bold")).pack(side="left", padx=20)
        
        if self.trajectory is not None:
            tk.Button(ctrl_frame, text="📈 Pop-out Graph", command=self.open_popup, bg="#8E44AD", fg="white", font=("Arial", 11, "bold")).pack(side="left", padx=20)
        
        tk.Button(ctrl_frame, text="Close Results", command=self.close, bg="#C0392B", fg="white", font=("Arial", 11, "bold")).pack(side="right", padx=20)

        if self.trajectory is not None:
            self.embed_graph()

        self.update_frame()

    def embed_graph(self):
        """Embeds the matplotlib graph into the bottom frame"""
        # Create Plot
        self.fig, self.ax = plt.subplots(figsize=(10, 2), dpi=100)
        self.fig.patch.set_facecolor('#17202A') # Match background
        self.ax.set_facecolor('#2C3E50')
        
        # Plot Data
        frames = np.arange(len(self.trajectory))
        self.ax.plot(frames, self.trajectory, label='Surface Y-Pos', color='#3498DB', linewidth=1)
        self.ax.axhline(y=self.init_y, color='#2ECC71', linestyle='--', label=f'Target Y={self.init_y}')
        
        # Vertical Marker Line
        self.marker_line = self.ax.axvline(x=0, color='#E74C3C', linewidth=2, label='Current')
        
        # Styling
        self.ax.set_title("Melt Pool Surface Drift", color="white", fontsize=10)
        self.ax.tick_params(axis='x', colors='white')
        self.ax.tick_params(axis='y', colors='white')
        self.ax.invert_yaxis() 
        self.ax.grid(True, alpha=0.1)
        
        # Tight layout to maximize space
        self.fig.tight_layout()

        # Embed
        self.canvas_graph = FigureCanvasTkAgg(self.fig, master=self.graph_frame)
        self.canvas_graph.draw()
        self.canvas_graph.get_tk_widget().pack(fill="both", expand=True)

    def reset(self):
        self.current_pos = 0
        
    def open_popup(self):
        if self.popup is None or not tk.Toplevel.winfo_exists(self.popup.window):
            self.popup = DriftGraphPopup(self.window, self.trajectory, self.init_y, self.current_pos)
        else:
            self.popup.window.lift()
        
    def update_frame(self):
        if not self.is_playing: return
        
        if self.current_pos >= self.total_frames:
            self.reset() # Loop
            
        try:
            # 1. Read Images
            f1 = imread_safe(self.orig_files[self.current_pos]) 
            stab_path = os.path.join(self.stab_folder, f"Stab_{self.current_pos:04d}.png")
            f3 = imread_safe(stab_path)
            
            if f1 is not None and f3 is not None:
                # 2. Create Visual Overlay
                f2 = f3.copy()
                h, w = f2.shape[:2]
                cv2.line(f2, (0, self.init_y), (w, self.init_y), (0,0,255), 2)
                cv2.putText(f2, "Surface", (10, self.init_y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                
                # 3. Display
                self.display(f1, f2, f3)
                
                # 4. Update Graph Marker (Embedded)
                if hasattr(self, 'marker_line'):
                    self.marker_line.set_xdata([self.current_pos])
                    self.canvas_graph.draw_idle() 

                # 5. Update Popup Marker 
                if self.popup is not None and tk.Toplevel.winfo_exists(self.popup.window):
                    self.popup.update_marker(self.current_pos)

                self.current_pos += 1
            else:
                # Frame read failed, skip
                self.current_pos += 1 
                
        except Exception as e:
            print(f"Playback error: {e}")
            self.current_pos += 1

        self.window.after(self.base_delay, self.update_frame)

    def display(self, f1, f2, f3):
        screen_w = self.window.winfo_width()
        # Adjust width to fit 3 videos side-by-side
        target_w = int((screen_w / 3) - 20)
        if target_w < 100: target_w = 300
        
        h, w = f1.shape[:2]
        ratio = target_w / w
        target_h = int(h * ratio)
        
        def process(img):
            img = cv2.resize(img, (target_w, target_h))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return ImageTk.PhotoImage(Image.fromarray(img))

        i1 = process(f1)
        i2 = process(f2)
        i3 = process(f3)
        
        self.lbl_1.config(image=i1); self.lbl_1.image = i1
        self.lbl_2.config(image=i2); self.lbl_2.image = i2
        self.lbl_3.config(image=i3); self.lbl_3.image = i3

    def close(self):
        self.is_playing = False
        
        if self.popup is not None:
            try: self.popup.close()
            except: pass
        
        try: plt.close(self.fig)
        except: pass
        self.window.destroy()

# ==============================================================================
# THERMAL DRIFT CORRECTION APP
# ==============================================================================
class ThermalCorrectionApp:
    def __init__(self, parent_frame, on_back, on_proceed_to_norm=None):
        self.frame = tk.Frame(parent_frame)
        self.frame.pack(fill="both", expand=True)
        self.on_back = on_back
        self.on_proceed_to_norm = on_proceed_to_norm 

        # --- Variables ---
        self.input_folder = tk.StringVar()
        self.output_folder = tk.StringVar()
        self.fps = tk.IntVar(value=60)
        self.um_per_px = tk.DoubleVar(value=2.0)
        self.smooth_window = tk.IntVar(value=5)
        
        # --- NEW VISUAL VARIABLES ---
        self.vis_min = tk.DoubleVar(value=0)
        self.vis_max = tk.DoubleVar(value=4095) 
        self.vis_gamma = tk.DoubleVar(value=1.0)
        self.vis_gain = tk.DoubleVar(value=1.0)
        self.vis_bright = tk.DoubleVar(value=0.0)
        self.vis_auto = tk.BooleanVar(value=False)
        self.vis_clahe = tk.BooleanVar(value=False)
        self.vis_clahe_clip = tk.DoubleVar(value=2.0)
        self.vis_clahe_tile = tk.IntVar(value=8)
        self.current_preview_raw = None
        
        # State
        self.is_cine = False
        self.cine_video = None
        self.image_files = []
        self.roi_coords = None 
        self.start_x = 0; self.start_y = 0
        
        # Drawing Objects
        self.cur_rect = None
        self.cross_h = None
        self.cross_v = None
        
        self.preview_scale = 1.0; self.original_h = 0; self.original_w = 0
        self.is_running = False

        self.setup_ui()

    def setup_ui(self):
        # Header
        top = tk.Frame(self.frame, bg="#D7BDE2", height=60)
        top.pack(fill="x")
        tk.Button(top, text="← Back to Menu", command=self.go_back, bg="#95A5A6", fg="white").pack(side="left", padx=20)
        tk.Label(top, text="Pre-Processing: Thermal Drift Correction", bg="#D7BDE2", fg="#8E44AD", font=("Arial", 16, "bold")).pack(side="left", padx=20)
        
        if self.on_proceed_to_norm:
            tk.Button(top, text="Skip / Proceed to Normalizer →", command=self.go_to_norm, 
                      bg="#16A085", fg="white", font=("Arial", 10, "bold")).pack(side="right", padx=20)

        # Content
        content = tk.Frame(self.frame)
        content.pack(fill="both", expand=True, padx=10, pady=10)

        # --- Left Column: Controls ---
        left_col = tk.Frame(content, width=400)
        left_col.pack(side="left", fill="y", padx=(0, 10))

        # 1. IO
        f_io = tk.LabelFrame(left_col, text="1. Input / Output", font=("Arial", 10, "bold"))
        f_io.pack(fill="x", pady=5)
        
        tk.Label(f_io, text="Input Images:").pack(anchor="w", padx=5)
        tk.Entry(f_io, textvariable=self.input_folder).pack(fill="x", padx=5)
        tk.Button(f_io, text="Browse & Load", command=self.load_images).pack(fill="x", padx=5, pady=2)

        tk.Label(f_io, text="Output Folder:").pack(anchor="w", padx=5, pady=(5,0))
        tk.Entry(f_io, textvariable=self.output_folder).pack(fill="x", padx=5)
        tk.Button(f_io, text="Browse", command=self.browse_out).pack(fill="x", padx=5, pady=2)

        # 2. Parameters
        f_param = tk.LabelFrame(left_col, text="2. Parameters", font=("Arial", 10, "bold"))
        f_param.pack(fill="x", pady=5)
        grid_opts = {'padx': 5, 'pady': 5, 'sticky': 'w'}
        tk.Label(f_param, text="FPS:").grid(row=0, column=0, **grid_opts)
        tk.Entry(f_param, textvariable=self.fps, width=10).grid(row=0, column=1, **grid_opts)
        tk.Label(f_param, text="µm / px:").grid(row=1, column=0, **grid_opts)
        tk.Entry(f_param, textvariable=self.um_per_px, width=10).grid(row=1, column=1, **grid_opts)
        tk.Label(f_param, text="Smooth Win:").grid(row=2, column=0, **grid_opts)
        tk.Entry(f_param, textvariable=self.smooth_window, width=10).grid(row=2, column=1, **grid_opts)

        # ============================================================
        # Visual Adjustments 
        # ============================================================
        f_vis = tk.LabelFrame(left_col, text="3. Visual Adjustments", font=("Arial", 10, "bold"), bg="#EAECEE")
        f_vis.pack(fill="x", pady=5)

        def add_slider(parent, label, var, from_, to_, res, row):
            tk.Label(parent, text=label, bg="#EAECEE", anchor="w", width=12).grid(row=row, column=0, padx=5, sticky="w")
            s = tk.Scale(parent, variable=var, from_=from_, to=to_, resolution=res, orient="horizontal", 
                         showvalue=False, bg="#EAECEE", length=180, command=lambda x: self.refresh_preview_visuals())
            s.grid(row=row, column=1, padx=5)
            tk.Entry(parent, textvariable=var, width=6).grid(row=row, column=2, padx=5)

        add_slider(f_vis, "Levels Min:", self.vis_min, 0, 65535, 1, 0)
        add_slider(f_vis, "Levels Max:", self.vis_max, 10, 65535, 10, 1)
        add_slider(f_vis, "Gamma:", self.vis_gamma, 0.1, 5.0, 0.1, 2)
        add_slider(f_vis, "Gain:", self.vis_gain, 0.1, 10.0, 0.1, 3)
        add_slider(f_vis, "Brightness:", self.vis_bright, -100, 100, 1, 4)
        add_slider(f_vis, "CLAHE clip:", self.vis_clahe_clip, 0.5, 40.0, 0.5, 5)
        add_slider(f_vis, "CLAHE tile:", self.vis_clahe_tile, 4, 32, 2, 6)
        
        tk.Checkbutton(f_vis, text="Auto-Levels", variable=self.vis_auto, 
                       bg="#EAECEE", command=self.refresh_preview_visuals).grid(row=7, column=0, columnspan=3, sticky="w", padx=5)
        tk.Checkbutton(
            f_vis,
            text="Enable CLAHE (local contrast)",
            variable=self.vis_clahe,
            bg="#EAECEE",
            command=self.refresh_preview_visuals,
        ).grid(row=8, column=0, columnspan=3, sticky="w", padx=5)

        # ============================================================
        # Pixel Histogram
        # ============================================================
        f_hist = tk.LabelFrame(left_col, text="Pixel Histogram", font=("Arial", 10, "bold"))
        f_hist.pack(fill="x", pady=2)
        
        self.fig, self.ax = plt.subplots(figsize=(4, 2), dpi=80)
        self.fig.patch.set_facecolor('#F0F0F0')
        self.canvas_hist = FigureCanvasTkAgg(self.fig, master=f_hist)
        self.canvas_hist.get_tk_widget().pack(fill="both", expand=True)
        


        # 3. Actions
        self.btn_run = tk.Button(left_col, text="▶ START CORRECTION", bg="#8E44AD", fg="white",disabledforeground="#E8DAEF", font=("Arial", 12, "bold"), 
                                 height=2, command=self.start_processing, state="disabled")
        self.btn_run.pack(fill="x", pady=20)

        self.log_area = scrolledtext.ScrolledText(left_col, height=10)
        self.log_area.pack(fill="both", expand=True)

        # --- Right Column: Canvas ---
        right_col = tk.Frame(content, bg="black", bd=2, relief="sunken")
        right_col.pack(side="right", fill="both", expand=True)
        tk.Label(right_col, text="Draw ROI Box over the SURFACE LINE (Left Click & Drag)", bg="black", fg="yellow").pack(side="top", fill="x")
        self.canvas = tk.Canvas(right_col, bg="gray", cursor="cross")
        self.canvas.pack(fill="both", expand=True)
        
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

    def _safe_imread(self, path, color=True):
        """Helper to force read Korean paths on Windows"""
        try:
            path = os.path.abspath(path)
            stream = open(path, "rb")
            bytes = bytearray(stream.read())
            numpyarray = np.asarray(bytes, dtype=np.uint8)
            mode = cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE
            return cv2.imdecode(numpyarray, mode)
        except Exception as e:
            return None

    def _safe_imwrite(self, path, img):
        """Helper to force write Korean paths on Windows"""
        try:
            path = os.path.abspath(path)
        
            if len(path) > 259 and os.name == 'nt' and not path.startswith('\\\\?\\'):
                path = '\\\\?\\' + path

            ext = os.path.splitext(path)[1]
            result, n = cv2.imencode(ext, img)
            if result:
                with open(path, mode='wb') as f:
                    n.tofile(f)
            return True
        except Exception as e:
            print(f"WRITE ERROR: {e}") 
            return False

    def update_histogram(self, data):
        if data is None or not hasattr(self, 'ax'): return
        self.ax.clear()
        flat_data = data.flatten()[::10] # Subsample for speed
        self.ax.hist(flat_data, bins=50, color='gray', alpha=0.7)
        self.ax.set_title(f"Range: {np.min(data):.0f} - {np.max(data):.0f}", fontsize=8)
        self.ax.get_yaxis().set_visible(False)
        self.fig.tight_layout()
        self.canvas_hist.draw()

    def apply_visuals(self, img_raw):
        if img_raw is None: return None
        img = img_raw.astype(np.float32)
        
        if self.vis_auto.get():
            low = np.percentile(img, 1)
            high = np.percentile(img, 99)
        else:
            low = self.vis_min.get()
            high = self.vis_max.get()
            
        if high <= low: high = low + 1
        img = (img - low) / (high - low)
        img = np.clip(img, 0.0, 1.0)
        
        # Gamma/Gain/Bright
        gamma = self.vis_gamma.get()
        if gamma != 1.0 and gamma > 0: img = np.power(img, 1.0 / gamma)
        img = img * self.vis_gain.get() + (self.vis_bright.get() / 255.0)
        u8 = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
        if self.vis_clahe.get():
            u8 = apply_clahe_gray(u8, self.vis_clahe_clip.get(), self.vis_clahe_tile.get())
        return u8

    def refresh_preview_visuals(self, event=None):
        if self.current_preview_raw is None: return
        
        # 1. Apply visuals
        vis_img = self.apply_visuals(self.current_preview_raw)
        
        # 2. Convert for Tkinter
        nw = int(self.original_w * self.preview_scale)
        nh = int(self.original_h * self.preview_scale)
        pil_img = Image.fromarray(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
        pil_img = pil_img.resize((nw, nh), Image.Resampling.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(pil_img)
        
        # 3. Update Canvas Background Only 
        self.canvas.itemconfig("bg_img", image=self.tk_img)

    def go_back(self):
        if self.cine_video:
            try: self.cine_video.close()
            except: pass
        self.frame.destroy()
        self.on_back()
    
    def go_to_norm(self):
        self.frame.destroy()
        if self.on_proceed_to_norm: self.on_proceed_to_norm()

    def log(self, msg):
        self.log_area.insert(tk.END, msg + "\n")
        self.log_area.see(tk.END)

    def browse_out(self):
        d = filedialog.askdirectory()
        if d: self.output_folder.set(d)

    def load_images(self):
        # 1. Ask user for type
        ans = messagebox.askyesno("Input Type", "Are you loading a CINE file?\n\nYes = .cine file\nNo = Folder of images")
        
        if ans:
            # CINE Mode
            f = filedialog.askopenfilename(filetypes=[("CINE", "*.cine"), ("All", "*.*")])
            if not f: return
            self.input_folder.set(f)
            self.is_cine = True
            
            # Suggest output folder based on file location
            d = os.path.dirname(f)
            raw_path = os.path.join(d, "Processed_Stabilized")
            self.output_folder.set(os.path.abspath(raw_path))
            
            self.log(f"Selected CINE: {os.path.basename(f)}")
            self.show_first_frame()
            
        else:
            # Folder Mode (Your original logic)
            d = filedialog.askdirectory(title="Select Input Image Folder")
            if not d: return
            self.input_folder.set(d)
            self.is_cine = False
            
            if not self.output_folder.get():
                raw_path = os.path.join(d, "..", "Processed_Stabilized")
                self.output_folder.set(os.path.abspath(raw_path))

            self.log(f"Scanning folder: {d}...")

            self.image_files = collect_image_paths(d)
            
            if not self.image_files:
                messagebox.showerror("Error", "No images found!\nCheck if the folder contains images.")
                return

            self.log(f"Success: Loaded {len(self.image_files)} images.")
            self.show_first_frame()

    def show_first_frame(self):
        img = None
        
        if self.is_cine:
            try:
                v = pims.open(self.input_folder.get())
                img = v[0] 
                v.close()
                img = np.array(img)
                if img.dtype == np.uint16:
                    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
                    img = img.astype(np.uint8)
                if len(img.shape) == 3: img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                elif len(img.shape) == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            except Exception as e:
                self.log(f"ERROR: Could not read CINE file: {e}")
                return
        else:
            if not self.image_files: return
            img = self._safe_imread(self.image_files[0], color=True)
        
        if img is None: 
            self.log("ERROR: Could not read first image.")
            return

        self.original_h, self.original_w = img.shape[:2]

        
        if img is None: 
            self.log("ERROR: Could not read first image. Check paths.")
            return

        self.original_h, self.original_w = img.shape[:2]
        
        # 1. Save RAW data for the sliders to use
        self.current_preview_raw = img.copy()
        
        # 2. Update Histogram
        self.update_histogram(self.current_preview_raw)
        
        # 3. Setup Canvas Scaling
        cw = self.canvas.winfo_width(); ch = self.canvas.winfo_height()
        if cw < 100: cw = 600; ch = 400
        self.preview_scale = min(cw / self.original_w, ch / self.original_h)
        
        # 4. Draw Initial Image (Using Visual Logic)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", tags="bg_img") 
        
        # 5. Trigger the visual refresh to populate the image
        self.refresh_preview_visuals()
        
        self.roi_coords = None
        self.btn_run.config(state="disabled")
        self.log("Draw box over Surface Line to enable Start.")


    def on_mouse_down(self, event):
        self.start_x = event.x; self.start_y = event.y
        if self.cur_rect: self.canvas.delete(self.cur_rect)
        if self.cross_h: self.canvas.delete(self.cross_h)
        if self.cross_v: self.canvas.delete(self.cross_v)
        
        self.cur_rect = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline="red", width=2)
        self.cross_h = self.canvas.create_line(self.start_x, self.start_y, self.start_x, self.start_y, fill="red", dash=(4,4))
        self.cross_v = self.canvas.create_line(self.start_x, self.start_y, self.start_x, self.start_y, fill="red", dash=(4,4))

    def on_mouse_drag(self, event):
        self.canvas.coords(self.cur_rect, self.start_x, self.start_y, event.x, event.y)
        cx = (self.start_x + event.x) / 2
        cy = (self.start_y + event.y) / 2
        self.canvas.coords(self.cross_h, self.start_x, cy, event.x, cy)
        self.canvas.coords(self.cross_v, cx, self.start_y, cx, event.y)

    def on_mouse_up(self, event):
        x1, y1 = self.start_x, self.start_y; x2, y2 = event.x, event.y
        rx = min(x1, x2); ry = min(y1, y2); rw = abs(x2 - x1); rh = abs(y2 - y1)
        if rw < 5 or rh < 5: return
        
        real_x = int(rx / self.preview_scale); real_y = int(ry / self.preview_scale)
        real_w = int(rw / self.preview_scale); real_h = int(rh / self.preview_scale)
        real_x = max(0, min(real_x, self.original_w - 1))
        real_y = max(0, min(real_y, self.original_h - 1))
        real_w = min(real_w, self.original_w - real_x)
        real_h = min(real_h, self.original_h - real_y)
        
        self.roi_coords = (real_x, real_y, real_w, real_h)
        self.log(f"ROI Selected: {self.roi_coords}")
        self.btn_run.config(state="normal")

    def start_processing(self):
        if self.is_running: return
        self.is_running = True
        self.btn_run.config(state="disabled", text="Processing...")
        threading.Thread(target=self.run_algorithm, daemon=True).start()

    def run_algorithm(self):
        video_reader = None
        try:
            # 1. SETUP PATHS
            out_folder = os.path.abspath(self.output_folder.get())
            if not out_folder:
                self.log("Error: Output folder invalid.")
                return

            try: os.makedirs(out_folder, exist_ok=True)
            except: pass

            img_sub = os.path.join(out_folder, "Stabilized_Images")
            os.makedirs(img_sub, exist_ok=True)
            rx, ry, rw, rh = self.roi_coords
            
            # 2. INITIALIZE READER (CINE vs FOLDER)
            total_frames_count = 0
            if self.is_cine:
                self.log("Opening CINE file...")
                video_reader = pims.open(self.input_folder.get())
                total_frames_count = len(video_reader)
            else:
                total_frames_count = len(self.image_files)

            if total_frames_count == 0:
                self.log("Error: No frames found.")
                return

            # --- PHASE 1: ANALYSIS (DRIFT DETECTION) ---
            self.log(f"Analyzing Drift Trajectory ({total_frames_count} frames)...")
            from scipy.signal import savgol_filter
            raw_y = []
            
            for i in range(total_frames_count):
                if i % 50 == 0: self.log(f"Scanning {i}/{total_frames_count}")
                
                # A. Retrieve Frame (Gray for analysis)
                img = None
                if self.is_cine:
                    try: 
                        # PIMS access
                        frame_data = video_reader[i]
                        img = np.array(frame_data)
                        if len(img.shape) == 3: 
                            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                    except Exception as e: 
                        img = None
                else:
                    # Folder access
                    img = self._safe_imread(self.image_files[i], color=False)
                
                # Handle read failures
                if img is None:
                    val = raw_y[-1] if raw_y else 0
                    raw_y.append(val)
                    continue
                
                # B. Calculate Drift
                h_img, w_img = img.shape[:2]
                valid_w = min(rw, w_img - rx)
                
                strip = img[:, rx:rx+valid_w]
                prof = np.mean(strip, axis=1)
                grad = np.gradient(prof)
                
                # Find strongest edge
                raw_y.append(np.argmax(np.abs(grad)))

            if not raw_y:
                self.log("Error: No data extracted. Check input.")
                return

            # C. Smoothing
            raw_y = np.array(raw_y)
            win = self.smooth_window.get()
            if win % 2 == 0: win += 1
            
            if len(raw_y) <= win:
                self.log("Warning: Video too short for smoothing. Using raw data.")
                smooth_y = raw_y
            else:
                try: 
                    smooth_y = savgol_filter(raw_y, win, 2)
                except Exception as e: 
                    self.log(f"Filter Error: {e}. Using raw.")
                    smooth_y = raw_y
            
            init_y = smooth_y[0]
            if np.isnan(init_y): init_y = 0

            self.log("Correcting Images & Writing Videos...")
            
            h, w = self.original_h, self.original_w
            fps_val = self.fps.get()
            
            path_orig = os.path.join(out_folder, "1_Original.mp4")
            path_stab = os.path.join(out_folder, "2_Stabilized_Clean.mp4")
            path_vis  = os.path.join(out_folder, "3_Stabilized_With_Marker.mp4")

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            v_orig = cv2.VideoWriter(path_orig, fourcc, fps_val, (w, h))
            v_stab = cv2.VideoWriter(path_stab, fourcc, fps_val, (w, h))
            v_vis  = cv2.VideoWriter(path_vis, fourcc, fps_val, (w, h))

            if not v_orig.isOpened():
                self.log("Error: Could not open VideoWriter.")

            count = 0
            for i in range(total_frames_count):
                if i % 50 == 0: self.log(f"Writing {i}/{total_frames_count}")
                
                # A. Retrieve Frame (Color for Output)
                frame = None
                if self.is_cine:
                    try:
                        frame_data = video_reader[i]
                        frame = np.array(frame_data)
                        
                        # PIMS usually reads RGB, OpenCV needs BGR
                        if len(frame.shape) == 3:
                            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        elif len(frame.shape) == 2:
                            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    except: frame = None
                else:
                    frame = self._safe_imread(self.image_files[i], color=True)
                
                if frame is None: 
                    self.log(f"Skip frame {i}: Read failed")
                    continue
                
                count += 1
                
                # B. Apply Shift
                cur_y = smooth_y[i]
                if np.isnan(cur_y): cur_y = init_y
                
                shift_y = init_y - cur_y
                M = np.float32([[1, 0, 0], [0, 1, shift_y]])
                stab = cv2.warpAffine(frame, M, (w, h))
                
                # C. Export Individual Images (PNG)
                save_p = os.path.join(img_sub, f"Stab_{i:04d}.png")
                self._safe_imwrite(save_p, stab)

                # D. Export Video Frames
                v_orig.write(frame)
                v_stab.write(stab)

                # E. Visualization Frame (Red Line)
                vis = stab.copy()
                cv2.line(vis, (0, int(init_y)), (w, int(init_y)), (0,0,255), 2)
                cv2.putText(vis, "Surface Level", (10, int(init_y)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                v_vis.write(vis)

            # Cleanup Writers
            v_orig.release()
            v_stab.release()
            v_vis.release()
            
            if video_reader: 
                try: video_reader.close()
                except: pass
            
            self.log(f"Done. Processed {count} frames.")
            
            player_source = self.input_folder.get() if self.is_cine else self.image_files
            
            self.frame.after(0, lambda: TripleVideoPlayer(self.frame, player_source, img_sub, init_y, trajectory=smooth_y, fps=fps_val))
            
            messagebox.showinfo("Complete", f"Correction finished.\nSaved to: {out_folder}")

        except Exception as e: 
            self.log(f"CRITICAL ERROR: {e}")
            print(f"CRITICAL ERROR: {e}")
            import traceback
            traceback.print_exc()
            if video_reader: 
                try: video_reader.close()
                except: pass
        finally:
            self.is_running = False
            self.btn_run.config(state="normal", text="▶ START CORRECTION", fg ="#FFFFFF")
# ==============================================================================
#  DUAL VIDEO PLAYER
# ==============================================================================
class DualVideoPlayer:
    def __init__(self, master, original_path, normalized_path, fps=30):
        self.window = Toplevel(master)
        self.window.title("FAST-AM Viewer")
        self.window.geometry("")
        self.window.update_idletasks()
        try: self.window.state('zoomed')
        except: self.window.attributes('-fullscreen', True)
        self.window.configure(bg="#2C3E50")
        
        self.fps = fps
        self.base_delay = int(1000.0 / self.fps)
        
        self.cap_orig = cv2.VideoCapture(original_path)
        self.cap_norm = cv2.VideoCapture(normalized_path)
        self.total_frames = int(self.cap_orig.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.is_playing = True
        self.play_direction = 1 
        self.current_pos = 0

        title_frame = tk.Frame(self.window, bg="#2C3E50")
        title_frame.pack(side="top", fill="x", pady=10)
        tk.Label(title_frame, text="Original Input", fg="white", bg="#2C3E50", font=("Arial", 16)).pack(side="left", expand=True)
        tk.Label(title_frame, text="Normalized Output", fg="#F1C40F", bg="#2C3E50", font=("Arial", 16, "bold")).pack(side="right", expand=True)
        
        video_frame = tk.Frame(self.window, bg="black")
        video_frame.pack(expand=True, fill="both", padx=20, pady=10)
        self.lbl_orig = tk.Label(video_frame, bg="black")
        self.lbl_orig.pack(side="left", expand=True, fill="both", padx=5)
        self.lbl_norm = tk.Label(video_frame, bg="black")
        self.lbl_norm.pack(side="right", expand=True, fill="both", padx=5)
        
        ctrl_frame = tk.Frame(self.window, bg="#34495E", pady=15)
        ctrl_frame.pack(side="bottom", fill="x")
        
        tk.Label(ctrl_frame, text="Speed:", bg="#34495E", fg="white", font=("Arial", 12)).pack(side="left", padx=10)
        self.speed_scale = tk.Scale(ctrl_frame, from_=0.1, to=30.0, resolution=0.1, orient="horizontal", length=300, bg="#34495E", fg="white", highlightthickness=0, label="Multiplier")
        self.speed_scale.set(1.0)
        self.speed_scale.pack(side="left", padx=10)

        btn_box = tk.Frame(ctrl_frame, bg="#34495E")
        btn_box.pack(side="left", padx=30)

        tk.Button(btn_box, text="◀ Reverse", command=self.play_reverse, width=10, bg="#E74C3C", fg="white", font=("Arial", 10, "bold")).pack(side="left", padx=5)
        tk.Button(btn_box, text="⏹ Stop", command=self.stop_reset, width=8, bg="#7F8C8D", fg="white", font=("Arial", 10, "bold")).pack(side="left", padx=5)
        self.btn_play = tk.Button(btn_box, text="⏸ Pause", command=self.toggle_play, width=10, bg="#F1C40F", fg="black", font=("Arial", 10, "bold"))
        self.btn_play.pack(side="left", padx=5)
        tk.Button(btn_box, text="Forward ▶", command=self.play_forward, width=10, bg="#2ECC71", fg="white", font=("Arial", 10, "bold")).pack(side="left", padx=5)
        tk.Button(ctrl_frame, text="Close", command=self.close, bg="black", fg="white", font=("Arial", 10, "bold")).pack(side="right", padx=20)
        
        self.update_frame()

    def toggle_play(self):
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.btn_play.config(text="⏸ Pause", bg="#F1C40F")
            self.update_frame() 
        else:
            self.btn_play.config(text="▶ Play", bg="#27AE60")

    def play_forward(self):
        self.play_direction = 1
        if not self.is_playing: self.toggle_play()

    def play_reverse(self):
        self.play_direction = -1
        if not self.is_playing: self.toggle_play()

    def stop_reset(self):
        self.is_playing = False
        self.current_pos = 0
        self.btn_play.config(text="▶ Play", bg="#27AE60")
        self.seek_and_show(0)

    def seek_and_show(self, frame_idx):
        self.cap_orig.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        self.cap_norm.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret1, frame1 = self.cap_orig.read()
        ret2, frame2 = self.cap_norm.read()
        if ret1 and ret2: self.display(frame1, frame2)

    def update_frame(self):
        if not self.is_playing: return
        speed = self.speed_scale.get()
        step = int(speed) if speed >= 1.0 else 1
        
        next_pos = self.current_pos + (step * self.play_direction)
        if next_pos >= self.total_frames: next_pos = 0
        elif next_pos < 0: next_pos = self.total_frames - 1
            
        self.current_pos = next_pos
        self.cap_orig.set(cv2.CAP_PROP_POS_FRAMES, self.current_pos)
        self.cap_norm.set(cv2.CAP_PROP_POS_FRAMES, self.current_pos)
        
        ret1, frame1 = self.cap_orig.read()
        ret2, frame2 = self.cap_norm.read()
        
        if ret1 and ret2: self.display(frame1, frame2)
        
        delay = self.base_delay
        if speed < 1.0: delay = int(self.base_delay / speed)
        self.window.after(delay, self.update_frame)

    def display(self, frame1, frame2):
        screen_w = self.window.winfo_width()
        disp_w = int((screen_w / 2) - 40)
        if disp_w < 100: disp_w = 600
        h, w = frame1.shape[:2]
        scale = disp_w / w
        disp_h = int(h * scale)
        
        img1 = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(cv2.resize(frame1, (disp_w, disp_h)), cv2.COLOR_BGR2RGB)))
        img2 = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(cv2.resize(frame2, (disp_w, disp_h)), cv2.COLOR_BGR2RGB)))
        self.lbl_orig.configure(image=img1); self.lbl_orig.image = img1
        self.lbl_norm.configure(image=img2); self.lbl_norm.image = img2

    def close(self):
        self.is_playing = False
        self.cap_orig.release()
        self.cap_norm.release()
        self.window.destroy()

# ==============================================================================
# NORMALIZER APP 
# ==============================================================================
class NormalizerApp:
    def __init__(self, parent_frame, on_back):
        self.frame = tk.Frame(parent_frame)
        self.frame.pack(fill="both", expand=True)
        self.on_back = on_back
        
        # --- File Paths ---
        self.input_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.video_name = tk.StringVar(value="normalized_output.mp4")
        self.flat_path = tk.StringVar()
        
        # --- Processing Settings ---
        self.export_all = tk.BooleanVar(value=True)
        self.start_frame = tk.IntVar(value=0)
        self.end_frame = tk.IntVar(value=1000)
        self.bg_start_frame = tk.IntVar(value=0)
        self.bg_frames = tk.IntVar(value=10)
        self.start_after_bg = tk.BooleanVar(value=True)
        self.window_size = tk.IntVar(value=1)
        self.fps = tk.DoubleVar(value=30.0)
        self.use_gpu = tk.BooleanVar(value=utils.HAS_GPU_NORMALIZER)
        self.use_flatfield = tk.BooleanVar(value=False)
        self.flat_norm_strength = tk.DoubleVar(value=1.0)
        
        self.save_vis_imgs = tk.BooleanVar(value=False)
        
        # --- Normalization Settings ---
        self.norm_mode = tk.StringVar(value="grey") 
        self.min_ratio = tk.DoubleVar(value=0.60)
        self.max_ratio = tk.DoubleVar(value=1.40)
        self.binary_thresh = tk.DoubleVar(value=0.075)

        # --- Visual Adjustment Variables ---
        self.vis_gamma = tk.DoubleVar(value=1.0)
        self.vis_gain = tk.DoubleVar(value=1.0)
        self.vis_bright = tk.DoubleVar(value=0.0)
        self.vis_min = tk.IntVar(value=0)
        self.vis_max = tk.IntVar(value=4095) 
        self.vis_auto = tk.BooleanVar(value=True)

        # --- X-ray enhancement (CLAHE + local contrast; ver05 defaults) ---
        self.norm_clahe = tk.BooleanVar(value=False)
        self.norm_clahe_clip = tk.DoubleVar(value=2.0)
        self.norm_clahe_tile = tk.IntVar(value=8)
        self.local_contrast_enabled = tk.BooleanVar(value=False)
        self.local_contrast_strength = tk.DoubleVar(value=1.5)
        self.local_contrast_radius = tk.IntVar(value=15)
        self.flip_h = tk.BooleanVar(value=False)
        self.flip_v = tk.BooleanVar(value=False)

        # --- Preview State ---
        self.preview_frame_idx = tk.IntVar(value=0) 
        self.is_running = False
        self.current_preview_raw = None
        self.current_preview_bg = None
        self.fig = None 
        self.ax = None
        self.canvas_hist = None
        
        self.is_cine_mode = False
        self.is_mraw_mode = False
        self.cine_video = None
        self.image_files = []
        self.mraw_rows = 0
        self.mraw_cols = 0
        self.mraw_frame_bytes = 0
        self.mraw_bits = 12
        self.flat_master = None
        
        self.setup_ui()

    def setup_ui(self):
        # --- Header ---
        top = tk.Frame(self.frame, bg="#E8F8F5", height=50)
        top.pack(fill="x")
        tk.Button(top, text="← Back", command=self.go_back, bg="#95A5A6", fg="white").pack(side="left", padx=10, pady=10)
        tk.Label(top, text="Image Normalizer & Enhancer", bg="#E8F8F5", fg="#16A085", font=("Arial", 16, "bold")).pack(side="left", padx=10)

        status_color = "green" if HAS_GPU_NORMALIZER else "red"
        tk.Label(top, text=f"GPU: {GPU_NAME}", fg=status_color, bg="#E8F8F5", font=("Arial", 9, "bold")).pack(side="right", padx=10)

        # --- Main Layout ---
        main_content = tk.Frame(self.frame)
        main_content.pack(fill="both", expand=True, padx=10, pady=5)

        # Left Column (scrollable) — wide enough for all controls/buttons
        left_scroll_host = tk.Frame(main_content, width=560)
        left_scroll_host.pack(side="left", fill="y", padx=(0, 5))
        left_scroll_host.pack_propagate(False)

        self._left_canvas = tk.Canvas(left_scroll_host, width=540, highlightthickness=0, bg="#F4F6F6")
        left_sb = tk.Scrollbar(left_scroll_host, orient="vertical", command=self._left_canvas.yview)
        left_col = tk.Frame(self._left_canvas, bg="#F4F6F6")
        left_col.bind(
            "<Configure>",
            lambda e: self._left_canvas.configure(scrollregion=self._left_canvas.bbox("all")),
        )
        self._left_panel_id = self._left_canvas.create_window((0, 0), window=left_col, anchor="nw")
        self._left_canvas.configure(yscrollcommand=left_sb.set)
        self._left_canvas.bind(
            "<Configure>",
            lambda e: self._left_canvas.itemconfig(self._left_panel_id, width=max(e.width, 520)),
        )
        self._left_canvas.pack(side="left", fill="both", expand=True)
        left_sb.pack(side="right", fill="y")

        def _on_left_wheel(event):
            if getattr(event, "delta", 0):
                self._left_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
            elif event.num == 4:
                self._left_canvas.yview_scroll(-1, "units")
            elif event.num == 5:
                self._left_canvas.yview_scroll(1, "units")

        for widget in (left_scroll_host, self._left_canvas, left_col):
            widget.bind("<MouseWheel>", _on_left_wheel)
            widget.bind("<Button-4>", _on_left_wheel)
            widget.bind("<Button-5>", _on_left_wheel)

        # Right Column
        right_col = tk.Frame(main_content)
        right_col.pack(side="right", fill="both", expand=True, padx=(5, 0))

        # ================= LEFT COLUMN =================
        # 1. IO
        f_io = tk.LabelFrame(left_col, text="1. Input / Output", font=("Arial", 10, "bold"), fg="#2C3E50")
        f_io.pack(fill="x", pady=2)
        grid_opts = {'padx': 5, 'pady': 2, 'sticky': 'w'}

        tk.Label(f_io, text="Input (.cine/.mraw/.tif):").grid(row=0, column=0, **grid_opts)
        self.ent_input = tk.Entry(f_io, textvariable=self.input_path, width=35)
        self.ent_input.grid(row=0, column=1, **grid_opts)
        self.ent_input.bind("<Return>", lambda _e: self.load_input_source())
        tk.Button(f_io, text="Browse", command=self.browse_input).grid(row=0, column=2, **grid_opts)
        
        tk.Label(f_io, text="Output Folder:").grid(row=1, column=0, **grid_opts)
        tk.Entry(f_io, textvariable=self.output_path, width=35).grid(row=1, column=1, **grid_opts)
        tk.Button(f_io, text="Browse", command=lambda: self.browse_folder(self.output_path)).grid(row=1, column=2, **grid_opts)
        
        tk.Label(f_io, text="Video Name:").grid(row=2, column=0, **grid_opts)
        tk.Entry(f_io, textvariable=self.video_name, width=35).grid(row=2, column=1, **grid_opts)

        tk.Label(f_io, text="Flat-field:").grid(row=3, column=0, **grid_opts)
        tk.Entry(f_io, textvariable=self.flat_path, width=35).grid(row=3, column=1, **grid_opts)
        tk.Button(f_io, text="Browse", command=self.browse_flatfield).grid(row=3, column=2, **grid_opts)

        flat_opts = tk.Frame(f_io)
        flat_opts.grid(row=4, column=0, columnspan=3, sticky="w", padx=5, pady=(0, 4))
        tk.Checkbutton(
            flat_opts, text="Use flat-field correction",
            variable=self.use_flatfield, command=self.on_flatfield_toggle,
        ).pack(side="left")
        tk.Label(flat_opts, text="Strength:").pack(side="left", padx=(10, 2))
        tk.Entry(flat_opts, textvariable=self.flat_norm_strength, width=5).pack(side="left")
        self.lbl_flat_status = tk.Label(flat_opts, text="(none)", fg="#7F8C8D")
        self.lbl_flat_status.pack(side="left", padx=8)

        # 7BM quick load (shown only when batch folders exist)
        self._7bm_quick_frame = tk.LabelFrame(
            left_col,
            text="7BM experiment quick load",
            font=("Arial", 10, "bold"),
            fg="#2C3E50",
        )
        qf = tk.Frame(self._7bm_quick_frame)
        qf.pack(fill="x", padx=5, pady=4)
        tk.Label(
            qf,
            text="Loads .mraw from folder + No087 flat-field",
            font=("Arial", 8),
            fg="#566573",
        ).pack(anchor="w")
        btn_row = tk.Frame(qf)
        btn_row.pack(fill="x", pady=(4, 2))
        n_quick = 0
        for label, folder in DEFAULT_7BM_EXPERIMENTS:
            if os.path.isdir(folder):
                tk.Button(
                    btn_row,
                    text=label,
                    command=lambda f=folder: self.load_7bm_experiment(f),
                    bg="#1F618D",
                    fg="white",
                    font=("Arial", 9, "bold"),
                ).pack(side="left", padx=(0, 6), pady=2)
                n_quick += 1
        if n_quick:
            self._7bm_quick_frame.pack(fill="x", pady=2)

        # 2. Normalization Params
        f_norm = tk.LabelFrame(left_col, text="2. Normalization Physics", font=("Arial", 10, "bold"), fg="#2C3E50")
        f_norm.pack(fill="x", pady=2)
        
        # Method Selection
        tk.Label(f_norm, text="Method:").grid(row=0, column=0, padx=5, sticky="w")
        tk.Radiobutton(f_norm, text="Grey-scale", variable=self.norm_mode, value="grey", command=self.update_mode_ui).grid(row=0, column=1)
        tk.Radiobutton(f_norm, text="Binary", variable=self.norm_mode, value="binary", command=self.update_mode_ui).grid(row=0, column=2)
        tk.Button(
            f_norm, text="Apply", command=self.apply_norm_preview,
            bg="#16A085", fg="white", font=("Arial", 9, "bold"),
        ).grid(row=0, column=3, padx=5, sticky="w")

        # Ratios (For Grey-scale)
        tk.Label(f_norm, text="Min Ratio (Black):").grid(row=1, column=0, padx=5, sticky="w")
        self.ent_min = tk.Entry(f_norm, textvariable=self.min_ratio, width=6)
        self.ent_min.grid(row=1, column=1, sticky="w")
        
        tk.Label(f_norm, text="Max Ratio (White):").grid(row=1, column=2, padx=5, sticky="w")
        self.ent_max = tk.Entry(f_norm, textvariable=self.max_ratio, width=6)
        self.ent_max.grid(row=1, column=3, sticky="w")

        # Binary Threshold (For Binary)
        tk.Label(f_norm, text="Binary Threshold:").grid(row=2, column=0, padx=5, sticky="w")
        self.ent_bin = tk.Entry(f_norm, textvariable=self.binary_thresh, width=6)
        self.ent_bin.grid(row=2, column=1, sticky="w")

        # Temporal / background settings
        tk.Label(f_norm, text="BG Start Frame:").grid(row=3, column=0, padx=5, sticky="w")
        tk.Entry(f_norm, textvariable=self.bg_start_frame, width=6).grid(row=3, column=1, sticky="w")
        tk.Button(
            f_norm, text="Use Current", command=self.set_bg_from_current,
            bg="#2980B9", fg="white", font=("Arial", 8, "bold"),
        ).grid(row=3, column=2, columnspan=2, padx=5, sticky="w")

        tk.Label(f_norm, text="Bg Frame Count:").grid(row=4, column=0, padx=5, sticky="w")
        tk.Entry(f_norm, textvariable=self.bg_frames, width=6).grid(row=4, column=1, sticky="w")
        
        tk.Label(f_norm, text="Temp Window:").grid(row=4, column=2, padx=5, sticky="w")
        tk.Entry(f_norm, textvariable=self.window_size, width=6).grid(row=4, column=3, sticky="w")

        tk.Label(
            f_norm,
            text="X-ray Enhancement (CLAHE + Local Contrast):",
            font=("Arial", 9, "bold"),
        ).grid(row=5, column=0, columnspan=4, padx=5, sticky="w", pady=(8, 0))

        tk.Button(
            f_norm,
            text="✨ CLAHE + Local Contrast",
            command=self.apply_clahe_local_contrast_quick,
            bg="#16A085",
            fg="white",
            font=("Arial", 9, "bold"),
        ).grid(row=6, column=0, columnspan=4, padx=5, pady=4, sticky="ew")

        tk.Checkbutton(
            f_norm,
            text="Enable CLAHE",
            variable=self.norm_clahe,
            command=self.on_enhancement_change,
        ).grid(row=7, column=0, columnspan=2, sticky="w", padx=5)
        tk.Label(f_norm, text="Clip:").grid(row=7, column=2, padx=5, sticky="e")
        tk.Entry(f_norm, textvariable=self.norm_clahe_clip, width=6).grid(row=7, column=3, sticky="w")
        tk.Label(f_norm, text="Tile size:").grid(row=8, column=0, padx=5, sticky="w")
        tk.Entry(f_norm, textvariable=self.norm_clahe_tile, width=6).grid(row=8, column=1, sticky="w")

        tk.Checkbutton(
            f_norm,
            text="Enable Local Contrast",
            variable=self.local_contrast_enabled,
            command=self.on_enhancement_change,
        ).grid(row=9, column=0, columnspan=2, sticky="w", padx=5, pady=(4, 0))
        tk.Label(f_norm, text="Strength:").grid(row=10, column=0, padx=5, sticky="w")
        tk.Entry(f_norm, textvariable=self.local_contrast_strength, width=6).grid(row=10, column=1, sticky="w")
        tk.Label(f_norm, text="Radius:").grid(row=10, column=2, padx=5, sticky="e")
        tk.Entry(f_norm, textvariable=self.local_contrast_radius, width=6).grid(row=10, column=3, sticky="w")

        # 3. Visual Tools
        f_tools = tk.LabelFrame(left_col, text="3. Visual Adjustments (Original Only)", font=("Arial", 10, "bold"), bg="#EAECEE", fg="#C0392B")
        f_tools.pack(fill="x", pady=5)
        
        tk.Label(f_tools, text="⚠ NOTE: Levels/gamma/gain/brightness do NOT affect Normalization.", 
                 bg="#EAECEE", fg="red", font=("Arial", 9, "bold")).pack(anchor="w", padx=5, pady=(2,2))
        tk.Label(f_tools, text="CLAHE + Local Contrast (section 2) apply to both previews/export.", 
                 bg="#EAECEE", fg="#7F8C8D", font=("Arial", 8)).pack(anchor="w", padx=5, pady=(0,5))
        
        tk.Checkbutton(f_tools, text="Auto-Levels", variable=self.vis_auto, command=self.update_vis_mode, bg="#EAECEE").pack(anchor="w", padx=5)

        flip_row = tk.Frame(f_tools, bg="#EAECEE")
        flip_row.pack(fill="x", padx=5, pady=(4, 6))
        self.btn_flip_h = tk.Button(
            flip_row, text="Flip Horizontal: OFF",
            command=self.toggle_flip_h,
            bg="#34495E", fg="white", font=("Arial", 9, "bold"), width=18,
        )
        self.btn_flip_h.pack(side="left", padx=(0, 6))
        self.btn_flip_v = tk.Button(
            flip_row, text="Flip Vertical: OFF",
            command=self.toggle_flip_v,
            bg="#34495E", fg="white", font=("Arial", 9, "bold"), width=18,
        )
        self.btn_flip_v.pack(side="left")
        
        self.f_sliders = tk.Frame(f_tools, bg="#EAECEE")
        self.f_sliders.pack(fill="x", padx=5, pady=2)
        
        def make_slider(parent, label, var, vmin, vmax, res, row):
            tk.Label(parent, text=label, bg="#EAECEE", width=10, anchor='w').grid(row=row, column=0, sticky="w")
            s = tk.Scale(parent, variable=var, from_=vmin, to=vmax, resolution=res, orient="horizontal", bg="#EAECEE", length=220, showvalue=False, command=lambda x: self.refresh_preview_visuals())
            s.grid(row=row, column=1)
            tk.Entry(parent, textvariable=var, width=5).grid(row=row, column=2, padx=5)

        make_slider(self.f_sliders, "Levels Min:", self.vis_min, 0, 65535, 1, 0)
        make_slider(self.f_sliders, "Levels Max:", self.vis_max, 1, 65535, 1, 1)        
        make_slider(self.f_sliders, "Gamma:", self.vis_gamma, 0.1, 5.0, 0.1, 2)
        make_slider(self.f_sliders, "Gain:", self.vis_gain, 0.1, 10.0, 0.1, 3)
        make_slider(self.f_sliders, "Brightness:", self.vis_bright, -100, 100, 1, 4)

        # 4. Histogram
        f_hist = tk.LabelFrame(left_col, text="Pixel Histogram", font=("Arial", 10, "bold"))
        f_hist.pack(fill="x", pady=2)
        self.fig, self.ax = plt.subplots(figsize=(4, 3), dpi=80)
        self.fig.patch.set_facecolor('#F0F0F0')
        self.canvas_hist = FigureCanvasTkAgg(self.fig, master=f_hist)
        self.canvas_hist.get_tk_widget().pack(fill="both", expand=True)

        # 5. Run
        f_run = tk.LabelFrame(left_col, text="5. Execution", font=("Arial", 10, "bold"), fg="#2C3E50")
        f_run.pack(fill="x", pady=5)
        r1 = tk.Frame(f_run); r1.pack(fill="x", padx=5)
        tk.Label(r1, text="Save Frames:").pack(side="left")
        self.ent_start_frame = tk.Entry(r1, textvariable=self.start_frame, width=6)
        self.ent_start_frame.pack(side="left")
        tk.Label(r1, text="to").pack(side="left", padx=2)
        tk.Entry(r1, textvariable=self.end_frame, width=6).pack(side="left")
        tk.Label(r1, text="(inclusive)", font=("Arial", 8), fg="#7F8C8D").pack(side="left", padx=2)

        r1b = tk.Frame(f_run); r1b.pack(fill="x", padx=5, pady=(2, 0))
        tk.Checkbutton(r1b, text="Export All", variable=self.export_all,
                       command=self._update_export_range_ui).pack(side="left")
        tk.Checkbutton(
            r1b, text="Start after BG frames", variable=self.start_after_bg,
            command=self._update_export_range_ui,
        ).pack(side="left", padx=5)

        r1c = tk.Frame(f_run); r1c.pack(fill="x", padx=5, pady=(2, 0))
        tk.Button(
            r1c, text="Set Start = Current", command=self.set_start_from_current,
            bg="#34495E", fg="white", font=("Arial", 8, "bold"),
        ).pack(side="left", padx=(0, 4))
        tk.Button(
            r1c, text="Set End = Current", command=self.set_end_from_current,
            bg="#34495E", fg="white", font=("Arial", 8, "bold"),
        ).pack(side="left", padx=4)
        tk.Button(
            r1c, text="Start after BG", command=self.set_start_after_bg,
            bg="#16A085", fg="white", font=("Arial", 8, "bold"),
        ).pack(side="left", padx=4)
        
        # --- Checkbox UI ---
        r2 = tk.Frame(f_run); r2.pack(fill="x", padx=5, pady=(0,5))
        tk.Checkbutton(r2, text="Save Enhanced Imgs (PNG)", variable=self.save_vis_imgs, 
                       font=("Arial", 9, "bold"), fg="#C0392B").pack(side="left", padx=5)
        # ------------------------
        
        self.btn_run = tk.Button(left_col, text="▶ START PROCESSING", bg="#16A085", fg="white", font=("Arial", 12, "bold"), height=2, command=self.start_processing)
        self.btn_run.pack(fill="x", pady=10)
        
        self.log_area = scrolledtext.ScrolledText(left_col, height=6, bg="#F4F6F6", font=("Consolas", 9))
        self.log_area.pack(fill="both", expand=True)

        # ================= RIGHT COLUMN =================
        f_prev_ctrl = tk.Frame(right_col, bg="#D5D8DC", pady=5)
        f_prev_ctrl.pack(fill="x")
        
        tk.Button(f_prev_ctrl, text="◀", command=self.prev_frame, bg="#34495E", fg="white", font=("Arial", 10, "bold")).pack(side="left", padx=(10, 2))
        
        tk.Label(f_prev_ctrl, text="Frame:", bg="#D5D8DC", font=("Arial", 10, "bold")).pack(side="left", padx=5)
        # Allow pressing Enter in the frame box to jump
        self.ent_frame_idx = tk.Entry(f_prev_ctrl, textvariable=self.preview_frame_idx, width=8)
        self.ent_frame_idx.pack(side="left")
        self.ent_frame_idx.bind('<Return>', lambda e: self.generate_preview()) 
        
        tk.Button(f_prev_ctrl, text="▶", command=self.next_frame, bg="#34495E", fg="white", font=("Arial", 10, "bold")).pack(side="left", padx=(2, 10))
        
        tk.Label(f_prev_ctrl, text="Frame Index:", bg="#D5D8DC", font=("Arial", 10, "bold")).pack(side="left", padx=10)
        tk.Entry(f_prev_ctrl, textvariable=self.preview_frame_idx, width=8).pack(side="left")
        tk.Button(f_prev_ctrl, text="Show", command=lambda: self.generate_preview(random_pick=False), bg="#2ECC71", fg="white").pack(side="left", padx=5)
        tk.Label(f_prev_ctrl, text="|", bg="#D5D8DC").pack(side="left", padx=5)
        tk.Button(f_prev_ctrl, text="🎲 Random", command=lambda: self.generate_preview(random_pick=True), bg="#8E44AD", fg="white").pack(side="left", padx=5)
        
        self.lbl_status_monitor = tk.Label(f_prev_ctrl, text="Ready", bg="#D5D8DC")
        self.lbl_status_monitor.pack(side="right", padx=10)
        
        f_list = tk.Frame(right_col, bg="#ECF0F1", height=150)
        f_list.pack(fill="x", pady=(0, 5))
        f_list.pack_propagate(False) # Fix the height
        
        sb = tk.Scrollbar(f_list)
        sb.pack(side="right", fill="y")
        
        self.file_listbox = tk.Listbox(f_list, yscrollcommand=sb.set, font=("Consolas", 9), selectmode="browse", bg="white")
        self.file_listbox.pack(side="left", fill="both", expand=True)
        
        sb.config(command=self.file_listbox.yview)
        # Bind selection event
        self.file_listbox.bind("<<ListboxSelect>>", self.on_file_select)
        
        f_imgs = tk.Frame(right_col)
        f_imgs.pack(fill="both", expand=True)
        
        tk.Label(f_imgs, text="Original Input (Visually Enhanced)", font=("Arial", 10, "bold"), fg="#2980B9").pack(pady=(5,0))
        self.lbl_preview_orig = tk.Label(f_imgs, bg="black", width=60, height=20)
        self.lbl_preview_orig.pack(pady=5)
        
        tk.Label(f_imgs, text="Normalized Output (Analysis Ready)", font=("Arial", 10, "bold"), fg="#D35400").pack(pady=(5,0))
        self.lbl_preview_norm = tk.Label(f_imgs, bg="black", width=60, height=20)
        self.lbl_preview_norm.pack(pady=5)


        self.update_vis_mode()
        self.update_mode_ui()
        self._update_export_range_ui()
        self._update_flip_button_styles()

    # --- HELPERS: FRAME RANGE / BG SELECTION ---
    def _total_frame_count(self):
        return int(self.file_listbox.size())

    def _bg_index_range(self, total_frames):
        """Return inclusive-start / exclusive-end indices used for background averaging."""
        if total_frames <= 0:
            return 0, 0
        bg_start = max(0, min(int(self.bg_start_frame.get()), total_frames - 1))
        bg_count = max(1, int(self.bg_frames.get()))
        bg_end = min(total_frames, bg_start + bg_count)
        return bg_start, bg_end

    def _process_index_range(self, total_frames):
        """Return inclusive-start / exclusive-end indices for frames to save/process."""
        if total_frames <= 0:
            return 0, 0
        if self.export_all.get():
            return 0, total_frames

        bg_start, bg_end = self._bg_index_range(total_frames)
        if self.start_after_bg.get():
            start = bg_end
        else:
            start = max(0, min(int(self.start_frame.get()), total_frames - 1))

        end_inclusive = max(0, min(int(self.end_frame.get()), total_frames - 1))
        end = end_inclusive + 1  # convert inclusive UI value to exclusive slice end
        if start >= end:
            return start, start
        return start, end

    def set_bg_from_current(self):
        total = self._total_frame_count()
        if total <= 0:
            messagebox.showwarning("Background", "Load images first.")
            return
        idx = max(0, min(int(self.preview_frame_idx.get()), total - 1))
        self.bg_start_frame.set(idx)
        self.export_all.set(False)
        if self.start_after_bg.get():
            self.set_start_after_bg(silent=True)
        self._update_export_range_ui()
        self.log(f"Background start set to frame {idx} (count={self.bg_frames.get()}).")
        self.apply_norm_preview()

    def set_start_from_current(self):
        total = self._total_frame_count()
        if total <= 0:
            messagebox.showwarning("Range", "Load images first.")
            return
        idx = max(0, min(int(self.preview_frame_idx.get()), total - 1))
        self.start_frame.set(idx)
        self.start_after_bg.set(False)
        self.export_all.set(False)
        self._update_export_range_ui()
        self.log(f"Save start set to frame {idx}.")

    def set_end_from_current(self):
        total = self._total_frame_count()
        if total <= 0:
            messagebox.showwarning("Range", "Load images first.")
            return
        idx = max(0, min(int(self.preview_frame_idx.get()), total - 1))
        self.end_frame.set(idx)
        self.export_all.set(False)
        self._update_export_range_ui()
        self.log(f"Save end set to frame {idx}.")

    def set_start_after_bg(self, silent=False):
        total = self._total_frame_count()
        if total <= 0:
            if not silent:
                messagebox.showwarning("Range", "Load images first.")
            return
        bg_start, bg_end = self._bg_index_range(total)
        # First frame after the background block (clamped if BG reaches the end)
        start = bg_end if bg_end < total else max(0, total - 1)
        self.start_frame.set(start)
        self.start_after_bg.set(True)
        self.export_all.set(False)
        self._update_export_range_ui()
        if not silent:
            self.log(
                f"Save start set to frame {start} "
                f"(after BG frames {bg_start}-{bg_end - 1})."
            )

    def _update_export_range_ui(self):
        if not hasattr(self, "ent_start_frame"):
            return
        if self.export_all.get():
            self.ent_start_frame.config(state="disabled")
            return
        if self.start_after_bg.get():
            total = self._total_frame_count()
            if total > 0:
                _, bg_end = self._bg_index_range(total)
                start = bg_end if bg_end < total else max(0, total - 1)
                self.start_frame.set(start)
            self.ent_start_frame.config(state="disabled")
        else:
            self.ent_start_frame.config(state="normal")

    # --- HELPERS: SAFE IO ---
    def _safe_imread(self, path):
        """Read grayscale image preserving bit depth (16-bit TIFF safe)."""
        try:
            path = os.path.abspath(path)
            stream = np.fromfile(path, dtype=np.uint8)
            img = cv2.imdecode(stream, cv2.IMREAD_UNCHANGED)
            if img is None:
                return None
            if img.ndim == 3:
                if img.shape[2] >= 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                else:
                    img = img[:, :, 0]
            return img.astype(np.float32)
        except Exception:
            return None

    def _safe_imwrite(self, path, img):
        try:
            path = os.path.abspath(path)
            if len(path) > 259 and os.name == 'nt' and not path.startswith('\\\\?\\'):
                path = '\\\\?\\' + path
            ext = os.path.splitext(path)[1]
            result, n = cv2.imencode(ext, img)
            if result:
                with open(path, mode='wb') as f:
                    n.tofile(f)
            return True
        except Exception as e:
            print(f"WRITE ERROR: {e}")
            return False

    def update_vis_mode(self):
        state = "disabled" if self.vis_auto.get() else "normal"
        for child in self.f_sliders.winfo_children():
            try: child.configure(state=state)
            except: pass
        self.refresh_preview_visuals()

    def _apply_xray_enhancement(self, img):
        """Apply CLAHE then local contrast (ver05 order). Works on float or uint8."""
        out = img.astype(np.float32)
        if self.norm_clahe.get():
            out = apply_clahe_float(out, self.norm_clahe_clip.get(), self.norm_clahe_tile.get())
        if self.local_contrast_enabled.get():
            out = apply_local_contrast(
                out, self.local_contrast_strength.get(), self.local_contrast_radius.get()
            )
        return out

    def _apply_norm_clahe(self, img_u8):
        """Legacy name: apply full x-ray enhancement on an 8-bit normalized result."""
        out = self._apply_xray_enhancement(img_u8)
        return np.clip(out, 0, 255).astype(np.uint8)

    def apply_clahe_local_contrast_quick(self):
        """One-click CLAHE + local contrast defaults (same as ver05)."""
        if self.flat_master is None:
            messagebox.showinfo(
                "Enhancement",
                "Load a flat-field first for best results (optional but recommended).",
            )
        self.norm_clahe.set(True)
        self.norm_clahe_clip.set(2.0)
        self.norm_clahe_tile.set(8)
        self.local_contrast_enabled.set(True)
        self.local_contrast_strength.set(1.5)
        self.local_contrast_radius.set(15)
        self.log("CLAHE + local contrast enabled (clip=2.0, tile=8, strength=1.5, radius=15).")
        self.on_enhancement_change()

    def on_enhancement_change(self):
        if self.current_preview_raw is not None:
            self.refresh_preview_visuals()
            if self.current_preview_bg is not None:
                self._render_norm_preview(self.current_preview_raw, self.current_preview_bg)

    def on_flatfield_toggle(self):
        if self.use_flatfield.get() and self.flat_master is None:
            if self.flat_path.get().strip():
                self.load_flatfield(ask_if_missing=False)
            else:
                messagebox.showinfo("Flat-field", "Browse and select a flat-field file first.")
                self.use_flatfield.set(False)
                return
        self.generate_preview()

    def _apply_flips(self, frame):
        if frame is None:
            return None
        out = frame
        if self.flip_h.get():
            out = np.ascontiguousarray(np.fliplr(out))
        if self.flip_v.get():
            out = np.ascontiguousarray(np.flipud(out))
        return out

    def _prepare_frame(self, frame):
        """Flat-field (optional) then horizontal/vertical flips."""
        frame = self._maybe_apply_flat(frame)
        return self._apply_flips(frame)

    def _update_flip_button_styles(self):
        if hasattr(self, "btn_flip_h"):
            on = self.flip_h.get()
            self.btn_flip_h.config(
                text=f"Flip Horizontal: {'ON' if on else 'OFF'}",
                bg="#27AE60" if on else "#34495E",
            )
        if hasattr(self, "btn_flip_v"):
            on = self.flip_v.get()
            self.btn_flip_v.config(
                text=f"Flip Vertical: {'ON' if on else 'OFF'}",
                bg="#27AE60" if on else "#34495E",
            )

    def toggle_flip_h(self):
        self.flip_h.set(not self.flip_h.get())
        self._update_flip_button_styles()
        self.generate_preview()

    def toggle_flip_v(self):
        self.flip_v.set(not self.flip_v.get())
        self._update_flip_button_styles()
        self.generate_preview()

    def _maybe_apply_flat(self, frame):
        if frame is None:
            return None
        if not self.use_flatfield.get() or self.flat_master is None:
            return frame
        try:
            return apply_flat_division(frame, self.flat_master, self.flat_norm_strength.get())
        except Exception as e:
            self.log(f"Flat-field apply failed: {e}")
            return frame

    def _read_frame_raw(self, idx, inp=None):
        """Read one raw frame by index from CINE / MRAW / image folder."""
        inp = inp or self.input_path.get()
        if self.is_mraw_mode:
            return read_mraw_frame(
                inp, idx, self.mraw_frame_bytes, self.mraw_rows, self.mraw_cols
            )
        if self.is_cine_mode:
            if self.cine_video is None:
                return None
            return np.asarray(self.cine_video[idx], dtype=np.float32)
        if self.image_files and 0 <= idx < len(self.image_files):
            return self._safe_imread(self.image_files[idx])
        if self.file_listbox.size() > 0 and 0 <= idx < self.file_listbox.size():
            fname = self.file_listbox.get(idx)
            return self._safe_imread(os.path.join(inp, fname))
        return None

    def _load_preview_background(self, inp):
        total_frames = self.file_listbox.size()
        if total_frames == 0:
            return None

        bg_start, bg_end = self._bg_index_range(total_frames)
        if bg_start >= bg_end:
            return None

        bg_data_list = []
        for i in range(bg_start, bg_end):
            im = self._read_frame_raw(i, inp)
            if im is not None:
                bg_data_list.append(self._prepare_frame(im))
        if not bg_data_list:
            return None
        return np.mean(np.array(bg_data_list).astype(np.float32), axis=0)

    def _using_flatfield(self):
        return bool(self.use_flatfield.get() and self.flat_master is not None)

    def _flat_corrected_to_u8(self, frame):
        """Display/export mapping for flat-corrected frames (no grey-scale temporal norm)."""
        img = np.asarray(frame, dtype=np.float32)
        low = float(np.percentile(img, 0.1))
        high = float(np.percentile(img, 99.9))
        if high <= low:
            high = low + 1.0
        u8 = np.clip((img - low) / (high - low) * 255.0, 0, 255).astype(np.uint8)
        return self._apply_norm_clahe(u8)

    def _render_norm_preview(self, target_raw, bg):
        # Flat-field already removes background — skip grey-scale temporal normalization.
        if self._using_flatfield():
            res = self._flat_corrected_to_u8(target_raw)
            self.show_on_label(self.lbl_preview_norm, res)
            return

        target_blur = cv2.GaussianBlur(target_raw, (3, 3), 0)
        bg_blur = cv2.GaussianBlur(bg, (3, 3), 0)
        correction = np.mean(bg_blur) / (np.mean(target_blur) + 1e-6)
        norm = (target_blur * correction) / (bg_blur + 1e-6)

        if self.norm_mode.get() == "grey":
            min_r = self.min_ratio.get()
            max_r = self.max_ratio.get()
            res = np.clip(norm, min_r, max_r)
            res = ((res - min_r) * (255.0 / (max_r - min_r))).astype(np.uint8)
        else:
            mask = np.abs(norm - 1.0) > self.binary_thresh.get()
            res = (mask * 255).astype(np.uint8)

        res = self._apply_norm_clahe(res)
        self.show_on_label(self.lbl_preview_norm, res)

    def apply_norm_preview(self):
        inp = self.input_path.get().strip()
        if not inp or not os.path.exists(inp):
            messagebox.showwarning("Input", "Load images first.")
            return
        if self.current_preview_raw is None:
            self.generate_preview()
            return

        if self._using_flatfield():
            self._render_norm_preview(self.current_preview_raw, None)
            return

        bg = self._load_preview_background(inp)
        if bg is None:
            messagebox.showwarning("Background", "Could not build background for preview.")
            return

        self.current_preview_bg = bg
        self._render_norm_preview(self.current_preview_raw, bg)

    def update_mode_ui(self):
        mode = self.norm_mode.get()
        if hasattr(self, 'ent_bin'):
            if mode == "grey":
                self.ent_min.config(state="normal")
                self.ent_max.config(state="normal")
                self.ent_bin.config(state="disabled")
            else:
                self.ent_min.config(state="disabled")
                self.ent_max.config(state="disabled")
                self.ent_bin.config(state="normal")

    def go_back(self):
        self.frame.destroy()
        self.on_back()

    def load_7bm_experiment(self, experiment_folder: str, flatfield_folder: str = None):
        """Load a 7BM run folder (.mraw inside) and apply the batch flat-field."""
        try:
            mraw_path, _label = resolve_input_to_mraw(experiment_folder)
        except FileNotFoundError as e:
            messagebox.showerror("7BM load", str(e))
            return

        self.input_path.set(mraw_path)
        self.load_input_source(auto_flatfield=False)

        flat_folder = flatfield_folder
        if not flat_folder or not os.path.isdir(flat_folder):
            if os.path.isdir(DEFAULT_7BM_FLATFIELD_FOLDER):
                flat_folder = DEFAULT_7BM_FLATFIELD_FOLDER
            else:
                flat_folder = find_sibling_flatfield_folder(mraw_path)

        if flat_folder and os.path.isdir(flat_folder):
            try:
                flat_mraw = find_mraw_in_folder(flat_folder)
                self.flat_path.set(flat_mraw)
                if self.load_flatfield(ask_if_missing=False):
                    self.log(f"7BM: flat-field from {flat_folder}")
            except FileNotFoundError as e:
                messagebox.showwarning("Flat-field", f"No flat-field .mraw found:\n{e}")
        elif self.flat_master is None:
            if messagebox.askyesno(
                "Flat-field",
                "7BM experiment loaded.\n\nSelect flat-field MRAW now?",
            ):
                self.browse_flatfield()

    def _try_resolve_mraw_folder(self, inp: str) -> str:
        """If inp is an experiment folder, return the .mraw path inside it."""
        if not os.path.isdir(inp):
            return inp
        try:
            mraw = find_mraw_in_folder(inp)
            self.log(f"Using MRAW: {os.path.basename(mraw)}")
            return mraw
        except FileNotFoundError:
            return inp

    def load_input_source(self, auto_flatfield=True):
        """Loads CINE / MRAW / image folder and populates the listbox."""
        inp = self.input_path.get().strip()
        if not inp or not os.path.exists(inp):
            messagebox.showwarning("Input", "Select or enter a valid input path first.")
            return

        inp = os.path.abspath(inp)

        self.is_cine_mode = False
        self.is_mraw_mode = False
        if os.path.isdir(inp):
            inp = self._try_resolve_mraw_folder(inp)
        elif inp.lower().endswith(".cine"):
            self.is_cine_mode = True
        elif inp.lower().endswith(".mraw"):
            self.is_mraw_mode = True

        self.input_path.set(inp)

        self.file_listbox.delete(0, tk.END)
        self.preview_frame_idx.set(0)
        self.image_files = []
        self.mraw_rows = self.mraw_cols = self.mraw_frame_bytes = 0

        if self.cine_video:
            try:
                self.cine_video.close()
            except Exception:
                pass
            self.cine_video = None

        if self.is_mraw_mode:
            try:
                rows, cols, count, bits, fps, frame_bytes = open_mraw_info(inp)
                self.mraw_rows = rows
                self.mraw_cols = cols
                self.mraw_frame_bytes = frame_bytes
                self.mraw_bits = bits
                self.fps.set(float(fps))
                for i in range(count):
                    self.file_listbox.insert(tk.END, f"Frame {i}")
                self.log(
                    f"Loaded MRAW: {count} frames, {cols}x{rows}, "
                    f"{bits}-bit, {fps} fps"
                )
            except Exception as e:
                self.log(f"Error opening MRAW: {e}")
                messagebox.showerror("MRAW error", f"Could not open MRAW file:\n{e}")
                self.is_mraw_mode = False
                return

        elif self.is_cine_mode:
            try:
                self.cine_video = pims.open(inp)
                count = len(self.cine_video)
                for i in range(count):
                    self.file_listbox.insert(tk.END, f"Frame {i}")
                self.log(f"Loaded CINE: {count} frames.")
            except Exception as e:
                self.log(f"Error opening CINE: {e}")
                messagebox.showerror("CINE error", f"Could not open CINE file:\n{e}")
                self.is_cine_mode = False
                return

        else:
            self.image_files = collect_image_paths(inp, recursive=False)
            for f in self.image_files:
                self.file_listbox.insert(tk.END, os.path.basename(f))
            self.log(f"Loaded folder: {len(self.image_files)} image(s) from\n{inp}")
            if not self.image_files:
                messagebox.showerror(
                    "No images found",
                    "No images found in that folder.\n\n"
                    "Supported formats: .tif, .tiff, .png, .jpg, .jpeg, .bmp\n"
                    "Or load a .cine / .mraw file.",
                )
                return

        if self.file_listbox.size() > 0:
            self.bg_start_frame.set(0)
            self.end_frame.set(max(0, self.file_listbox.size() - 1))
            if self.start_after_bg.get():
                self.set_start_after_bg(silent=True)
            self._update_export_range_ui()
            self.generate_preview()

        if auto_flatfield and self.is_mraw_mode and self.flat_master is None:
            auto_flat = ""
            if os.path.isdir(DEFAULT_7BM_FLATFIELD_FOLDER):
                auto_flat = DEFAULT_7BM_FLATFIELD_FOLDER
            else:
                auto_flat = find_sibling_flatfield_folder(inp)
            if auto_flat:
                try:
                    self.flat_path.set(find_mraw_in_folder(auto_flat))
                    if self.load_flatfield(ask_if_missing=False):
                        self.log(f"Auto flat-field: {auto_flat}")
                        return
                except FileNotFoundError:
                    pass
            if messagebox.askyesno(
                "Flat-field",
                "MRAW loaded.\n\nSelect a flat-field file now?",
            ):
                self.browse_flatfield()

    def browse_input(self):
        dialog = Toplevel(self.frame)
        dialog.title("Input Type")
        dialog.geometry("420x280")
        dialog.resizable(False, False)
        dialog.configure(bg="white")
        dialog.transient(self.frame.winfo_toplevel())
        dialog.grab_set()
        choice = {"v": None}

        tk.Label(
            dialog,
            text="What are you loading?",
            font=("Arial", 12, "bold"),
            bg="white",
        ).pack(pady=(18, 10))

        def pick(v):
            choice["v"] = v
            dialog.destroy()

        btn = {"font": ("Arial", 10, "bold"), "width": 28, "height": 2}
        tk.Button(dialog, text="CINE file (.cine)", bg="#2980B9", fg="white",
                  command=lambda: pick("cine"), **btn).pack(pady=4)
        tk.Button(dialog, text="MRAW file (.mraw)", bg="#8E44AD", fg="white",
                  command=lambda: pick("mraw"), **btn).pack(pady=4)
        tk.Button(dialog, text="7BM experiment folder (.mraw inside)", bg="#1F618D", fg="white",
                  command=lambda: pick("7bm_folder"), **btn).pack(pady=4)
        tk.Button(dialog, text="Folder of images", bg="#16A085", fg="white",
                  command=lambda: pick("folder"), **btn).pack(pady=4)
        tk.Button(dialog, text="Cancel", bg="#95A5A6", fg="white",
                  command=lambda: pick(None), width=12).pack(pady=8)
        self.frame.wait_window(dialog)

        path = ""
        if choice["v"] == "cine":
            path = filedialog.askopenfilename(filetypes=[("CINE", "*.cine"), ("All", "*.*")])
            self.is_cine_mode = True
            self.is_mraw_mode = False
        elif choice["v"] == "mraw":
            path = filedialog.askopenfilename(filetypes=[("MRAW", "*.mraw"), ("All", "*.*")])
            self.is_mraw_mode = True
            self.is_cine_mode = False
        elif choice["v"] == "7bm_folder":
            path = filedialog.askdirectory(title="7BM experiment folder (contains .mraw)")
            if path:
                self.load_7bm_experiment(path)
            return
        elif choice["v"] == "folder":
            path = filedialog.askdirectory()
            self.is_cine_mode = False
            self.is_mraw_mode = False
        else:
            return

        if path:
            self.input_path.set(os.path.abspath(path))
            self.load_input_source()

    def browse_flatfield(self):
        dialog = Toplevel(self.frame)
        dialog.title("Flat-field Type")
        dialog.geometry("420x200")
        dialog.resizable(False, False)
        dialog.configure(bg="white")
        dialog.transient(self.frame.winfo_toplevel())
        dialog.grab_set()
        choice = {"v": None}

        tk.Label(
            dialog,
            text="Select flat-field source",
            font=("Arial", 12, "bold"),
            bg="white",
        ).pack(pady=(18, 10))

        def pick(v):
            choice["v"] = v
            dialog.destroy()

        btn = {"font": ("Arial", 10, "bold"), "width": 28, "height": 2}
        tk.Button(dialog, text="Flat-field MRAW (.mraw)", bg="#8E44AD", fg="white",
                  command=lambda: pick("mraw"), **btn).pack(pady=4)
        tk.Button(dialog, text="Flat-field image / folder", bg="#16A085", fg="white",
                  command=lambda: pick("image"), **btn).pack(pady=4)
        tk.Button(dialog, text="Cancel", bg="#95A5A6", fg="white",
                  command=lambda: pick(None), width=12).pack(pady=8)
        self.frame.wait_window(dialog)

        path = ""
        if choice["v"] == "mraw":
            path = filedialog.askopenfilename(filetypes=[("MRAW", "*.mraw"), ("All", "*.*")])
        elif choice["v"] == "image":
            path = filedialog.askopenfilename(
                filetypes=[
                    ("Images", "*.tif *.tiff *.png *.jpg *.jpeg *.bmp"),
                    ("All", "*.*"),
                ]
            )
            if not path:
                path = filedialog.askdirectory(title="Or select a folder of flat images / MRAW")
        else:
            return

        if path:
            self.flat_path.set(os.path.abspath(path))
            self.load_flatfield(ask_if_missing=False)

    def load_flatfield(self, ask_if_missing=True):
        path = self.flat_path.get().strip()
        if not path or not os.path.exists(path):
            if ask_if_missing:
                messagebox.showwarning("Flat-field", "Select a valid flat-field path first.")
            self.flat_master = None
            if hasattr(self, "lbl_flat_status"):
                self.lbl_flat_status.config(text="(none)", fg="#7F8C8D")
            return False

        path = os.path.abspath(path)
        self.flat_path.set(path)
        try:
            if os.path.isdir(path):
                try:
                    path = find_mraw_in_folder(path)
                    self.flat_path.set(path)
                except FileNotFoundError:
                    files = collect_image_paths(path, recursive=False)
                    if not files:
                        raise ValueError("Folder has no .mraw or flat-field images.")
                    imgs = []
                    for f in files[:50]:
                        im = self._safe_imread(f)
                        if im is not None:
                            imgs.append(im)
                    if not imgs:
                        raise ValueError("Could not read flat-field image(s).")
                    self.flat_master = np.maximum(np.mean(np.array(imgs), axis=0).astype(np.float32), 1.0)
                    self.use_flatfield.set(True)
                    if hasattr(self, "lbl_flat_status"):
                        self.lbl_flat_status.config(
                            text=f"Ready {self.flat_master.shape[1]}x{self.flat_master.shape[0]}",
                            fg="#27AE60",
                        )
                    self.log(f"Flat-field loaded: {self.flat_path.get()}")
                    self.generate_preview()
                    return True

            if path.lower().endswith(".mraw"):
                if self.is_mraw_mode and self.mraw_frame_bytes > 0:
                    rows, cols, frame_bytes = self.mraw_rows, self.mraw_cols, self.mraw_frame_bytes
                    max_frames = max(1, min(50, self._total_frame_count() or 50))
                else:
                    rows, cols, n_frames, _bits, _fps, frame_bytes = open_mraw_info(path)
                    max_frames = max(1, min(50, n_frames))
                self.flat_master = compute_flat_master_mraw(
                    path, frame_bytes, rows, cols, max_frames=max_frames
                )
                if self.is_mraw_mode and self.mraw_rows > 0:
                    if self.flat_master.shape != (self.mraw_rows, self.mraw_cols):
                        raise ValueError(
                            f"Flat-field size {self.flat_master.shape[1]}x{self.flat_master.shape[0]} "
                            f"does not match data {self.mraw_cols}x{self.mraw_rows}."
                        )
            else:
                files = collect_image_paths(path, recursive=False)
                if not files:
                    raise ValueError("No flat-field images found at that path.")
                imgs = []
                for f in files[:50]:
                    im = self._safe_imread(f)
                    if im is not None:
                        imgs.append(im)
                if not imgs:
                    raise ValueError("Could not read flat-field image(s).")
                self.flat_master = np.maximum(np.mean(np.array(imgs), axis=0).astype(np.float32), 1.0)

            self.use_flatfield.set(True)
            if hasattr(self, "lbl_flat_status"):
                self.lbl_flat_status.config(
                    text=f"Ready {self.flat_master.shape[1]}x{self.flat_master.shape[0]}",
                    fg="#27AE60",
                )
            self.log(f"Flat-field loaded: {path}")
            self.generate_preview()
            return True
        except Exception as e:
            self.flat_master = None
            self.use_flatfield.set(False)
            if hasattr(self, "lbl_flat_status"):
                self.lbl_flat_status.config(text="failed", fg="#C0392B")
            messagebox.showerror("Flat-field", f"Could not load flat-field:\n{e}")
            self.log(f"Flat-field error: {e}")
            return False

    def prev_frame(self):
        idx = self.preview_frame_idx.get()
        if idx > 0:
            self.preview_frame_idx.set(idx - 1)
            self.generate_preview()

    def next_frame(self):
        idx = self.preview_frame_idx.get()
        total = self.file_listbox.size()
        if idx < total - 1:
            self.preview_frame_idx.set(idx + 1)
            self.generate_preview()

    def on_file_select(self, event):
        """Handles clicks in the listbox."""
        sel = self.file_listbox.curselection()
        if sel:
            idx = sel[0]
            self.preview_frame_idx.set(idx)
            # Tell generate_preview not to re-sync the listbox to avoid loops
            self.generate_preview(from_listbox=True)

    def sync_listbox(self):
        """Keeps the listbox selection in sync with the current frame index."""
        if self.file_listbox.size() > 0:
            idx = self.preview_frame_idx.get()
            self.file_listbox.selection_clear(0, tk.END)
            if 0 <= idx < self.file_listbox.size():
                self.file_listbox.selection_set(idx)
                self.file_listbox.see(idx) # Ensure it's visible

    def browse_folder(self, var):
        path = filedialog.askdirectory()
        if path: var.set(os.path.abspath(path))

    def log(self, msg):
        self.log_area.config(state='normal')
        self.log_area.insert(tk.END, msg + "\n")
        self.log_area.see(tk.END)
        self.log_area.config(state='disabled')

    def update_histogram(self, data):
        if data is None: return
        self.ax.clear()
        flat_data = data.flatten()[::10]
        self.ax.hist(flat_data, bins=50, color='gray', alpha=0.7)
        self.ax.set_title(f"Range: {np.min(data):.0f} - {np.max(data):.0f}", fontsize=8)
        self.ax.get_yaxis().set_visible(False)
        self.canvas_hist.draw()

    def apply_visuals(self, img_raw):
        img = self._apply_xray_enhancement(img_raw.astype(np.float32))
        
        # --- AUTO LOGIC START ---
        if self.vis_auto.get():
            low = np.percentile(img, 0.1)
            high = np.percentile(img, 99.9)
            if high <= low: high = low + 1
            
            self.vis_min.set(int(low))
            self.vis_max.set(int(high))
        
        else:
            low = self.vis_min.get()
            high = self.vis_max.get()

        img = (img - low) / (high - low)
        img = np.clip(img, 0.0, 1.0)
        
        if not self.vis_auto.get():
            gamma = self.vis_gamma.get()
            gain = self.vis_gain.get()
            bright = self.vis_bright.get() / 255.0
            
            if gamma != 1.0 and gamma > 0: img = np.power(img, 1.0 / gamma)
            img = img * gain
            img = img + bright
            img = np.clip(img, 0.0, 1.0)

        return (img * 255).astype(np.uint8)

    def refresh_preview_visuals(self, event=None):
        if self.current_preview_raw is None: return
        vis_img = self.apply_visuals(self.current_preview_raw)
        self.show_on_label(self.lbl_preview_orig, vis_img)

    def generate_preview(self, random_pick=False, from_listbox=False):
        inp = self.input_path.get()
        if not inp or not os.path.exists(inp): return
        
        total_frames = self.file_listbox.size()
        if total_frames == 0: return

        target_raw = None 
        bg = None
        
        try:
            # Determine the index to show
            idx = 0
            if random_pick:
                idx = random.randint(0, total_frames - 1)
            else:
                idx = self.preview_frame_idx.get()
                # Clamp index to valid range
                if idx < 0: idx = 0
                if idx >= total_frames: idx = total_frames - 1
            
            self.preview_frame_idx.set(idx)

            bg = None
            if not self._using_flatfield():
                bg = self._load_preview_background(inp)
                if bg is None:
                    return

            target_raw = self._read_frame_raw(idx, inp)
            if target_raw is None:
                return
            target_raw = self._prepare_frame(target_raw)

            if not from_listbox:
                self.sync_listbox()
            
            self.current_preview_raw = target_raw
            self.current_preview_bg = bg
            self.update_histogram(target_raw)
            self.refresh_preview_visuals()
            self._render_norm_preview(target_raw, bg)

        except Exception as e: print(f"Preview Error: {e}")

    def show_on_label(self, label, img_array):
        h, w = img_array.shape
        disp_w = 500
        scale = disp_w / w
        disp_h = int(h * scale)
        resized = cv2.resize(img_array, (disp_w, disp_h))
        pil_img = Image.fromarray(resized)
        tk_img = ImageTk.PhotoImage(pil_img)
        label.config(image=tk_img, width=disp_w, height=disp_h)
        label.image = tk_img

    def start_processing(self):
        if self.is_running: return
        self.is_running = True
        self.btn_run.config(state='disabled', text="Processing...", bg="gray")
        threading.Thread(target=self.run_logic, daemon=True).start()

    def run_logic(self):
        try:
            inp = os.path.abspath(self.input_path.get())
            out = os.path.abspath(self.output_path.get())
            vname = self.video_name.get()
            using_gpu = self.use_gpu.get() and HAS_GPU_NORMALIZER
            mode = self.norm_mode.get()
            
            norm_dir = os.path.join(out, "Normalized_Video")
            orig_dir = os.path.join(out, "Original_Video")
            img_dir = os.path.join(out, "Normalized_Images")
            
            vis_img_dir = os.path.join(out, "Original_Images_Enhanced")
            save_vis_imgs = self.save_vis_imgs.get()
            
            for d in [norm_dir, orig_dir, img_dir]: 
                # Long path creation check
                try: os.makedirs(d, exist_ok=True)
                except: pass
                
            norm_path = os.path.join(norm_dir, vname)
            orig_path = os.path.join(orig_dir, "Original_" + vname)
            
            if save_vis_imgs: 
                try: os.makedirs(vis_img_dir, exist_ok=True)
                except: pass

            is_cine = False
            is_mraw = False
            video = None
            files = None

            if os.path.isfile(inp) and inp.lower().endswith('.cine'):
                is_cine = True
                self.log("Opening CINE...")
                video = pims.open(inp)
                total_frames = len(video)
            elif os.path.isfile(inp) and inp.lower().endswith('.mraw'):
                is_mraw = True
                self.log("Opening MRAW...")
                if self.is_mraw_mode and self.mraw_frame_bytes > 0:
                    total_frames = self._total_frame_count()
                else:
                    rows, cols, total_frames, bits, fps, frame_bytes = open_mraw_info(inp)
                    self.mraw_rows, self.mraw_cols = rows, cols
                    self.mraw_frame_bytes = frame_bytes
                    self.mraw_bits = bits
                    self.fps.set(float(fps))
            else:
                files = self.image_files or collect_image_paths(inp, recursive=False)
                if not files:
                    self.log("Error: No images found in input folder.")
                    return
                total_frames = len(files)

            bg_start, bg_end = self._bg_index_range(total_frames)
            start, end = self._process_index_range(total_frames)
            if start >= end:
                self.log(
                    f"Error: Nothing to save. Process range is empty "
                    f"(start={start}, end inclusive={max(0, end - 1)}). "
                    f"BG uses frames {bg_start}-{max(bg_start, bg_end - 1)}."
                )
                return

            use_flat = self.use_flatfield.get() and self.flat_master is not None
            skip_grey_norm = use_flat  # flat-field replaces grey-scale temporal normalization

            if skip_grey_norm:
                self.log("Flat-field ON: skipping grey-scale temporal background normalization.")
            else:
                self.log(
                    f"Background frames: {bg_start} to {bg_end - 1} "
                    f"({bg_end - bg_start} frame(s))"
                )
            self.log(
                f"Saving frames: {start} to {end - 1} "
                f"({end - start} frame(s), inclusive)"
            )
            self.log(f"Flat-field correction: {'ON' if use_flat else 'OFF'}")

            def _load_idx(i):
                if is_mraw:
                    return read_mraw_frame(
                        inp, i, self.mraw_frame_bytes, self.mraw_rows, self.mraw_cols
                    )
                if is_cine:
                    return np.asarray(video[i], dtype=np.float32)
                return self._safe_imread(files[i])

            self.log(f"Loading {end - start} frame(s) to save...")
            raw_data = []
            for i in range(start, end):
                im = _load_idx(i)
                if im is not None:
                    if use_flat:
                        im = apply_flat_division(im, self.flat_master, self.flat_norm_strength.get())
                    im = self._apply_flips(im)
                    raw_data.append(im)
            if not raw_data:
                self.log("Error: Could not read any frames from disk.")
                return
            frames_cpu = np.array(raw_data)

            bg_cpu = None
            if not skip_grey_norm:
                self.log("Loading background frames...")
                bg_list = []
                for i in range(bg_start, bg_end):
                    im = _load_idx(i)
                    if im is not None:
                        im = self._apply_flips(im)
                        bg_list.append(im)
                if not bg_list:
                    self.log("Error: Could not read background frames.")
                    return
                bg_cpu = np.array(bg_list).astype(np.float32)

            if using_gpu:
                utils.init_gpu_normalizer()
                try:
                    import cupy as cp
                    import cupyx.scipy.ndimage  # noqa: F401
                except ImportError:
                    using_gpu = False
                    self.log("CuPy not available; falling back to CPU.")
                else:
                    self.log("Moving to GPU...")
                    frames_data = cp.array(frames_cpu)
                    bg_data = cp.mean(cp.array(bg_cpu), axis=0).astype(cp.float32) if bg_cpu is not None else None
                    xp = cp
            if not using_gpu:
                frames_data = frames_cpu
                xp = np
                bg_data = np.mean(bg_cpu, axis=0).astype(np.float32) if bg_cpu is not None else None

            if not skip_grey_norm:
                self.log("Calculating Background...")
                if mode == "binary":
                    if using_gpu:
                        bg_data = cupyx.scipy.ndimage.gaussian_filter(bg_data, 1.0)
                    else:
                        bg_data = scipy.ndimage.gaussian_filter(bg_data, 1.0)

            h, w = frames_data.shape[1:]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            
            # Use resolved paths for VideoWriter
            out_norm = cv2.VideoWriter(norm_path, fourcc, self.fps.get(), (w, h), True)
            out_orig = cv2.VideoWriter(orig_path, fourcc, self.fps.get(), (w, h), True)
            
            min_r = self.min_ratio.get(); max_r = self.max_ratio.get()
            scale_fac = 255.0 / (max_r - min_r)
            bg_mean = xp.mean(bg_data) if bg_data is not None else None
            
            self.log("Processing...")

            for i in range(len(frames_data)):
                curr_raw = frames_data[i].astype(xp.float32)
                raw_cpu = frames_data[i].get() if using_gpu else frames_data[i]

                # EXPORT ORIGINAL (Using Visual Enhancements)
                vis_frame = self.apply_visuals(raw_cpu.astype(np.float32))
                out_orig.write(cv2.cvtColor(vis_frame, cv2.COLOR_GRAY2BGR))

                if save_vis_imgs:
                    save_path = os.path.join(vis_img_dir, f"Img{start+i:06d}.png")
                    self._safe_imwrite(save_path, vis_frame)

                if skip_grey_norm:
                    # Flat-field only: no grey-scale temporal BG normalization
                    res_cpu = self._flat_corrected_to_u8(raw_cpu.astype(np.float32))
                else:
                    # NORMALIZATION (temporal background)
                    if mode == "binary":
                        if using_gpu:
                            curr = cupyx.scipy.ndimage.gaussian_filter(curr_raw, 1.0)
                        else:
                            curr = scipy.ndimage.gaussian_filter(curr_raw, 1.0)
                    else:
                        curr = curr_raw

                    norm = xp.divide(curr * (bg_mean / (xp.mean(curr)+1e-6)), bg_data + 1e-6)
                    
                    if mode == "grey":
                        res = ((xp.clip(norm, min_r, max_r) - min_r) * scale_fac).astype(xp.uint8)
                    else:
                        res = ((xp.abs(norm - 1.0) > self.binary_thresh.get()) * 255).astype(xp.uint8)
                    
                    res_cpu = res.get() if using_gpu else res
                    res_cpu = self._apply_norm_clahe(res_cpu)

                out_norm.write(cv2.cvtColor(res_cpu, cv2.COLOR_GRAY2BGR))
                
                # Safe Image Save
                save_p_norm = os.path.join(img_dir, f"Img{start+i:06d}.png")
                self._safe_imwrite(save_p_norm, res_cpu)

                if i % 50 == 0:
                    self.lbl_status_monitor.config(text=f"Frame {i}/{len(frames_data)}")
                    self.frame.update_idletasks()

            out_norm.release(); out_orig.release()
            if is_cine: 
                try: video.close() 
                except: pass
            
            self.log("Done!")
            self.lbl_status_monitor.config(text="Complete")
            

            def open_player():
                try:
                    DualVideoPlayer(self.frame, str(orig_path), str(norm_path), self.fps.get())
                except Exception as e:
                    messagebox.showerror("Player Error", f"Could not open player: {e}")

            self.frame.after(0, open_player)

        except Exception as e:
            self.log(f"Error: {e}"); import traceback; traceback.print_exc()
        finally:
            self.is_running = False
            self.frame.after(0, lambda: self.btn_run.config(state='normal', text="▶ START PROCESSING", bg="#16A085"))