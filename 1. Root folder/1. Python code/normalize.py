import utils 
from utils import *

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

        # --- Initialize Components ---
        if self.trajectory is not None:
            self.embed_graph()

        self.update_frame()

    def embed_graph(self):
        """Embeds the matplotlib graph into the bottom frame"""
        # Create Plot (Wide and short to fit)
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
        self.ax.invert_yaxis() # Image Y grows downwards
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
                    self.canvas_graph.draw_idle() # Efficient redraw

                # 5. Update Popup Marker (If open)
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
        self.vis_max = tk.DoubleVar(value=4095) # 12-bit default
        self.vis_gamma = tk.DoubleVar(value=1.0)
        self.vis_gain = tk.DoubleVar(value=1.0)
        self.vis_bright = tk.DoubleVar(value=0.0)
        self.vis_auto = tk.BooleanVar(value=False)
        self.current_preview_raw = None
        
        # State
        self.is_cine = False
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
        # NEW: 3. Visual Adjustments (Moved Actions to 4)
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
        
        tk.Checkbutton(f_vis, text="Auto-Levels", variable=self.vis_auto, 
                       bg="#EAECEE", command=self.refresh_preview_visuals).grid(row=5, column=0, columnspan=3, sticky="w", padx=5)

        # ============================================================
        # NEW: Pixel Histogram
        # ============================================================
        f_hist = tk.LabelFrame(left_col, text="Pixel Histogram", font=("Arial", 10, "bold"))
        f_hist.pack(fill="x", pady=2)
        
        self.fig, self.ax = plt.subplots(figsize=(4, 2), dpi=80)
        self.fig.patch.set_facecolor('#F0F0F0')
        self.canvas_hist = FigureCanvasTkAgg(self.fig, master=f_hist)
        self.canvas_hist.get_tk_widget().pack(fill="both", expand=True)
        
        # (Original "3. Actions" continues below here...)


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
            # FIX: Resolve ".." before opening
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
            print(f"WRITE ERROR: {e}") # Print the actual error reason
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
        return (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)

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
        
        # 3. Update Canvas Background Only (Keep ROI box on top)
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

            valid_exts = {'.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp'}
            self.image_files = []
            for root, dirs, files in os.walk(d):
                for file in files:
                    ext = os.path.splitext(file)[1].lower()
                    if ext in valid_exts:
                        self.image_files.append(os.path.join(root, file))

            self.image_files = natsorted(self.image_files)
            
            if not self.image_files:
                messagebox.showerror("Error", "No images found!\nCheck if the folder contains images.")
                return

            self.log(f"Success: Loaded {len(self.image_files)} images.")
            self.show_first_frame()

    def show_first_frame(self):
        img = None
        
        # ADDED CINE SUPPORT HERE
        if self.is_cine:
            try:
                v = pims.open(self.input_folder.get())
                img = v[0] 
                v.close()
                img = np.array(img)
                if img.dtype == np.uint16:
                    # Normalize the 16-bit range to 8-bit (0-255) for display
                    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
                    img = img.astype(np.uint8)
                # Convert RGB to BGR for consistency if PIMS returns RGB
                if len(img.shape) == 3: img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                elif len(img.shape) == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            except Exception as e:
                self.log(f"ERROR: Could not read CINE file: {e}")
                return
        else:
            if not self.image_files: return
            img = self._safe_imread(self.image_files[0], color=True)
        
        # (Rest of the function remains the same)
        if img is None: 
            self.log("ERROR: Could not read first image.")
            return

        self.original_h, self.original_w = img.shape[:2]
        # ... (keep existing canvas drawing logic) ...
        # (Copy the rest from the full code provided in the previous turn)
        
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
        # IMPORTANT: Tag the image as "bg_img" so we can update it later
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
                        # Convert to Gray if needed
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
                # Ensure ROI is within bounds (e.g. if image size changes)
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

            # --- PHASE 2: WRITING (STABILIZATION) ---
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
            
            # --- START PLAYER ---
            # IMPORTANT: Pass input_path if CINE, or image_files list if Folder
            player_source = self.input_folder.get() if self.is_cine else self.image_files
            
            # --- MODIFIED LINE BELOW: Added 'trajectory=smooth_y' ---
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
        
        # --- Processing Settings ---
        self.export_all = tk.BooleanVar(value=True)
        self.start_frame = tk.IntVar(value=0)
        self.end_frame = tk.IntVar(value=1000)
        self.bg_frames = tk.IntVar(value=10)
        self.window_size = tk.IntVar(value=1)
        self.fps = tk.DoubleVar(value=30.0)
        self.use_gpu = tk.BooleanVar(value=utils.HAS_GPU_NORMALIZER)
        
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

        # --- Preview State ---
        self.preview_frame_idx = tk.IntVar(value=0) 
        self.is_running = False
        self.current_preview_raw = None 
        self.fig = None 
        self.ax = None
        self.canvas_hist = None
        
        self.is_cine_mode = False
        self.cine_video = None 
        
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
        
        # Left Column
        left_col = tk.Frame(main_content, width=400)
        left_col.pack(side="left", fill="both", padx=(0, 5))
        
        # Right Column
        right_col = tk.Frame(main_content)
        right_col.pack(side="right", fill="both", expand=True, padx=(5, 0))

        # ================= LEFT COLUMN =================
        # 1. IO
        f_io = tk.LabelFrame(left_col, text="1. Input / Output", font=("Arial", 10, "bold"), fg="#2C3E50")
        f_io.pack(fill="x", pady=2)
        grid_opts = {'padx': 5, 'pady': 2, 'sticky': 'w'}
        
        tk.Label(f_io, text="Input (.cine/.tif):").grid(row=0, column=0, **grid_opts)
        tk.Entry(f_io, textvariable=self.input_path, width=35).grid(row=0, column=1, **grid_opts)
        tk.Button(f_io, text="Browse", command=self.browse_input).grid(row=0, column=2, **grid_opts)
        
        tk.Label(f_io, text="Output Folder:").grid(row=1, column=0, **grid_opts)
        tk.Entry(f_io, textvariable=self.output_path, width=35).grid(row=1, column=1, **grid_opts)
        tk.Button(f_io, text="Browse", command=lambda: self.browse_folder(self.output_path)).grid(row=1, column=2, **grid_opts)
        
        tk.Label(f_io, text="Video Name:").grid(row=2, column=0, **grid_opts)
        tk.Entry(f_io, textvariable=self.video_name, width=35).grid(row=2, column=1, **grid_opts)

        # 2. Normalization Params
        f_norm = tk.LabelFrame(left_col, text="2. Normalization Physics", font=("Arial", 10, "bold"), fg="#2C3E50")
        f_norm.pack(fill="x", pady=2)
        
        # Method Selection
        tk.Label(f_norm, text="Method:").grid(row=0, column=0, padx=5, sticky="w")
        tk.Radiobutton(f_norm, text="Grey-scale", variable=self.norm_mode, value="grey", command=self.update_mode_ui).grid(row=0, column=1)
        tk.Radiobutton(f_norm, text="Binary", variable=self.norm_mode, value="binary", command=self.update_mode_ui).grid(row=0, column=2)

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

        # Temporal Settings
        tk.Label(f_norm, text="Bg Frames:").grid(row=3, column=0, padx=5, sticky="w")
        tk.Entry(f_norm, textvariable=self.bg_frames, width=6).grid(row=3, column=1, sticky="w")
        
        tk.Label(f_norm, text="Temp Window:").grid(row=3, column=2, padx=5, sticky="w")
        tk.Entry(f_norm, textvariable=self.window_size, width=6).grid(row=3, column=3, sticky="w")

        # 3. Visual Tools
        f_tools = tk.LabelFrame(left_col, text="3. Visual Adjustments (Original Only)", font=("Arial", 10, "bold"), bg="#EAECEE", fg="#C0392B")
        f_tools.pack(fill="x", pady=5)
        
        tk.Label(f_tools, text="⚠ NOTE: These sliders DO NOT affect Normalization.", 
                 bg="#EAECEE", fg="red", font=("Arial", 9, "bold")).pack(anchor="w", padx=5, pady=(2,5))
        
        tk.Checkbutton(f_tools, text="Auto-Levels", variable=self.vis_auto, command=self.update_vis_mode, bg="#EAECEE").pack(anchor="w", padx=5)
        
        self.f_sliders = tk.Frame(f_tools, bg="#EAECEE")
        self.f_sliders.pack(fill="x", padx=5, pady=2)
        
        def make_slider(parent, label, var, vmin, vmax, res, row):
            tk.Label(parent, text=label, bg="#EAECEE", width=10, anchor='w').grid(row=row, column=0, sticky="w")
            s = tk.Scale(parent, variable=var, from_=vmin, to=vmax, resolution=res, orient="horizontal", bg="#EAECEE", length=180, showvalue=False, command=lambda x: self.refresh_preview_visuals())
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
        tk.Label(r1, text="Frames:").pack(side="left")
        tk.Entry(r1, textvariable=self.start_frame, width=6).pack(side="left")
        tk.Label(r1, text="to").pack(side="left", padx=2)
        tk.Entry(r1, textvariable=self.end_frame, width=6).pack(side="left")
        tk.Checkbutton(r1, text="Export All", variable=self.export_all).pack(side="left", padx=5)
        
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

    # --- HELPERS: SAFE IO ---
    def _safe_imread(self, path):
        """Helper to force read Korean paths on Windows"""
        try:
            path = os.path.abspath(path)
            stream = open(path, "rb")
            bytes = bytearray(stream.read())
            numpyarray = np.asarray(bytes, dtype=np.uint8)
            return cv2.imdecode(numpyarray, cv2.IMREAD_GRAYSCALE) # Normalizer usually works in Gray
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
        self.generate_preview()

    def go_back(self):
        self.frame.destroy()
        self.on_back()
        

    def load_input_source(self):
        """Loads the CINE file or scans the folder and populates the listbox."""
        inp = self.input_path.get()
        if not inp or not os.path.exists(inp): return
        
        self.file_listbox.delete(0, tk.END)
        self.preview_frame_idx.set(0)
        
        # Close existing CINE handle if it exists
        if self.cine_video:
            try: self.cine_video.close()
            except: pass
            self.cine_video = None

        if self.is_cine_mode:
            try:
                self.cine_video = pims.open(inp)
                count = len(self.cine_video)
                # For CINE, show Frame numbers
                for i in range(count):
                    self.file_listbox.insert(tk.END, f"Frame {i}")
                self.log(f"Loaded CINE: {count} frames.")
            except Exception as e:
                self.log(f"Error opening CINE: {e}")
                self.is_cine_mode = False # Fallback option
        
        if not self.is_cine_mode:
            # Folder mode: Scan for images
            files = []
            for ext in ['*.tif', '*.tiff', '*.png', '*.jpg', '*.bmp']:
                files.extend(glob.glob(os.path.join(inp, ext)))
            files = natsorted(files)
            
            for f in files:
                self.file_listbox.insert(tk.END, os.path.basename(f))
            self.log(f"Loaded Folder: {len(files)} images.")
            
        # Trigger initial preview if data was found
        if self.file_listbox.size() > 0:
            self.generate_preview()

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
    # ==========================================================================     
    def browse_folder(self, var):
        path = filedialog.askdirectory()
        if path: var.set(os.path.abspath(path))

    def browse_input(self):
        ans = messagebox.askyesno("Input Type", "Are you loading a .CINE file?\n\nYes = .cine file\nNo = Folder of images")
        path = ""
        if ans:
            path = filedialog.askopenfilename(filetypes=[("CINE", "*.cine"), ("All", "*.*")])
            self.is_cine_mode = True
        else:
            path = filedialog.askdirectory()
            self.is_cine_mode = False
            
        if path: 
            self.input_path.set(os.path.abspath(path))
            # --- MODIFIED: Call the new loader ---
            self.load_input_source()

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
        img = img_raw.astype(np.float32)
        
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
        
        # Get total counts from the already populated listbox
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

            # --- RETRIEVE DATA BASED ON MODE ---
            if self.is_cine_mode:
                if self.cine_video is None: return
                video = self.cine_video
                
                # Calculate BG from the persistent video object
                bg_count = min(self.bg_frames.get(), len(video))
                bg_frames = np.array([video[i] for i in range(bg_count)]).astype(np.float32)
                bg = np.mean(bg_frames, axis=0)

                # Get target frame
                target_raw = video[idx].astype(np.float32)
                
            else: # Folder Mode
                # Calculate BG from files
                bg_count = min(self.bg_frames.get(), total_frames)
                bg_data_list = []
                for i in range(bg_count):
                    fname = self.file_listbox.get(i)
                    fpath = os.path.join(inp, fname)
                    im = self._safe_imread(fpath)
                    if im is not None: bg_data_list.append(im)
                
                if not bg_data_list: return
                bg_frames = np.array(bg_data_list).astype(np.float32)
                bg = np.mean(bg_frames, axis=0)
                
                # Get target frame file
                filename = self.file_listbox.get(idx)
                target_file = os.path.join(inp, filename)
                target_raw = self._safe_imread(target_file).astype(np.float32)

            # Sync listbox if the change came from buttons or random pick
            if not from_listbox:
                self.sync_listbox()
            
            # --- COMMON PREVIEW LOGIC (Same as before) ---
            self.current_preview_raw = target_raw
            self.update_histogram(target_raw)
            self.refresh_preview_visuals()

            # Normalization Preview
            target_blur = cv2.GaussianBlur(target_raw, (3, 3), 0)
            bg_blur = cv2.GaussianBlur(bg, (3, 3), 0)
            correction = np.mean(bg_blur) / (np.mean(target_blur) + 1e-6)
            norm = (target_blur * correction) / (bg_blur + 1e-6)
            
            if self.norm_mode.get() == "grey":
                min_r = self.min_ratio.get(); max_r = self.max_ratio.get()
                res = np.clip(norm, min_r, max_r)
                res = ((res - min_r) * (255.0 / (max_r - min_r))).astype(np.uint8)
            else:
                mask = np.abs(norm - 1.0) > self.binary_thresh.get()
                res = (mask * 255).astype(np.uint8)
            
            self.show_on_label(self.lbl_preview_norm, res)

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
            # FIX: Clean Paths immediately
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
            if os.path.isfile(inp) and inp.lower().endswith('.cine'):
                is_cine = True
                self.log("Opening CINE...")
                video = pims.open(inp)
                total_frames = len(video)
                start = 0 if self.export_all.get() else self.start_frame.get()
                end = min(total_frames, total_frames if self.export_all.get() else self.end_frame.get())
                self.log(f"Loading frames {start}-{end} into RAM...")
                frames_cpu = np.array([video[i] for i in range(start, end)])
            else:
                files = []
                for ext in ['*.tif', '*.tiff', '*.png', '*.jpg']:
                    files.extend(glob.glob(os.path.join(inp, ext)))
                files = natsorted(files)
                start = 0 if self.export_all.get() else self.start_frame.get()
                end = min(len(files), len(files) if self.export_all.get() else self.end_frame.get())
                self.log("Loading Images (Safe Mode)...")
                
                # SAFE LOAD LOOP
                raw_data = []
                for i in range(start, end):
                    im = self._safe_imread(files[i])
                    if im is not None: raw_data.append(im)
                    
                frames_cpu = np.array(raw_data)

            if using_gpu:
                self.log("Moving to GPU...")
                frames_data = cp.array(frames_cpu)
                xp = cp
            else:
                frames_data = frames_cpu
                xp = np

            self.log("Calculating Background...")
            bg_data = xp.mean(frames_data[:self.bg_frames.get()], axis=0).astype(xp.float32)
            
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
            bg_mean = xp.mean(bg_data)
            
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

                # NORMALIZATION
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