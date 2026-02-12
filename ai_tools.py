from utils import *
import utils

try:
    import sam2.utils.misc
    import glob
    from PIL import Image
    import torch
    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(0))
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
    print("SUCCESS: SAM 2 patched to support PNG/TIF images.")

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


class SamLabelerApp:
    def __init__(self, parent_frame, on_back):
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
        
        # --- Y-limit Variables ---
        self.y_limit = None
        self.setting_limit_mode = False
        self.crosshair_lines = []
        
        # --- CT Specific Variables ---
        self.is_ct_mode = False
        self.ct_volume = None       
        self.ct_dims = (0,0,0)      
        self.slice_axis = tk.StringVar(value="Z") 
        self.current_slice_idx = 0  
        self.plotter = None 
        
        # --- State Variables ---
        self.mode = tk.StringVar(value="ai") 
        self.label_preset_mode = tk.StringVar(value="lpbf") # Default to LPBF
        self.new_class_entry_var = tk.StringVar()
        self.auto_ai_var = tk.BooleanVar(value=False)
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

        self.PALETTE = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), 
            (255, 0, 255), (0, 255, 255), (255, 128, 0), (128, 0, 255),
            (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 0)
        ]
        self.LPBF_PRESETS = ["Keyhole", "KH-Pore", "Gas-Pore", "Bubble", "Spatter", "Plume"]

        self.show_landing_page()
        if not HAS_SAM: messagebox.showerror("Error", "SAM library not found.")
        

            
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
        
        tk.Label(center_frame, text="Select type of x-ray analysis", font=("Helvetica", 24, "bold"), bg="#ECF0F1", fg="#2C3E50").pack(pady=(0, 40))
        
        # Button 1: X-ray Imaging (2D)
        btn_2d = tk.Button(center_frame, text="1. X-ray Imaging\n(2D Image Stack)", font=("Arial", 16, "bold"), 
                           bg="#3498DB", fg="white", width=25, height=5, cursor="hand2",
                           command=self.setup_ui_2d)
        btn_2d.pack(pady=10)
        
        # Button 2: X-ray Tomography (3D)
        btn_3d = tk.Button(center_frame, text="2. X-ray Tomography\n(3D CT / NIfTI)", font=("Arial", 16, "bold"), 
                           bg="#E67E22", fg="white", width=25, height=5, cursor="hand2",
                           command=self.setup_ui_ct)
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

# 1. RIGHT SIDE: Auto AI Button
        f_right = tk.Frame(self.toolbar, bg="#D6EAF8")
        f_right.pack(side="right", padx=10)
        tk.Checkbutton(f_right, text="⚡ Auto-Ready", variable=self.auto_ai_var, 
                       indicatoron=0,
                       bg="#8E44AD", fg="white", selectcolor="#2ECC71", # Purple=OFF, Green=ON
                       font=("Arial", 10, "bold"), width=12, height=1,
                       # If clicked, run embedding on the current image immediately
                       command=lambda: self.run_embedding_thread() if self.auto_ai_var.get() else None
                       ).pack(side="left", padx=5)

        # 2. LEFT SIDE: Class Management
        self.f_classes_container = tk.Frame(self.toolbar, bg="#D6EAF8")
        self.f_classes_container.pack(side="left", padx=5, fill="x")
        
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

    def update_presets_2d(self):
        # Clear existing widgets in container
        for w in self.f_classes_container.winfo_children(): w.destroy()
        
        # 1. "New Class" Entry
        tk.Label(self.f_classes_container, text="New Class:", bg="#D6EAF8", font=("Arial", 9)).pack(side="left")
        tk.Entry(self.f_classes_container, textvariable=self.new_class_entry_var, width=12).pack(side="left", padx=2)
        tk.Button(self.f_classes_container, text="+", command=self.add_from_entry, bg="#2ECC71", width=3, cursor="hand2").pack(side="left", padx=(0, 15))

        # 2. Active Class Indicator
        tk.Label(self.f_classes_container, text="|  Active:", bg="#D6EAF8", font=("Arial", 9)).pack(side="left")
        self.lbl_active_class = tk.Label(self.f_classes_container, text="None", font=("Arial", 10, "bold"), bg="#D6EAF8", fg="blue")
        self.lbl_active_class.pack(side="left", padx=5)

        # 3. Manage Button
        tk.Button(self.f_classes_container, text="⚙ Manage", command=self.manage_active_class, 
                  bg="#95A5A6", fg="white", font=("Arial", 8), cursor="hand2").pack(side="left", padx=5)
        
        self.refresh_class_ui()

    def auto_segment_2d(self):
        """Automatically generates masks for the entire image using SAM."""
        if not self.is_ai_ready or self.raw_image is None:
            messagebox.showwarning("AI Not Ready", "Please load an image and click '⚡ Run AI' (yellow button) first to load the model embeddings.")
            return

        # 1. Check if class is selected
        if self.current_class_id.get() == -1:
            messagebox.showwarning("No Class", "Please select a Class (e.g., Keyhole, Pore) from the toolbar before running Auto AI.")
            return

        self.log("Running Auto-Segment... (This may take a few seconds)", "orange")
        self.frame.update_idletasks()

        try:
            # 2. Initialize Generator

            from segment_anything import SamAutomaticMaskGenerator
            mask_generator = SamAutomaticMaskGenerator(
                self.predictor.model,
                points_per_side=32, # Adjust grid density (higher = more detection, slower)
                pred_iou_thresh=0.86,
                stability_score_thresh=0.92,
                crop_n_layers=0,
                crop_n_points_downscale_factor=1,
                min_mask_region_area=100,
            )
            
            # 3. Generate Masks
            masks = mask_generator.generate(self.raw_image)
            
            if not masks:
                self.log("No objects detected.", "red")
                return

            self.save_state_for_undo()
            
            # 4. Convert Masks to Polygons
            h, w = self.raw_image.shape[:2]
            added_count = 0
            
            for res in masks:

                m = res['segmentation'].astype(np.uint8) * 255
                
                # Find contours
                cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for c in cnts:
                    if cv2.contourArea(c) < 50: continue # Filter noise
                    
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
        

        # --- 3D SPECIFIC TOOLBAR (MANUAL CLASS ONLY) ---
        tk.Label(self.toolbar, text="| New Class:", bg="#D6EAF8").pack(side="left", padx=5)
        tk.Entry(self.toolbar, textvariable=self.new_class_entry_var, width=10).pack(side="left", padx=2)
        tk.Button(self.toolbar, text="+ Add", command=self.add_from_entry, bg="#3498DB", fg="white", font=("Arial", 8, "bold")).pack(side="left", padx=2)
        
        tk.Label(self.toolbar, text="Active:", bg="#D6EAF8").pack(side="left", padx=10)
        tk.Button(self.toolbar, text="⚙", command=self.manage_active_class, bg="#95A5A6", fg="white", width=2).pack(side="left", padx=2)

        self.f_active_classes = tk.Frame(self.toolbar, bg="#D6EAF8")
        self.f_active_classes.pack(side="left", padx=5)
        
        tk.Checkbutton(self.toolbar, text="Auto-Run AI", variable=self.auto_ai_var, 
                       bg="#D6EAF8", font=("Arial", 9, "bold")).pack(side="right", padx=20)
        
        # Split Layout
        self.main_split = tk.PanedWindow(self.frame, orient=tk.HORIZONTAL, sashwidth=5, bg="#BDC3C7")
        self.main_split.pack(fill="both", expand=True)

        # LEFT
        self.left_panel = tk.Frame(self.main_split, bg="white", width=400)
        self.main_split.add(self.left_panel, minsize=100)
        
        tk.Label(self.left_panel, text="3D Controls", font=("Arial", 12, "bold"), bg="white", fg="#E67E22").pack(pady=10)
        
        f_3d = tk.LabelFrame(self.left_panel, text="PyVista Settings", bg="white", padx=10, pady=10)
        f_3d.pack(fill="x", padx=10)
        
        self.threshold_val = tk.IntVar(value=100)
        self.opacity_val = tk.DoubleVar(value=0.5)
        
        tk.Label(f_3d, text="Iso-Threshold:", bg="white").pack(anchor="w")
        tk.Scale(f_3d, from_=0, to=255, orient=tk.HORIZONTAL, variable=self.threshold_val, bg="white", 
                 command=lambda x: self.update_isosurface_3d()).pack(fill="x")
        
        tk.Label(f_3d, text="Opacity:", bg="white").pack(anchor="w", pady=(5,0))
        tk.Scale(f_3d, from_=0.0, to=1.0, resolution=0.1, orient=tk.HORIZONTAL, variable=self.opacity_val, bg="white", 
                 command=lambda x: self.update_isosurface_3d()).pack(fill="x")

        tk.Button(f_3d, text="Re-Launch 3D Window", command=self.launch_pyvista_window, bg="#2ECC71", fg="white").pack(fill="x", pady=15)
        
        # RIGHT
        self.right_panel = tk.Frame(self.main_split, bg="#2C3E50")
        self.main_split.add(self.right_panel, minsize=400)

        self.canvas_container = tk.Frame(self.right_panel, bg="#2C3E50")
        self.canvas_container.pack(side="top", fill="both", expand=True)
        self.canvas = tk.Canvas(self.canvas_container, bg="black", cursor="cross", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        self.ct_ctrl_frame = tk.Frame(self.right_panel, bg="#34495E", pady=5)
        self.ct_ctrl_frame.pack(side="bottom", fill="x")
        
        f_axes = tk.Frame(self.ct_ctrl_frame, bg="#34495E")
        f_axes.pack(side="top", fill="x", pady=2)
        tk.Label(f_axes, text="Plane:", fg="#F39C12", bg="#34495E", font=("Arial", 10, "bold")).pack(side="left", padx=10)
        tk.Radiobutton(f_axes, text="X", variable=self.slice_axis, value="X", command=self.on_axis_change, bg="#34495E", fg="white", selectcolor="#2C3E50").pack(side="left")
        tk.Radiobutton(f_axes, text="Y", variable=self.slice_axis, value="Y", command=self.on_axis_change, bg="#34495E", fg="white", selectcolor="#2C3E50").pack(side="left")
        tk.Radiobutton(f_axes, text="Z", variable=self.slice_axis, value="Z", command=self.on_axis_change, bg="#34495E", fg="white", selectcolor="#2C3E50").pack(side="left")
        
        f_slide = tk.Frame(self.ct_ctrl_frame, bg="#34495E")
        f_slide.pack(side="top", fill="x")
        tk.Label(f_slide, text="Slice:", fg="white", bg="#34495E").pack(side="left", padx=5)
        self.slice_slider = tk.Scale(f_slide, from_=0, to=100, orient=tk.HORIZONTAL, command=self.on_slider_change, bg="#34495E", fg="white", length=400)
        self.slice_slider.pack(side="left", fill="x", expand=True, padx=10)

        self._bind_events()
        self.file_listbox = tk.Listbox(self.frame)
        

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
        
        # Toolbar
        self.toolbar = tk.Frame(self.frame, bg="#D6EAF8", pady=5)
        self.toolbar.pack(fill="x")
        
        tk.Button(self.toolbar, text="Load Data", command=self.load_data_router, bg="white").pack(side="left", padx=10)
        tk.Button(self.toolbar, text="Load Model", command=self.load_model_thread, bg="white").pack(side="left", padx=5)
        tk.Button(self.toolbar, text="⚡ Run AI", command=self.run_embedding_thread, bg="#F1C40F", font=("Arial", 9, "bold")).pack(side="left", padx=15)


        # Y-LIMIT BUTTON
        self.btn_limit = tk.Button(self.toolbar, text="⛔ Set Top Limit", 
                                   command=self.toggle_limit_mode, 
                                   bg="#EC7063", fg="white", font=("Arial", 9, "bold"))
        self.btn_limit.pack(side="left", padx=5)

        tk.Label(self.toolbar, text="| Tool:", bg="#D6EAF8").pack(side="left", padx=5)

        tk.Radiobutton(self.toolbar, text="Point", variable=self.mode, value="ai", command=self.clear_current_annotation, bg="#D6EAF8").pack(side="left")
        tk.Radiobutton(self.toolbar, text="Poly", variable=self.mode, value="polygon", command=self.clear_current_annotation, bg="#D6EAF8").pack(side="left")
        tk.Radiobutton(self.toolbar, text="Edit/Select", variable=self.mode, value="edit", command=self.clear_current_annotation, bg="#D6EAF8").pack(side="left")
        
        # Delete Button
        tk.Button(self.toolbar, text="Delete Selected", command=self.delete_selected, bg="#E74C3C", fg="white", font=("Arial", 9, "bold")).pack(side="left", padx=15)
        
        tk.Button(self.toolbar, text="Reset View", command=self.reset_zoom, bg="#D7BDE2").pack(side="left", padx=5)
    def _bind_events(self):
        self.canvas.bind("<Button-1>", self.on_left_click)
        self.canvas.bind("<Button-3>", self.on_right_click)
        self.canvas.bind("<ButtonPress-2>", self.start_pan)
        self.canvas.bind("<B2-Motion>", self.do_pan)
        self.canvas.bind("<Control-MouseWheel>", self.on_mousewheel) 
        self.canvas.bind("<Motion>", self.on_mouse_move)
        self.frame.bind_all("<Delete>", self.clear_current_annotation) 
        self.frame.bind_all("s", self.ask_label_and_save)              
        self.frame.bind_all("<Left>", self.on_left_arrow)
        self.frame.bind_all("<Right>", self.on_right_arrow)
        self.frame.bind_all("<Control-z>", self.undo)
        self.canvas.bind("<Configure>", self.on_resize)
        
    def toggle_limit_mode(self):
        if self.y_limit is not None:
            # If limit exists, clear it
            self.y_limit = None
            self.btn_limit.config(text="⛔ Set Top Limit", bg="#EC7063")
            self.show_image(self.raw_image, self.current_mask)
            self.log("Y-Limit Cleared", "green")
        else:
            # Enter setting mode
            self.setting_limit_mode = True
            self.btn_limit.config(text="Click on Image...", bg="#F7DC6F", fg="black")
            self.canvas.config(cursor="tcross")
            self.log("Move mouse to set limit. Click to confirm.", "blue")

    def on_mouse_move(self, e):
        
        self.canvas.delete("crosshair")
        
        if not self.setting_limit_mode or self.raw_image is None:
            return

        # Get Canvas dimensions
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        
        # Draw Dotted Crosshair
        self.canvas.create_line(0, e.y, cw, e.y, fill="red", dash=(4, 4), width=2, tags="crosshair")
        self.canvas.create_line(e.x, 0, e.x, ch, fill="red", dash=(4, 4), width=2, tags="crosshair")
        
        self.canvas.create_rectangle(0, 0, cw, e.y, fill="red", stipple="gray50", outline="", tags="crosshair")
        self.canvas.create_text(e.x + 15, e.y - 15, text="Limit Top", fill="red", font=("Arial", 10, "bold"), anchor="sw", tags="crosshair")

    def apply_y_limit_constraint(self, mask):
        """Helper to zero out mask above the limit"""
        if self.y_limit is None or mask is None: return mask
        
        # Set all pixels above y_limit to False/0
        # mask shape is (H, W)
        h, w = mask.shape[:2]
        safe_limit = min(max(0, self.y_limit), h)
        
        mask[:safe_limit, :] = 0 
        return mask    
    # ==========================================================================
    # DATA LOADING & ROUTING
    # ==========================================================================
    def load_data_router(self):
        if self.is_ct_mode:
            self.load_folder_ct()
        else:
            self.load_folder_2d()

    def load_folder_ct(self):
        # Check for libraries
        if not HAS_NIBABEL and not HAS_H5PY:
            messagebox.showerror("Error", "Missing libraries.\nNeed 'nibabel' (for .nii) or 'h5py' (for .h5)")
            return

        d = filedialog.askdirectory(title="Select Folder containing .nii or .h5 files")
        if not d: return
        
        # 1. Clear previous data
        self.image_list = []
        self.ct_volume = None
        self.raw_image = None
        
        # 2. Robust File Scanning (Added .h5)
        if os.path.exists(d):
            for f in os.listdir(d):
                lower_f = f.lower()
                # --- MODIFIED LINE ---
                if lower_f.endswith(('.nii', '.nii.gz', '.h5')):
                    self.image_list.append(os.path.join(d, f))
        
        # Sort naturally
        try:
            from natsort import natsorted
            self.image_list = natsorted(self.image_list)
        except:
            self.image_list.sort() 
        
        if not self.image_list:
            messagebox.showerror("Error", f"No .nii or .h5 files found in folder:\n{d}")
            return

        # 3. Setup UI for CT
        self.is_ct_mode = True
        # FIX: Ensure frame exists before packing
        if hasattr(self, 'ct_ctrl_frame') and self.ct_ctrl_frame.winfo_exists():
            self.ct_ctrl_frame.pack(side="bottom", fill="x", pady=5)
        
        self.log(f"Found {len(self.image_list)} volumes. Loading first...", "blue")
        self.load_classes_txt(d)
        
        # 4. Load the first volume
        self.current_idx = 0
        self.load_current_ct_volume()

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
        self.refresh_class_ui()

    # ==========================================================================
    # PYVISTA 3D LOGIC
    # ==========================================================================

    def launch_pyvista_window(self):
        if not HAS_PYVISTA or self.ct_volume is None: return
        
        # Close existing plotter if open
        if self.plotter is not None:
            try: self.plotter.close()
            except: pass
            self.plotter = None

        try:
            # 1. Prepare Data
            self.pv_grid = pv.ImageData()
            self.pv_grid.dimensions = self.ct_volume.shape
            self.pv_grid.spacing = (1, 1, 1) 
            self.pv_grid.point_data["values"] = self.ct_volume.flatten(order="F") 
            
            # 2. Create Plotter Window
            self.plotter = pv.Plotter(title="3D CT Reference (Interactive)")
            self.plotter.add_axes()
            self.plotter.add_bounding_box(color="white")
            
            # 3. Initial Draw
            self.update_isosurface_3d(reset_cam=True)

            # 4. Show Window (Non-Blocking Mode)
            # interactive_update=True allows us to control the loop manually
            self.plotter.show(interactive_update=True, auto_close=False)
            
            # 5. Start the "Heartbeat" Loop

            self._run_pv_loop()
            
            self.log("3D Window Active", "green")
            
        except Exception as e:
            self.log(f"3D Init Error: {e}", "red")
            self.plotter = None

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

    def update_isosurface_3d(self, reset_cam=False):
        """Called only when Threshold/Opacity sliders are moved."""
        if self.plotter is None or self.pv_grid is None: return
        
        try:
            mesh = self.pv_grid.contour([self.threshold_val.get()])
            
            self.plotter.add_mesh(mesh, color="salmon", opacity=self.opacity_val.get(), name="iso")
            
            if reset_cam: self.plotter.reset_camera()
            if self.plotter.iren.initialized: self.plotter.update()
        except: pass

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
            self.plotter.add_mesh(plane, color="white", style="wireframe", opacity=0.8, line_width=2, name="slice")
            
            if self.plotter.iren.initialized: self.plotter.update()
        except: pass

    # ==========================================================================
    # DISPLAY LOGIC (2D & CT)
    # ==========================================================================
    def load_current_ct_volume(self):
        # 1. CLEANUP (Prevents memory leaks from previous loads)
        if hasattr(self, 'h5_handle') and self.h5_handle:
            try: self.h5_handle.close()
            except: pass
            self.h5_handle = None
        
        self.ct_volume = None 
        import gc; gc.collect()

        path = self.image_list[self.current_idx]
        fname = os.path.basename(path)
        
        self.log(f"Smart Loading {fname}...", "orange")
        self.canvas.delete("all")
        self.canvas.create_text(200, 200, text="GENERATING 3D PREVIEW...\n(Downsampling to fit RAM...)", fill="white", font=("Arial", 14))
        self.frame.update_idletasks()
        
        try:
            # 2. FILE OPENING
            if path.lower().endswith('.h5'):
                if not HAS_H5PY: raise ImportError("h5py not installed")
                self.h5_handle = h5py.File(path, 'r')
                
                # Auto-find the dataset key
                dataset = None
                for k in ['data', 'exchange/data', 'volume', 'reconstruction']:
                    if k in self.h5_handle:
                        dataset = self.h5_handle[k]
                        break
                
                # Fallback search if standard keys fail
                if dataset is None:
                    def find_3d(name, node):
                        nonlocal dataset
                        if dataset: return
                        if isinstance(node, h5py.Dataset) and node.ndim == 3:
                            dataset = node
                    self.h5_handle.visititems(find_3d)

                if dataset is None: raise Exception("No 3D dataset found")
                self.ct_proxy = dataset 
            else:
                nii = nib.load(path)
                self.ct_proxy = nii.dataobj 
            
            # 3. SMART 3D GENERATION (The Crash Fix)
            self.ct_dims = self.ct_proxy.shape
            
            target_dim = 256
            max_dim = max(self.ct_dims)
            step = max(1, max_dim // target_dim)
            
            self.log(f"Downsampling 3D preview by {step}x to save RAM...", "blue")

            # Get contrast from the middle slice (Fast)
            mid_z = self.ct_dims[2] // 2
            mid_slice = np.asanyarray(self.ct_proxy[:, :, mid_z])
            self.v_min = np.min(mid_slice)
            self.v_max = np.max(mid_slice)

            # Load the Sparse 3D Preview 
            raw_sample = np.asanyarray(self.ct_proxy[::step, ::step, ::step])
            
            # Normalize to 8-bit for display
            if self.v_max - self.v_min > 0:
                self.ct_volume = ((raw_sample - self.v_min) / (self.v_max - self.v_min) * 255).astype(np.uint8)
            else:
                self.ct_volume = raw_sample.astype(np.uint8)
            
            # 4. LAUNCH WINDOW
            self.slice_axis.set("Z")
            self.current_slice_idx = mid_z
            self.update_slider_range()
            self.load_current_slice_ct() # Load the high-res 2D slice
            self.launch_pyvista_window() # Open the 3D window
            
            self.log(f"Ready: {fname} (3D scaled by 1/{step})", "green")
            
        except Exception as e:
            self.log(f"Load Error: {e}", "red")
            print(f"Details: {e}")

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
        if self.is_ct_mode and self.ct_volume is not None:
            self.current_slice_idx = int(val)
            self.load_current_slice_ct()

    def load_current_slice_ct(self):
        # Safety check
        if not hasattr(self, 'ct_proxy'): return

        axis = self.slice_axis.get()
        idx = self.current_slice_idx
        
        # 1. Read ONLY the requested slice from disk
        try:
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
                
            # 2. Normalize using the global stats we calculated earlier
            if self.v_max - self.v_min > 0:
                slice_u8 = ((slice_raw - self.v_min) / (self.v_max - self.v_min) * 255).astype(np.uint8)
            else:
                slice_u8 = slice_raw.astype(np.uint8)

            self.raw_image = cv2.cvtColor(slice_u8, cv2.COLOR_GRAY2RGB)
            
            # 3. Update Canvas
            self._reset_annotation_state()
            self.load_annotations_ct()
            self.canvas.delete("all")
            self.show_image(self.raw_image)
            
            if self.auto_ai_var.get(): 
                # Run embedding on this specific slice
                self.run_embedding_thread()
                
        except Exception as e:
            print(f"Slice Load Error: {e}")

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
        if name: self.add_class_by_name(name)

    def add_class_by_name(self, name):
        for i, n in self.class_mapping.items():
            if n.lower() == name.lower(): self.current_class_id.set(i); return
        nid = len(self.class_mapping)
        self.class_mapping[nid] = name
        self.class_colors[nid] = self.PALETTE[nid%len(self.PALETTE)]
        self.refresh_class_ui()
        self.current_class_id.set(nid)

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
        with open(p_txt, "w") as f:
            for a in self.current_annotations:
                f.write(f"{a['id']} " + " ".join([f"{c:.6f}" for c in a['coords']]) + "\n")

        # 4. Save Image / Mask
        if self.is_ct_mode:
            # CT: Save the actual image slice (BGR for OpenCV)
            bgr_img = cv2.cvtColor(self.raw_image, cv2.COLOR_RGB2BGR)
            imwrite_safe(p_img, bgr_img)
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

        if not silent: self.log("Saved Successfully", "green")

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

            # --- SAM 2.1 Logic ---
            if "sam2" in ckpt_name:
                from sam2.build_sam import build_sam2
                from sam2.sam2_image_predictor import SAM2ImagePredictor
                
                # Mapping specific files to YAML configs
                config_map = {
                    "tiny": "sam2.1_hiera_t.yaml",
                    "small": "sam2.1_hiera_s.yaml", 
                    "base_plus": "sam2.1_hiera_b+.yaml",
                    "large": "sam2.1_hiera_l.yaml"
                }
                
                model_type = "base_plus" 
                for k in config_map:
                    if k in ckpt_name:
                        model_type = k
                        break
                
                yaml_filename = config_map[model_type]
                model_dir = os.path.dirname(self.checkpoint)
                config_path = os.path.join(model_dir, yaml_filename)

                # Check precision
                if self.device == "cuda" and torch.cuda.is_bf16_supported():
                    print("✅ Using bfloat16 precision")
                else:
                    print("⚠ Using float32 precision")

                # Build Standard Image Model (Not Video)
                if os.path.exists(config_path):
                    sam2_model = build_sam2(config_path, self.checkpoint, device=self.device)
                else:
                    sam2_model = build_sam2(yaml_filename, self.checkpoint, device=self.device)
                
                self.predictor = SAM2ImagePredictor(sam2_model)
                self.is_sam2 = True 
                self.video_predictor = None # Disabled

            # --- SAM 1.0 (ViT) Logic ---
            else:
                from segment_anything import sam_model_registry, SamPredictor
                
                if "vit_h" in ckpt_name: model_type = "vit_h"
                elif "vit_l" in ckpt_name: model_type = "vit_l"
                else: model_type = "vit_b" 
                
                sam = sam_model_registry[model_type](checkpoint=self.checkpoint)
                sam.to(device=self.device)
                self.predictor = SamPredictor(sam)
                self.is_sam2 = False
                self.video_predictor = None

            self.log(f"Model Ready: {ckpt_name}", "green")

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
        
        if self.setting_limit_mode:
            # Convert screen Y to Image Y
            ry = int((e.y - self.off_y) / self.scale)
            h, w = self.raw_image.shape[:2]
            
            # Clamp to image bounds
            self.y_limit = max(0, min(ry, h))
            
            # Reset UI
            self.setting_limit_mode = False
            self.btn_limit.config(text=f"Limit: Y={self.y_limit} (Click to Clear)", bg="#2ECC71")
            self.canvas.config(cursor="cross")
            self.canvas.delete("crosshair")
            
            self.show_image(self.raw_image, self.current_mask)
            self.log(f"Y-Limit set to row {self.y_limit}", "green")
            return
        
        rx = int((e.x - self.off_x) / self.scale)
        ry = int((e.y - self.off_y) / self.scale)
        
        h, w = self.raw_image.shape[:2]
        if rx < 0 or rx >= w or ry < 0 or ry >= h: return
        
        # 2. AI Mode Logic
        if self.mode.get() == "ai":
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
    def run_sam_prediction(self):
        if self.predictor is None or not self.input_points: return
        
        try:
            # 1. Run the prediction
            masks, scores, _ = self.predictor.predict(
                point_coords=np.array(self.input_points),
                point_labels=np.array(self.input_labels),
                multimask_output=False
            )
            
            # 2. EXTRACT THE MASK
            raw_mask = masks[0]
            
            # 3. Apply the Y-Limit Constraint
            self.current_mask = self.apply_y_limit_constraint(raw_mask)
            
            # 4. Update Display
            self.show_image(self.raw_image, self.current_mask)
            
        except Exception as e:
            print(f"SAM Prediction Error: {e}")
            import traceback
            traceback.print_exc() 
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
                self.current_mask = mask.astype(bool)
                self.show_image(self.raw_image, self.current_mask)
                self.poly_points = []

    def ask_label_and_save(self, e=None):
        if self.current_mask is None or self.current_class_id.get()==-1: return
        self.save_state_for_undo()
        cnts,_ = cv2.findContours((self.current_mask*255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            c = max(cnts, key=cv2.contourArea); h,w=self.raw_image.shape[:2]
            norm = [x for pt in c.reshape(-1,2) for x in (pt[0]/w, pt[1]/h)]
            self.current_annotations.append({'id':self.current_class_id.get(), 'coords':norm})
            self.save_all_annotations_to_file(); self.current_mask=None; self.input_points=[]; self.input_labels=[]; self.poly_points=[]
            self.show_image(self.raw_image)

    def clear_current_annotation(self, e=None):
        self.save_state_for_undo(); self.current_mask=None; self.input_points=[]; self.input_labels=[]; self.poly_points=[]; self.show_image(self.raw_image)
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
    def on_left_arrow(self, e):
        if self.is_ct_mode: 
            v=self.slice_slider.get(); 
            if v>0: self.slice_slider.set(v-1)
        else:
            if self.current_idx>0: self.current_idx-=1; self.load_current_image_2d()
    def on_right_arrow(self, e):
        if self.is_ct_mode:
            v=self.slice_slider.get(); 
            if v<self.slice_slider.cget("to"): self.slice_slider.set(v+1)
        else:
            if self.current_idx<len(self.image_list)-1: self.current_idx+=1; self.load_current_image_2d()
    def on_file_select(self, e):
        sel=self.file_listbox.curselection()
        if sel: self.current_idx=sel[0]; self.load_current_image_2d()
    def log(self, m, c="gray"): self.lbl_status.config(text=f"Status: {m}", fg=c)
    def go_back_app(self):
        try: self.plotter.close()
        except: pass
        self.frame.destroy(); self.on_back_callback()


    def show_image(self, rgb, mask=None):
        if rgb is None: return
        disp = rgb.copy(); h, w = rgb.shape[:2]
        
        # Overlay Annotations
        overlay = disp.copy()
        for idx, ann in enumerate(self.current_annotations):
            pts = np.array([[int(c*w) if k%2==0 else int(c*h) for k,c in enumerate(ann['coords'])]], np.int32).reshape((-1,1,2))
            
            # Highlight Selection
            if idx == self.selected_annotation_index:
                cv2.polylines(disp, [pts], True, (255, 255, 255), 3) # Thick white border
                col = (255, 255, 255) # White fill
            else:
                col = self.class_colors.get(ann['id'], (255,255,255))
            
            cv2.fillPoly(overlay, [pts], col)
            
        cv2.addWeighted(overlay, 0.4, disp, 0.6, 0, disp)

        # Overlay Mask
        if mask is not None:
            mask = self.apply_y_limit_constraint(mask)
            m_u8 = (mask * 255).astype(np.uint8)
            c_mask = np.zeros_like(disp); c_mask[:] = (0,255,0)
            c_mask = cv2.bitwise_and(c_mask, c_mask, mask=m_u8)
            disp = cv2.addWeighted(disp, 1.0, c_mask, 0.5, 0)
        
        if self.y_limit is not None:
            # Create a red overlay for the excluded top region
            limit_overlay = disp.copy()
            cv2.rectangle(limit_overlay, (0, 0), (w, self.y_limit), (255, 0, 0), -1) # Red filled box
            cv2.line(limit_overlay, (0, self.y_limit), (w, self.y_limit), (255, 255, 0), 2) # Yellow line
            
            # Apply with transparency (Red tint)
            cv2.addWeighted(limit_overlay, 0.3, disp, 0.7, 0, disp)
        # ---------------------------------------------------------
        
        # Scale & Draw
        cw = self.canvas.winfo_width(); ch = self.canvas.winfo_height()
        if cw < 10: cw, ch = 800, 600
        self.scale = min(cw/w, ch/h) * self.zoom_level
        nw, nh = int(w*self.scale), int(h*self.scale)
        pil = Image.fromarray(disp).resize((nw, nh), Image.Resampling.LANCZOS)
        
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


# ==============================================================================
# YOLO TRAINER APP
# ==============================================================================
class YoloTrainerApp:
    def __init__(self, parent_frame, on_back):
        self.frame = tk.Frame(parent_frame)
        self.frame.pack(fill="both", expand=True)
        self.on_back = on_back
        
        self.source_folders = [] 
        self.out_path = tk.StringVar()
        self.epochs = tk.IntVar(value=50)
        self.batch = tk.IntVar(value=16)
        self.img_size = tk.IntVar(value=640)
        
        self.keyhole_only = tk.BooleanVar(value=False) 
        
        self.is_training = False
        self.setup_ui()
        if not HAS_YOLO:
            messagebox.showerror("Error", "Ultralytics YOLO not found.\nRun: pip install ultralytics")

    def setup_ui(self):
        top = tk.Frame(self.frame, bg="#F5EEF8", height=60)
        top.pack(fill="x")
        tk.Button(top, text="← Back", command=self.go_back, bg="#95A5A6", fg="white", font=("Arial", 11)).pack(side="left", padx=20)
        tk.Label(top, text="Tool: YOLOv8 Trainer & Tracker", bg="#F5EEF8", fg="#8E44AD", font=("Arial", 18, "bold")).pack(side="left", padx=20)

        content = tk.Frame(self.frame, padx=50, pady=20)
        content.pack(fill="both", expand=True)

        # 1. Dataset Selection
        f_data = tk.LabelFrame(content, text="1. Source Datasets (Add 'Input' folders)", font=("Arial", 11, "bold"), fg="#2C3E50")
        f_data.pack(fill="x", pady=5)
        
        self.folder_listbox = tk.Listbox(f_data, height=3)
        self.folder_listbox.pack(side="left", fill="x", expand=True, padx=10, pady=5)
        
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
    

        # 3. Actions
        self.btn_train = tk.Button(content, text="MERGE DATASETS & START TRAINING", bg="#8E44AD", fg="white", font=("Arial", 14, "bold"), height=2, command=self.start_training)
        self.btn_train.pack(fill="x", pady=10)
        
        self.log_area = scrolledtext.ScrolledText(content, height=6, bg="#F4F6F6")
        self.log_area.pack(fill="both", expand=True)

        # --- STATUS BAR ---
        self.lbl_stage = tk.Label(content, text="Status: Idle", bg="#D5D8DC", fg="#2C3E50", font=("Arial", 14, "bold"), height=2)
        self.lbl_stage.pack(fill="x", pady=10)

        # 4. INFERENCE & TRACKING SECTION
        f_infer = tk.LabelFrame(content, text="4. Inference & Keyhole Tracking", font=("Arial", 12, "bold"), fg="#2E86C1", bg="#EBF5FB")
        f_infer.pack(fill="x", pady=10, ipady=5)
        
        tk.Button(f_infer, text="Start Tracking Tool", command=self.open_tracking_tool, bg="#2E86C1", fg="white", font=("Arial", 12, "bold")).pack(pady=10)

    def go_back(self):
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

    def log(self, msg):
        self.log_area.insert(tk.END, msg + "\n")
        self.log_area.see(tk.END)

    def start_training(self):
        if self.is_training: return
        if not self.source_folders:
            messagebox.showwarning("Missing Data", "Please add at least one source dataset folder.")
            return
        if not self.out_path.get():
            messagebox.showwarning("Missing Output", "Please select an output directory.")
            return

        self.is_training = True
        self.btn_train.config(state='disabled', text="Training in Progress...", bg="gray")
        self.lbl_stage.config(text="Status: Preparing Workspace...", bg="#F39C12") 
        threading.Thread(target=self.run_training_logic, daemon=True).start()

    def on_train_epoch_end(self, trainer):
        current_epoch = trainer.epoch + 1
        total_epochs = trainer.epochs
        status_msg = f"Status: Training... Epoch {current_epoch}/{total_epochs}"
        self.frame.after(0, lambda: self.lbl_stage.config(text=status_msg, bg="#5DADE2"))

    def run_training_logic(self):
        temp_dir = None
        try:
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
                                if not parts: continue
                                cls_id = int(parts[0])
                                # Logic check: if keyhole_only is True, filter class 0. 
                                # Currently defaults to False (train all).
                                if self.keyhole_only.get():
                                    if cls_id == 0: fout.write(line)
                                else:
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
            names_dict = {0: 'Keyhole'} if self.keyhole_only.get() else {0: 'Keyhole', 1: 'KH-Pore', 2: 'Gas-Pore', 3: 'Bubble'}

            yaml_content = {
                'path':  str(temp_dir.absolute()).replace('\\', '/'),
                'train': 'images/train',
                'val':   'images/val',
                'names': names_dict
            }
            
            with open(yaml_path, 'w') as f:
                yaml.dump(yaml_content, f, sort_keys=False)

            # 5. RUN YOLO (SEGMENTATION MODEL)
            self.frame.after(0, lambda: self.lbl_stage.config(text="Status: Training Model...", bg="#5DADE2"))
            self.log("\n--- Step 2: Running YOLOv8 (Segmentation) ---")
            
            from ultralytics import YOLO
            model = YOLO("yolov8n-seg.pt") 
            model.add_callback("on_train_epoch_end", self.on_train_epoch_end)

            results = model.train(
                data=str(yaml_path),
                epochs=self.epochs.get(),
                imgsz=self.img_size.get(),
                batch=self.batch.get(),
                project=str(temp_dir / "runs").replace('\\', '/'),
                name="train_result",
                workers=0
            )
            
            self.log("\n✅ TRAINING COMPLETE!")

            # 6. SAVE RESULTS
            self.log("Copying results to Output Folder...")
            final_out = Path(self.out_path.get()) / "training_results"
            
            if final_out.exists(): shutil.rmtree(final_out, ignore_errors=True)
            
            source_res = Path(results.save_dir)
            shutil.copytree(source_res, final_out)
            
            self.frame.after(0, lambda: self.lbl_stage.config(text="Status: Finished!", bg="#58D68D"))
            messagebox.showinfo("Success", f"Training Finished!\nResults saved to:\n{final_out}")

        except Exception as e:
            self.log(f"\nCRITICAL ERROR: {e}")
            self.frame.after(0, lambda: self.lbl_stage.config(text="Status: Failed", bg="#E74C3C"))
            import traceback; traceback.print_exc()
        finally:
            self.is_training = False
            self.frame.after(0, lambda: self.btn_train.config(state='normal', text="MERGE DATASETS & START TRAINING", bg="#8E44AD"))            

    # --- TRACKING TOOL ---
    def open_tracking_tool(self):
        TrackingViewer(self.frame)

# ==============================================================================
# TRACKING VIEWER
# ==============================================================================
class TrackingViewer:
    def __init__(self, master):
        self.window = tk.Toplevel(master)
        self.window.title("YOLO Keyhole Tracker")
        self.window.geometry("1000x800")
        self.window.configure(bg="#2C3E50")
        
        self.model = None
        self.image_files = []
        self.current_idx = 0
        self.csv_path = None
        self.is_tracking = False
        
        # UI Layout
        ctrl = tk.Frame(self.window, bg="#34495E", pady=10)
        ctrl.pack(side="top", fill="x")
        
        tk.Button(ctrl, text="1. Load Model (.pt)", command=self.load_model, bg="#8E44AD", fg="white", font=("Arial", 10, "bold")).pack(side="left", padx=10)
        tk.Button(ctrl, text="2. Load Image Folder", command=self.load_images, bg="#2980B9", fg="white", font=("Arial", 10, "bold")).pack(side="left", padx=10)
        self.btn_track = tk.Button(ctrl, text="3. Start Tracking", command=self.start_tracking, bg="#27AE60", fg="white", font=("Arial", 10, "bold"), state="disabled")
        self.btn_track.pack(side="left", padx=10)
        
        self.lbl_info = tk.Label(ctrl, text="Load Model & Images to begin", bg="#34495E", fg="white", font=("Arial", 10))
        self.lbl_info.pack(side="left", padx=20)

        # Canvas for Display
        self.canvas = tk.Label(self.window, bg="black")
        self.canvas.pack(expand=True, fill="both", padx=10, pady=10)

    def load_model(self):
        try:
            path = filedialog.askopenfilename(filetypes=[("YOLO Model", "*.pt")])
            if path:
                from ultralytics import YOLO
                self.model = YOLO(path)
                self.lbl_info.config(text=f"Model Loaded: {os.path.basename(path)}")
                self.check_ready()
        except Exception as e:
            messagebox.showerror("Model Error", f"Failed to load model:\n{e}")

    def load_images(self):
        folder = filedialog.askdirectory(title="Select Image Folder")
        if not folder: return

        self.image_files = []
        valid_exts = ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff', '*.bmp']
        path_obj = Path(folder)
        
        # Use a set to auto-remove duplicates (e.g., .png vs .PNG)
        found_set = set()
        
        for ext in valid_exts:
            # On Windows, glob is case-insensitive, so we must filter duplicates manually
            for f in path_obj.glob(ext):
                found_set.add(str(f))
            for f in path_obj.glob(ext.upper()):
                found_set.add(str(f))
            
        # Sort naturally (1, 2, 10...)
        self.image_files = natsorted(list(found_set))
        
        if self.image_files:
            self.lbl_info.config(text=f"Images: {len(self.image_files)} loaded", fg="white")
            self.check_ready()
            self.current_idx = 0
            self.show_preview()
        else:
            messagebox.showwarning("Empty", f"No images found in folder.")

    def check_ready(self):
        if self.model and self.image_files:
            self.btn_track.config(state="normal")

    def show_preview(self):
        if not self.image_files: return
        try:
            img = cv2.imread(self.image_files[0])
            self.update_canvas(img)
        except: pass

    def start_tracking(self):
        if self.is_tracking: return
        
        # Ask for CSV output location
        self.csv_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV File", "*.csv")])
        if not self.csv_path: return
        
        # Initialize CSV Header
        try:
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                # --- CHANGE HERE: Added "Polygon" to the header list ---
                writer.writerow(["Frame_ID", "Center_X", "Center_Y", "Width", "Height", "Confidence", "Polygon"])
        except Exception as e:
            messagebox.showerror("CSV Error", f"Could not create file:\n{e}")
            return
            
        self.is_tracking = True
        self.current_idx = 0
        self.btn_track.config(state="disabled", text="Running...")
        self.process_next_frame()

    def process_next_frame(self):
        # Stop Condition
        if not self.is_tracking or self.current_idx >= len(self.image_files):
            self.is_tracking = False
            self.btn_track.config(state="normal", text="3. Start Tracking")
            messagebox.showinfo("Done", f"Tracking Completed!\nData saved to: {self.csv_path}")
            return
            
        try:
            img_path = self.image_files[self.current_idx]
            
            # Inference
            results = self.model.predict(img_path, verbose=False)
            result = results[0]
            
            # 1. Visualization
            plot_img = result.plot() # BGR numpy array
            
            # 2. Data Extraction
            boxes = result.boxes
            masks = result.masks
            
            best_conf = 0
            best_box = None
            best_poly_str = ""
            
            # Iterate by index to access both box and mask
            if boxes is not None:
                for i, box in enumerate(boxes):
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    if cls_id == 0 and conf > best_conf:
                        best_conf = conf
                        best_box = box.xywh[0] # center_x, center_y, w, h
                        
                        if masks is not None:
                            poly_coords = masks.xy[i]
                            
                            best_poly_str = str(poly_coords.tolist()) 
            
            # 3. Save to CSV
            with open(self.csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                if best_box is not None:
                    x, y, w, h = best_box.tolist()
                    writer.writerow([self.current_idx, x, y, w, h, best_conf, best_poly_str])
                else:
                    writer.writerow([self.current_idx, "NaN", "NaN", "NaN", "NaN", "0", ""])

            # 4. Update UI
            self.update_canvas(plot_img)
            self.current_idx += 1
            
            # Loop (1ms delay)
            self.window.after(1, self.process_next_frame)

        except Exception as e:
            self.is_tracking = False
            self.btn_track.config(state="normal", text="3. Start Tracking")
            messagebox.showerror("Tracking Crash", f"Error on frame {self.current_idx}:\n{e}")
            print(e)

    def update_canvas(self, cv_img):

        if cv_img is None: return
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        
        # Resize logic to fit window 
        disp_h = 700
        scale = disp_h / h
        disp_w = int(w * scale)
        
        resized = cv2.resize(rgb, (disp_w, disp_h))
        
        photo = ImageTk.PhotoImage(Image.fromarray(resized))
        self.canvas.config(image=photo)
        self.canvas.image = photo