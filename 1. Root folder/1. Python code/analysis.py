
from utils import *
import utils

import ast


# ==============================================================================
# AI CHAT WINDOW (OLLAMA)
# ==============================================================================
try:
    import ollama
    HAS_OLLAMA = True
except ImportError:
    HAS_OLLAMA = False
    print("Warning: 'ollama' library not found. Run: pip install ollama")

class AIChatWindow:
    def __init__(self, master, csv_path):
        self.window = tk.Toplevel(master)
        self.window.title("Llama 3 Data Assistant (via Ollama)")
        self.window.geometry("600x700")
        self.window.configure(bg="#F4F6F7")
        
        self.csv_path = csv_path
        self.history = []
        # This is a default model name. Change if you use a different tag (e.g. "llama3:latest")
        self.model_name = "llama3" 

        # --- UI LAYOUT ---
        # 1. Top Bar
        top = tk.Frame(self.window, bg="#D6EAF8", pady=10)
        top.pack(fill="x")
        
        tk.Label(top, text="Model:", bg="#D6EAF8").pack(side="left", padx=10)
        self.ent_model = tk.Entry(top, width=15)
        self.ent_model.insert(0, "llama3")
        self.ent_model.pack(side="left")
        
        tk.Button(top, text="🔄 Connect & Load Data", command=self.connect_and_load, 
                  bg="#2980B9", fg="white").pack(side="left", padx=10)
        
        self.lbl_status = tk.Label(top, text="Status: Idle", bg="#D6EAF8", fg="gray")
        self.lbl_status.pack(side="left", padx=10)

        # 2. Chat Area
        self.txt_chat = scrolledtext.ScrolledText(self.window, state='disabled', bg="white", font=("Arial", 10), wrap="word")
        self.txt_chat.pack(fill="both", expand=True, padx=10, pady=10)
        self.txt_chat.tag_config("user", foreground="blue", font=("Arial", 10, "bold"))
        self.txt_chat.tag_config("ai", foreground="green", font=("Arial", 10, "bold"))
        self.txt_chat.tag_config("sys", foreground="gray", font=("Arial", 9, "italic"))

        # 3. Input Area
        bot = tk.Frame(self.window, bg="#F4F6F7", pady=10)
        bot.pack(fill="x", padx=10)
        self.ent_msg = tk.Entry(bot, font=("Arial", 11))
        self.ent_msg.pack(side="left", fill="x", expand=True)
        self.ent_msg.bind("<Return>", self.send_message)
        tk.Button(bot, text="Send", command=self.send_message, bg="#2ECC71", fg="white").pack(side="right", padx=5)

    def connect_and_load(self):
        if not HAS_OLLAMA:
            messagebox.showerror("Error", "Ollama python library not found.\nRun: pip install ollama")
            return

        self.model_name = self.ent_model.get().strip()
        self.lbl_status.config(text="Connecting...", fg="orange")
        self.window.update()
        
        threading.Thread(target=self._init_context, daemon=True).start()

    def _init_context(self):
        try:
            
            self.window.after(0, lambda: self.lbl_status.config(text=f"Checking {self.model_name}...", fg="orange"))
            
            ollama.pull(self.model_name)
            
            # Prepare Data Context
            context_str = self.analyze_csv_context()
            self.history = [
                {"role": "system", "content": f"You are a helpful scientific assistant. Here is a statistical summary of the geometric data:\n{context_str}\nAnswer questions based on this data."}
            ]
            
            # Success updates
            self.window.after(0, lambda: self.lbl_status.config(text=f"Connected to {self.model_name}", fg="green"))
            self.window.after(0, lambda: self.append_chat("System", "Data loaded. Ready to chat.", "sys"))
            
        except Exception as e:
            err = str(e)
            if "Connection refused" in err:
                err = "Is Ollama running? (Run 'ollama serve')"
            self.window.after(0, lambda: self.lbl_status.config(text="Connection Failed", fg="red"))
            self.window.after(0, lambda: self.append_chat("System", f"Error: {err}", "sys"))

    def analyze_csv_context(self):
        """Reads CSV, calculates full stats, and summarizes for AI."""
        if not os.path.exists(self.csv_path): return "No CSV file found."
        try:
            # Initialize stats
            stats = {
                "count": 0, 
                "w_sum": 0, "h_sum": 0, 
                "min_w": float('inf'), "max_w": float('-inf'), # Proper init
                "min_h": float('inf'), "max_h": float('-inf')
            }
            rows = []
            
            with open(self.csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Filter bad rows (NaN or empty)
                    if row.get('Center_X') == "NaN" or not row.get('Width'): 
                        continue
                    
                    try:
                        w = float(row['Width'])
                        h = float(row['Height'])
                    except ValueError:
                        continue

                    # Update Stats
                    stats["count"] += 1
                    stats["w_sum"] += w
                    stats["h_sum"] += h
                    
                    if w < stats["min_w"]: stats["min_w"] = w
                    if w > stats["max_w"]: stats["max_w"] = w
                    if h < stats["min_h"]: stats["min_h"] = h
                    if h > stats["max_h"]: stats["max_h"] = h                 
                    if stats["count"] <= 5: 
                        row_clean = row.copy()
                        if 'Polygon' in row_clean: del row_clean['Polygon']
                        rows.append(str(row_clean))
            
            if stats["count"] == 0: return "Empty Data."
            
            avg_w = stats["w_sum"] / stats["count"]
            avg_h = stats["h_sum"] / stats["count"]
            
            
            summary = (
                f"Dataset Summary:\n"
                f"- Total Frames: {stats['count']}\n"
                f"- Width: Min={stats['min_w']:.2f}, Max={stats['max_w']:.2f}, Avg={avg_w:.2f}\n"
                f"- Depth: Min={stats['min_h']:.2f}, Max={stats['max_h']:.2f}, Avg={avg_h:.2f}\n\n"
                f"First 5 Rows of Data (for reference):\n{rows}"
            )
            return summary

        except Exception as e: return f"Error reading CSV: {e}"

    def send_message(self, event=None):
        msg = self.ent_msg.get().strip()
        if not msg: return
        self.ent_msg.delete(0, tk.END)
        self.append_chat("You", msg, "user")
        self.history.append({"role": "user", "content": msg})
        
        threading.Thread(target=self._run_inference, daemon=True).start()

    def _run_inference(self):
        try:
            response = ollama.chat(model=self.model_name, messages=self.history)
            reply = response['message']['content']
            
            self.history.append({"role": "assistant", "content": reply})
            self.window.after(0, lambda: self.append_chat("Llama", reply, "ai"))
        except Exception as e:
            # 1. Convert error to string immediately
            err_msg = str(e) 
    
            # 2. Use the string variable, not 'e'
            self.window.after(0, lambda: self.append_chat("System", f"Inference Error: {err_msg}", "sys"))

    def append_chat(self, sender, msg, tag):
        self.txt_chat.config(state='normal')
        self.txt_chat.insert(tk.END, f"\n{sender}: ", tag)
        self.txt_chat.insert(tk.END, f"{msg}\n")
        self.txt_chat.see(tk.END)
        self.txt_chat.config(state='disabled')


# ==============================================================================
# GEOMETRIC DATA PLOTTER
# ==============================================================================
class DataPlotterApp:
    def __init__(self, parent_frame, on_back):
        self.frame = tk.Frame(parent_frame)
        self.frame.pack(fill="both", expand=True)
        self.on_back = on_back
        
        self.csv_path = tk.StringVar()
        self.img_folder_path = tk.StringVar()
        self.pixel_cal = tk.DoubleVar(value=1.0)
        self.obj_type = tk.StringVar(value="Keyhole")
        
        # Data & State
        self.data_map = {}   
        self.image_files = [] 
        self.current_idx = 0
        
        # Zoom & Pan State
        self.plotter_zoom = 1.0
        self.pan_x = 0; self.pan_y = 0
        self.drag_start_x = 0; self.drag_start_y = 0
        
        self.fig = None; self.canvas = None; self.tk_img = None
        
        self.setup_ui()

    def setup_ui(self):
        # Header
        top = tk.Frame(self.frame, bg="#FDEDEC", height=60)
        top.pack(fill="x")
        tk.Button(top, text="← Back", command=self.go_back, bg="#95A5A6", fg="white", font=("Arial", 11)).pack(side="left", padx=20)
        tk.Label(top, text="Tool: Geometric Data Plotter", bg="#FDEDEC", fg="#C0392B", font=("Arial", 18, "bold")).pack(side="left", padx=20)

        # Control Panel
        ctrl = tk.Frame(self.frame, padx=20, pady=5)
        ctrl.pack(fill="x")
        
        f_file = tk.LabelFrame(ctrl, text="1. Data Source", font=("Arial", 10, "bold"), fg="#2C3E50")
        f_file.pack(fill="x", pady=2)
        tk.Label(f_file, text="CSV:").pack(side="left")
        tk.Entry(f_file, textvariable=self.csv_path, width=40).pack(side="left", padx=5)
        tk.Button(f_file, text="Browse", command=self.load_csv).pack(side="left")
        tk.Label(f_file, text="Images:").pack(side="left", padx=10)
        tk.Entry(f_file, textvariable=self.img_folder_path, width=40).pack(side="left", padx=5)
        tk.Button(f_file, text="Browse", command=self.load_images).pack(side="left")

        f_set = tk.LabelFrame(ctrl, text="2. Settings & Tools", font=("Arial", 10, "bold"), fg="#2C3E50")
        f_set.pack(fill="x", pady=2)
        tk.Label(f_set, text="Cal (µm/px):").pack(side="left", padx=5)
        tk.Entry(f_set, textvariable=self.pixel_cal, width=6).pack(side="left")
        tk.Label(f_set, text="Object:").pack(side="left", padx=10)
        tk.OptionMenu(f_set, self.obj_type, "Keyhole", "Pore", "Bubble").pack(side="left")
        
        # --- BUTTONS ---
        # REMOVED "Advanced Analysis" from here to prevent errors.
        
        btn_box = tk.Frame(f_set)
        btn_box.pack(side="right", padx=10)
        
        # AI CHAT BUTTON
        if HAS_OLLAMA:
            tk.Button(btn_box, text="🤖 Chat with Data", bg="#8E44AD", fg="white", font=("Arial", 10, "bold"), 
                      command=self.launch_ai_chat).pack(side="left", padx=10)
        else:
            tk.Label(btn_box, text="(Install 'ollama' library for AI)", fg="gray").pack(side="left", padx=10)

        tk.Button(btn_box, text="GENERATE PLOTS", bg="#E67E22", fg="white", font=("Arial", 11, "bold"), 
                  command=self.generate_plots).pack(side="left")

        # Main Content
        content = tk.Frame(self.frame)
        content.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Left: Plots
        self.plot_frame = tk.Frame(content, bg="white", width=600)
        self.plot_frame.pack(side="left", fill="both", expand=True)

        # Right: Visualization
        vis_frame = tk.LabelFrame(content, text="Visual Validation", font=("Arial", 11, "bold"), width=500)
        vis_frame.pack(side="right", fill="both", padx=(10,0))
        vis_frame.pack_propagate(False)

        # -- Image Canvas --
        self.vis_canvas = tk.Canvas(vis_frame, bg="#D5D8DC", cursor="fleur")
        self.vis_canvas.pack(side="top", fill="both", expand=True)
        
        self.vis_canvas.bind("<ButtonPress-1>", self.start_pan)
        self.vis_canvas.bind("<B1-Motion>", self.do_pan)
        self.vis_canvas.bind("<MouseWheel>", self.on_mousewheel) 
        self.vis_canvas.bind("<Button-4>", self.zoom_in)
        self.vis_canvas.bind("<Button-5>", self.zoom_out)
        
        self.lbl_sample_info = tk.Label(vis_frame, text="Load data to view stats", bg="#D5D8DC", font=("Consolas", 10))
        self.lbl_sample_info.pack(fill="x")

        # -- Controls --
        c_frame = tk.Frame(vis_frame)
        c_frame.pack(fill="x", pady=5)
        
        tk.Button(c_frame, text="< Prev", command=self.prev_image).pack(side="left", padx=5)
        tk.Button(c_frame, text="🎲 Random", command=self.pick_random, bg="#2980B9", fg="white").pack(side="left", padx=5)
        tk.Button(c_frame, text="Next >", command=self.next_image).pack(side="left", padx=5)
        
        tk.Label(c_frame, text="| Zoom:").pack(side="left", padx=5)
        tk.Button(c_frame, text="+", command=self.zoom_in, width=3).pack(side="left")
        tk.Button(c_frame, text="-", command=self.zoom_out, width=3).pack(side="left")
        tk.Button(c_frame, text="Rst", command=self.reset_view, width=3).pack(side="left")

        # -- File List --
        self.file_listbox = tk.Listbox(vis_frame, height=8, selectmode="browse", font=("Consolas", 9))
        self.file_listbox.pack(side="bottom", fill="x")
        self.file_listbox.bind("<<ListboxSelect>>", self.on_file_select)
        
        self.frame.bind_all("<plus>", self.zoom_in)
        self.frame.bind_all("=", self.zoom_in)
        self.frame.bind_all("<minus>", self.zoom_out)

    def launch_ai_chat(self):
        csv_f = self.csv_path.get()
        if not csv_f or not os.path.exists(csv_f):
            messagebox.showwarning("No Data", "Please select a valid CSV file first.")
            return
        AIChatWindow(self.frame, csv_f)

    # ... (Rest of DataPlotterApp methods: go_back, load_csv, etc. remain unchanged) ...
    def go_back(self):
        self.frame.unbind_all("<plus>"); self.frame.unbind_all("="); self.frame.unbind_all("<minus>")
        self.frame.destroy(); self.on_back()
    def load_csv(self):
        p = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if p: self.csv_path.set(p)
    def load_images(self):
        d = filedialog.askdirectory()
        if d: self.img_folder_path.set(d); self.populate_file_list()
    def populate_file_list(self):
        folder = self.img_folder_path.get()
        if not os.path.exists(folder): return
        
        path_obj = Path(folder)
        exts = ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff', '*.bmp']
        
        found_set = set()
        
        for ext in exts:

            for f in path_obj.glob(ext):
                found_set.add(str(f))
            for f in path_obj.glob(ext.upper()):
                found_set.add(str(f))
        

        self.image_files = natsorted(list(found_set))
        
        if not self.image_files: 
            messagebox.showwarning("Warning", f"No images found in:\n{folder}")
            return
            
        self.file_listbox.delete(0, tk.END)
        for f in self.image_files: 
            self.file_listbox.insert(tk.END, os.path.basename(f))
            
        self.current_idx = 0
        self.refresh_image()
    
    def calculate_shoelace_area(self, coords):
        n = len(coords)
        if n < 3: return 0.0
        
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += coords[i][0] * coords[j][1]
            area -= coords[j][0] * coords[i][1]
            
        return abs(area) / 2.0

    def generate_plots(self):
        csv_file = self.csv_path.get()
        if not os.path.exists(csv_file): 
            messagebox.showerror("Error", "CSV file not found.")
            return

        frames, widths, heights, areas, ratios = [], [], [], [], []
        cal = self.pixel_cal.get()

        self.data_map = {} 

        try:
            with open(csv_file, 'r') as f:
                reader = csv.DictReader(f)
                reader.fieldnames = [name.strip() for name in reader.fieldnames]

                for row in reader:
                    if 'Width' not in row: continue 
                    if 'Center_X' in row and row['Center_X'] == "NaN": continue
                    
                    fid = int(row['Frame_ID'])
                    w_px, h_px = float(row['Width']), float(row['Height'])
                    

                    cx = float(row.get('Center_X', 0.0))
                    cy = float(row.get('Center_Y', 0.0))

                    w_um, h_um = w_px * cal, h_px * cal
                    
                    bbox_area = w_um * h_um
                    final_area = bbox_area 
                    
                    poly_px = [] 
                    
                    if 'Polygon' in row and len(row['Polygon']) > 5:
                        try:
                            poly_raw = ast.literal_eval(row['Polygon'])
                            if isinstance(poly_raw, list) and len(poly_raw) > 2:
                                if isinstance(poly_raw[0], (list, tuple)):
                                    poly_px = [(float(p[0]), float(p[1])) for p in poly_raw]
                                elif isinstance(poly_raw[0], (int, float)):
                                    for i in range(0, len(poly_raw), 2):
                                        poly_px.append((float(poly_raw[i]), float(poly_raw[i+1])))

                            poly_um = [(p[0]*cal, p[1]*cal) for p in poly_px]
                            shoelace_val = self.calculate_shoelace_area(poly_um)
                            
                            if shoelace_val > 0:
                                final_area = shoelace_val

                        except Exception as e:
                            print(f"Frame {fid} Parsing Error: {e}")
                    

                    self.data_map[fid] = {
                        'cx': cx, 
                        'cy': cy,
                        'w': w_px, 
                        'h': h_px,
                        'poly': poly_px 
                    }

                    frames.append(fid)
                    widths.append(w_um)
                    heights.append(h_um)
                    areas.append(final_area) 
                    ratios.append(h_um / w_um if w_um > 0 else 0)

            if self.canvas: self.canvas.get_tk_widget().destroy()
            self.fig, axs = plt.subplots(2, 2, figsize=(6, 5), dpi=90)
            self.fig.suptitle(f"{self.obj_type.get()} Analysis", fontsize=12)
            
            axs[0, 0].plot(frames, heights, 'b', lw=1); axs[0, 0].set_title("Depth (µm)")
            axs[0, 1].plot(frames, widths, 'r', lw=1); axs[0, 1].set_title("Width (µm)")
            axs[1, 0].plot(frames, areas, 'g', lw=1); axs[1, 0].set_title("Area (µm²)") 
            axs[1, 1].plot(frames, ratios, 'm', lw=1); axs[1, 1].set_title("Aspect Ratio")
            
            for ax in axs.flat: ax.grid(True)
            plt.tight_layout()
            
            self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(fill="both", expand=True)

            self.refresh_image()

        except Exception as e:
            messagebox.showerror("Plot Error", str(e))
            print(f"Critical Plot Error: {e}")

    def prev_image(self):
        if self.current_idx > 0: self.current_idx -= 1; self.refresh_image()
    def next_image(self):
        if self.image_files and self.current_idx < len(self.image_files) - 1: self.current_idx += 1; self.refresh_image()
    def pick_random(self):
        if self.image_files: self.current_idx = random.randint(0, len(self.image_files) - 1); self.refresh_image()
    def on_file_select(self, event):
        sel = self.file_listbox.curselection()
        if sel: self.current_idx = sel[0]; self.refresh_image()
    def zoom_in(self, event=None): self.plotter_zoom *= 1.2; self.refresh_image()
    def zoom_out(self, event=None): 
        self.plotter_zoom /= 1.2; 
        if self.plotter_zoom < 0.1: self.plotter_zoom = 0.1
        self.refresh_image()
    def on_mousewheel(self, event): 
        if event.delta > 0: self.zoom_in()
        else: self.zoom_out()
    def reset_view(self): self.plotter_zoom = 1.0; self.pan_x = 0; self.pan_y = 0; self.refresh_image()
    def start_pan(self, event): self.drag_start_x = event.x; self.drag_start_y = event.y
    def do_pan(self, event):
        self.pan_x += event.x - self.drag_start_x; self.pan_y += event.y - self.drag_start_y
        self.drag_start_x = event.x; self.drag_start_y = event.y; self.refresh_image()
    def refresh_image(self):
        if not self.image_files: return
        
        self.file_listbox.selection_clear(0, tk.END)
        self.file_listbox.selection_set(self.current_idx)
        self.file_listbox.see(self.current_idx)
        
        path = self.image_files[self.current_idx]
        cv_img = cv2.imread(path)
        if cv_img is None: return
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        
        frame_id = self.current_idx
        overlay_txt = "No Data"
        
        if self.data_map and frame_id in self.data_map:
            d = self.data_map[frame_id]
            cx, cy = d['cx'], d['cy']
            w, h = d['w'], d['h']
            
            x1, y1 = int(cx - w/2), int(cy - h/2)
            x2, y2 = int(cx + w/2), int(cy + h/2)
            cv2.rectangle(cv_img, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.line(cv_img, (x1, int(cy)), (x2, int(cy)), (255, 0, 0), 2)
            cv2.line(cv_img, (int(cx), y1), (int(cx), y2), (0, 0, 255), 2)
            
            if 'poly' in d and d['poly']:
                try:
                    pts = np.array(d['poly'], dtype=np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    cv2.polylines(cv_img, [pts], isClosed=True, color=(50, 205, 50), thickness=2)
                except Exception as e:
                    print(f"Poly draw error on frame {frame_id}: {e}")

            cal = self.pixel_cal.get()
            overlay_txt = f"W: {w*cal:.1f}µm | D: {h*cal:.1f}µm"
        
        self.lbl_sample_info.config(text=f"Frame: {frame_id} | {overlay_txt}")
        
        pil_img = Image.fromarray(cv_img)
        cw = self.vis_canvas.winfo_width()
        ch = self.vis_canvas.winfo_height()
        
        if cw < 10: cw, ch = 500, 400
        
        iw, ih = pil_img.size
        base_scale = min(cw/iw, ch/ih)
        final_scale = base_scale * self.plotter_zoom
        
        new_w = int(iw * final_scale)
        new_h = int(ih * final_scale)
        pil_img = pil_img.resize((new_w, new_h), Image.Resampling.NEAREST)
        
        self.tk_img = ImageTk.PhotoImage(pil_img)
        
        center_x = (cw // 2) + self.pan_x
        center_y = (ch // 2) + self.pan_y
        
        self.vis_canvas.delete("all")
        self.vis_canvas.create_image(center_x, center_y, anchor="center", image=self.tk_img)
        
# ==============================================================================
# EVENT TAGGER APP 
# ==============================================================================
class EventTaggerApp:
    def __init__(self, parent_frame, on_back):
        self.frame = tk.Frame(parent_frame)
        self.frame.pack(fill="both", expand=True)
        self.on_back = on_back
        
        self.image_list = []
        self.current_idx = 0
        self.raw_image = None
        
        # Navigation State
        self.zoom_level = 1.0
        self.pan_x = 0; self.pan_y = 0
        self.scale = 1.0; self.off_x = 0; self.off_y = 0
        self.drag_start_x = 0; self.drag_start_y = 0
        
        # Data: { frame_idx: [ (x_norm, y_norm, type_string), ... ] }
        self.events_data = {} 
        self.event_type = tk.StringVar(value="Pore")
        self.new_event_name = tk.StringVar()

        self.EVENT_TYPES = ["Pore", "Bubble", "Spatter", "Plume"]
        self.COLORS = {"Pore": "red", "Bubble": "blue", "Spatter": "green", "Plume": "magenta"}

        self.setup_ui()

    def setup_ui(self):
        # --- Header ---
        top = tk.Frame(self.frame, bg="#FDEDEC", height=60)
        top.pack(fill="x")
        tk.Button(top, text="← Back", command=self.go_back, bg="#95A5A6", fg="white", font=("Arial", 11)).pack(side="left", padx=20)
        tk.Label(top, text="Tool: Event Tagger (Coordinate Marker)", bg="#FDEDEC", fg="#C0392B", font=("Arial", 18, "bold")).pack(side="left", padx=20)
        
        # --- Toolbar ---
        toolbar = tk.Frame(self.frame, pady=5, padx=10)
        toolbar.pack(fill="x")
        
        # Load/Save
        tk.Button(toolbar, text="1. Load Image Folder", command=self.load_folder, bg="#2980B9", fg="white").pack(side="left", padx=5)
        tk.Button(toolbar, text="💾 Export CSV", command=self.save_csv, bg="#27AE60", fg="white", font=("Arial", 10, "bold")).pack(side="left", padx=5)
        
        tk.Button(toolbar, text="⚡ Advanced Analysis", command=self.launch_advanced_analysis, 
                  bg="#8E44AD", fg="white", font=("Arial", 10, "bold")).pack(side="left", padx=20)
        
        # Event Type Selection Area
        tk.Label(toolbar, text="| Type:", font=("Arial", 11, "bold")).pack(side="left", padx=5)
        
        self.f_radios = tk.Frame(toolbar)
        self.f_radios.pack(side="left")
        
        # Initial Radios
        self.refresh_radio_buttons()

        # Custom Event Entry
        tk.Label(toolbar, text="| Add New:", fg="gray").pack(side="left", padx=5)
        tk.Entry(toolbar, textvariable=self.new_event_name, width=10).pack(side="left", padx=2)
        tk.Button(toolbar, text="+", command=self.add_custom_event, bg="#5D6D7E", fg="white", width=2).pack(side="left", padx=2)

        tk.Label(toolbar, text="[Left Click]=Mark | [Right Click]=Remove", fg="gray").pack(side="right", padx=20)

        # --- Main Layout ---
        main_split = tk.Frame(self.frame)
        main_split.pack(fill="both", expand=True)

        # Left: Canvas
        self.canvas = tk.Canvas(main_split, bg="black", cursor="cross")
        self.canvas.pack(side="left", fill="both", expand=True)
        
        # Right: Sidebar (Split into Events and File List)
        sidebar = tk.Frame(main_split, width=280, bg="#ECF0F1")
        sidebar.pack(side="right", fill="y")
        sidebar.pack_propagate(False)
        
        # 1. Event Log (Top Half)
        f_log = tk.LabelFrame(sidebar, text="Events on this Frame", bg="#ECF0F1", font=("Arial", 9, "bold"))
        f_log.pack(side="top", fill="both", expand=True, padx=2, pady=2)
        
        sb_log = tk.Scrollbar(f_log)
        sb_log.pack(side="right", fill="y")
        self.log_list = tk.Listbox(f_log, font=("Consolas", 9), height=10, yscrollcommand=sb_log.set)
        self.log_list.pack(side="left", fill="both", expand=True)
        sb_log.config(command=self.log_list.yview)

        # 2. File List (Bottom Half)
        f_files = tk.LabelFrame(sidebar, text="File List (Select to View)", bg="#ECF0F1", font=("Arial", 9, "bold"))
        f_files.pack(side="bottom", fill="both", expand=True, padx=2, pady=2)
        
        sb_files = tk.Scrollbar(f_files)
        sb_files.pack(side="right", fill="y")
        self.file_listbox = tk.Listbox(f_files, font=("Consolas", 9), selectmode="browse", yscrollcommand=sb_files.set)
        self.file_listbox.pack(side="left", fill="both", expand=True)
        sb_files.config(command=self.file_listbox.yview)
        
        self.file_listbox.bind("<<ListboxSelect>>", self.on_file_select)

        # Bindings
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<Button-3>", self.remove_last_event)
        
        # Pan/Zoom
        self.canvas.bind("<ButtonPress-2>", self.start_pan)
        self.canvas.bind("<B2-Motion>", self.do_pan)
        self.frame.bind_all("<plus>", self.zoom_in)
        self.frame.bind_all("=", self.zoom_in)
        self.frame.bind_all("<minus>", self.zoom_out)
        
        self.frame.bind_all("<Left>", self.prev_image)
        self.frame.bind_all("<Right>", self.next_image)
        self.canvas.bind("<Configure>", self.on_resize)

    def go_back(self):
        self.frame.unbind_all("<Left>"); self.frame.unbind_all("<Right>"); self.frame.unbind_all("<plus>"); self.frame.unbind_all("<minus>")
        self.frame.destroy()
        self.on_back()

    def launch_advanced_analysis(self):
        if not self.image_list:
            messagebox.showwarning("Warning", "Please load an image folder first.")
            return
            
        # Launch the new window, passing the current image list and current index
        AdvancedAnalysisWindow(self.frame, self.image_list, self.current_idx)

    def refresh_radio_buttons(self):
        # Clear existing radios
        for widget in self.f_radios.winfo_children():
            widget.destroy()
            
        # Re-create radios based on current EVENT_TYPES list
        for et in self.EVENT_TYPES:
            col = self.COLORS.get(et, "black")
            tk.Radiobutton(self.f_radios, text=et, variable=self.event_type, value=et, 
                           fg=col, font=("Arial", 10, "bold")).pack(side="left", padx=2)

    
    def launch_advanced_analysis(self):
        if not self.image_list:
            messagebox.showwarning("Warning", "Please load an image folder first.")
            return
            
        # Launch the new window, passing the current image list and current index
        AdvancedAnalysisWindow(self.frame, self.image_list, self.current_idx)

    def add_custom_event(self):
        new_type = self.new_event_name.get().strip()
        if not new_type: return
        
        if new_type in self.EVENT_TYPES:
            messagebox.showinfo("Info", "Event type already exists.")
            return

        # Generate a random bright color
        import random
        r = lambda: random.randint(50, 255)
        new_color = '#%02X%02X%02X' % (r(),r(),r())
        
        self.EVENT_TYPES.append(new_type)
        self.COLORS[new_type] = new_color
        
        self.refresh_radio_buttons()
        self.event_type.set(new_type) # Auto-select new type
        self.new_event_name.set("") # Clear entry

    def load_folder(self):
        d = filedialog.askdirectory()
        if not d: return
        
        self.image_list = []
        path_obj = Path(d)
        exts = ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff', '*.bmp']
        found = []
        
        for ext in exts:
            found.extend(list(path_obj.glob(ext)))
            found.extend(list(path_obj.glob(ext.upper())))
            

        unique_files = set([str(f) for f in found])
        self.image_list = natsorted(list(unique_files))
        # ----------------------
        
        if not self.image_list: 
            messagebox.showerror("Error", f"No images found in:\n{d}")
            return
        
        # Populate File List
        self.file_listbox.delete(0, tk.END)
        for f in self.image_list:
            self.file_listbox.insert(tk.END, os.path.basename(f))
            
        self.current_idx = 0
        self.load_image()

    def on_file_select(self, event):
        sel = self.file_listbox.curselection()
        if sel:
            self.current_idx = sel[0]
            self.load_image()

    def load_image(self):
        if not self.image_list: return
        
        self.file_listbox.selection_clear(0, tk.END)
        self.file_listbox.selection_set(self.current_idx)
        self.file_listbox.see(self.current_idx)

        path = self.image_list[self.current_idx]
        img = imread_safe(path)
        if img is None: return
        self.raw_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.show_image()
        self.update_log()

    def show_image(self):
        if self.raw_image is None: return
        h, w = self.raw_image.shape[:2]
        cw = self.canvas.winfo_width(); ch = self.canvas.winfo_height()
        if cw < 50: cw = 800; ch = 600
        
        base_scale = min(cw/w, ch/h)
        self.scale = base_scale * self.zoom_level
        new_w = int(w * self.scale); new_h = int(h * self.scale)
        
        self.off_x = (cw - new_w) // 2 + self.pan_x
        self.off_y = (ch - new_h) // 2 + self.pan_y
        
        pil = Image.fromarray(self.raw_image)
        pil = pil.resize((new_w, new_h), Image.Resampling.NEAREST)
        self.tk_img = ImageTk.PhotoImage(pil)
        
        self.canvas.delete("all")
        self.canvas.create_image(self.off_x, self.off_y, anchor="nw", image=self.tk_img)
        
        self.canvas.create_text(10, 10, text=f"Frame: {self.current_idx} / {len(self.image_list)}", anchor="nw", fill="white", font=("Arial", 12, "bold"))
        
        if self.current_idx in self.events_data:
            for (nx, ny, etype) in self.events_data[self.current_idx]:
                cx = int(nx * w * self.scale) + self.off_x
                cy = int(ny * h * self.scale) + self.off_y
                col = self.COLORS.get(etype, "white")
                r = 5
                self.canvas.create_oval(cx-r, cy-r, cx+r, cy+r, fill=col, outline="black")
                self.canvas.create_text(cx, cy-15, text=etype, fill=col, font=("Arial", 8, "bold"))

    def on_click(self, event):
        if self.raw_image is None: return
        h, w = self.raw_image.shape[:2]
        
        rx = (event.x - self.off_x) / self.scale
        ry = (event.y - self.off_y) / self.scale
        
        if 0 <= rx < w and 0 <= ry < h:
            nx = rx / w
            ny = ry / h
            etype = self.event_type.get()
            
            if self.current_idx not in self.events_data:
                self.events_data[self.current_idx] = []
            
            self.events_data[self.current_idx].append((nx, ny, etype))
            self.show_image()
            self.update_log()

    def remove_last_event(self, event):
        if self.current_idx in self.events_data and self.events_data[self.current_idx]:
            self.events_data[self.current_idx].pop()
            self.show_image()
            self.update_log()

    def update_log(self):
        self.log_list.delete(0, tk.END)
        if self.current_idx in self.events_data:
            for i, (nx, ny, etype) in enumerate(self.events_data[self.current_idx]):
                self.log_list.insert(tk.END, f"{i+1}. {etype} ({nx:.2f}, {ny:.2f})")
                self.log_list.itemconfig(tk.END, {'fg': self.COLORS.get(etype, "black")})

    def save_csv(self):
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if path:
            with open(path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Frame_ID", "Event_Type", "Norm_X", "Norm_Y"])
                for fid in sorted(self.events_data.keys()):
                    for (nx, ny, etype) in self.events_data[fid]:
                        writer.writerow([fid, etype, nx, ny])
            messagebox.showinfo("Saved", "Event log saved successfully.")

    # Nav/Zoom/Pan
    def next_image(self, event=None):
        if self.current_idx < len(self.image_list) - 1:
            self.current_idx += 1
            self.load_image()
    def prev_image(self, event=None):
        if self.current_idx > 0:
            self.current_idx -= 1
            self.load_image()
    def zoom_in(self, event=None):
        self.zoom_level *= 1.2
        self.show_image()
    def zoom_out(self, event=None):
        self.zoom_level /= 1.2
        if self.zoom_level < 1.0: self.zoom_level = 1.0
        self.show_image()
    def start_pan(self, event):
        self.drag_start_x = event.x; self.drag_start_y = event.y
    def do_pan(self, event):
        self.pan_x += event.x - self.drag_start_x
        self.pan_y += event.y - self.drag_start_y
        self.drag_start_x = event.x; self.drag_start_y = event.y
        self.show_image()
    def on_resize(self, event):
        self.show_image()
        
 # ==============================================================================
 # Advanced Analysis App
 # ==============================================================================       


class AdvancedAnalysisWindow:
    def __init__(self, master, image_list, current_idx_ref):
        self.window = tk.Toplevel(master)
        self.window.title("Advanced Pore Analysis (Space-Time Mapped)")
        
        # --- Maximize Window ---
        try:
            self.window.state('zoomed') # Windows
        except:
            self.window.attributes('-zoomed', True) # Linux/Mac
            
        self.window.configure(bg="#2C3E50") 
        
        # Data
        self.image_list = image_list
        self.current_idx = current_idx_ref 
        self.collected_data = [] 
        self.marked_files = set() 
        
        # State Machine
        self.marking_state = 0 
        self.temp_points = {} 
        self.dia_start_pt = None 

        # Image Info
        self.current_img_w = 0  
        self.scan_direction = tk.StringVar(value="L->R") 
        
        # Default Velocity
        self.fixed_velocity_mm_s = 800.0 

        # View State
        self.zoom_level = 1.0
        self.pan_x = 0; self.pan_y = 0
        self.drag_start_x = 0; self.drag_start_y = 0
        self.current_scale = 1.0
        self.offset_x = 0; self.offset_y = 0
        
        # Image Cache for display
        self.tk_img = None

        # --- UI LAYOUT ---
        self.setup_settings_panel()
        
        work_area = tk.Frame(self.window, bg="#2C3E50")
        work_area.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Canvas Frame
        self.canvas_frame = tk.Frame(work_area, bg="black")
        self.canvas_frame.pack(side="left", fill="both", expand=True)
        
        self.canvas = tk.Canvas(self.canvas_frame, bg="black", cursor="tcross")
        self.canvas.pack(fill="both", expand=True)
        
        # Sidebar
        self.right_panel = tk.Frame(work_area, width=350, bg="#ECF0F1")
        self.right_panel.pack(side="right", fill="y", padx=(5,0))
        self.right_panel.pack_propagate(False)
        
        self.setup_controls()
        self.setup_file_list()
        self.setup_data_list()

        # --- BINDINGS ---
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<Configure>", self.on_resize)
        
        self.window.bind("<Left>", self.prev_img)
        self.window.bind("<Right>", self.next_img)

        self.canvas.bind("<Enter>", self._bind_zoom)
        self.canvas.bind("<Leave>", self._unbind_zoom)

        self.canvas.bind("<ButtonPress-3>", self.start_pan)
        self.canvas.bind("<B3-Motion>", self.do_pan)

        self.refresh_image()

    def on_resize(self, event):
        self.refresh_image()

    # --- SMART ZOOM BINDINGS ---
    def _bind_zoom(self, event):
        self.window.bind("<MouseWheel>", self.on_mousewheel) 
        self.window.bind("<Button-4>", self.zoom_in)        
        self.window.bind("<Button-5>", self.zoom_out)       
        self.window.bind("<plus>", self.zoom_in)
        self.window.bind("=", self.zoom_in)
        self.window.bind("<minus>", self.zoom_out)

    def _unbind_zoom(self, event):
        self.window.unbind("<MouseWheel>")
        self.window.unbind("<Button-4>")
        self.window.unbind("<Button-5>")
        self.window.unbind("<plus>")
        self.window.unbind("=")
        self.window.unbind("<minus>")

    def setup_settings_panel(self):
        f = tk.LabelFrame(self.window, text="Parameters", bg="#D6EAF8", font=("Arial", 10, "bold"))
        f.pack(fill="x", padx=10, pady=5)
        
        # 1. Scale
        tk.Label(f, text="Scale (µm/px):", bg="#D6EAF8").pack(side="left", padx=5)
        self.ent_scale = tk.Entry(f, width=8)
        self.ent_scale.insert(0, "2.9216") 
        self.ent_scale.pack(side="left", padx=2)

        # 2. Velocity
        tk.Label(f, text="Vel (mm/s):", bg="#D6EAF8", fg="#C0392B").pack(side="left", padx=10)
        self.ent_velocity = tk.Entry(f, width=6)
        self.ent_velocity.insert(0, "800") 
        self.ent_velocity.pack(side="left", padx=2)

        # 3. Direction
        tk.Label(f, text="| Dir:", bg="#D6EAF8", font=("Arial", 9, "bold")).pack(side="left", padx=10)
        
        # FIX: Define the OptionMenu command properly
        self.om_direction = tk.OptionMenu(f, self.scan_direction, "L->R", "R->L", command=self.on_direction_change)
        self.om_direction.config(width=5, bg="white", font=("Arial", 9))
        self.om_direction.pack(side="left")
        
        tk.Button(f, text="📈 Plot", command=self.open_plotter, 
                  bg="#8E44AD", fg="white").pack(side="right", padx=10, pady=2)

    def on_direction_change(self, val):
        """Called when user changes the dropdown. Resets active measurement only."""
        # Only reset if we are in the middle of a measurement (state > 0)
        if self.marking_state > 0:
            self.reset_state()
            # We use after to avoid blocking the UI thread or causing race conditions
            self.window.after(100, lambda: messagebox.showinfo("Reset", f"Direction changed to {val}.\nCurrent measurement reset."))
        
        # Always refresh image to show the new direction label
        self.refresh_image()

    def setup_controls(self):
        nav = tk.Frame(self.right_panel, pady=5)
        nav.pack(fill="x")
        tk.Button(nav, text="< Prev", command=self.prev_img).pack(side="left", expand=True, fill="x")
        tk.Button(nav, text="Reset View", command=self.reset_view).pack(side="left", expand=True, fill="x")
        tk.Button(nav, text="Next >", command=self.next_img).pack(side="left", expand=True, fill="x")
        
        wf = tk.LabelFrame(self.right_panel, text="Marking Workflow", font=("Arial", 11, "bold"), fg="#C0392B", padx=5, pady=5)
        wf.pack(fill="x", pady=10)
        
        self.lbl_instruction = tk.Label(wf, text="Step 1: Click 'Start Sequence'", fg="blue", wraplength=300)
        self.lbl_instruction.pack(pady=5)
        
        self.btn_action = tk.Button(wf, text="Start Sequence", command=self.cycle_state, 
                                    bg="#27AE60", fg="white", font=("Arial", 10, "bold"))
        self.btn_action.pack(fill="x", pady=5)
        
        self.lbl_calc_info = tk.Label(wf, text="", fg="#1F618D", font=("Consolas", 10, "bold"), justify="left")
        self.lbl_calc_info.pack(pady=5, fill="x")
        
        tk.Label(self.right_panel, text="[Wheel] Zoom | [Right-Click] Pan", fg="gray", bg="#ECF0F1").pack()

    def setup_file_list(self):
        f = tk.LabelFrame(self.right_panel, text="Image Files", padx=5, pady=5)
        f.pack(fill="both", expand=True, pady=5)
        sb = tk.Scrollbar(f)
        sb.pack(side="right", fill="y")
        self.lst_files = tk.Listbox(f, font=("Consolas", 9), selectmode="browse", yscrollcommand=sb.set, 
                                    activestyle="dotbox", selectbackground="#3498DB", selectforeground="white")
        self.lst_files.pack(side="left", fill="both", expand=True)
        sb.config(command=self.lst_files.yview)
        self.lst_files.bind("<<ListboxSelect>>", self.on_file_select)
        
        self.refresh_file_list()

    def refresh_file_list(self):
        self.lst_files.delete(0, tk.END)
        for i, p in enumerate(self.image_list):
            fname = os.path.basename(p)
            if fname in self.marked_files:
                display_text = f"✅ {fname}"
                self.lst_files.insert(tk.END, display_text)
                self.lst_files.itemconfig(i, {'fg': 'green'}) 
            else:
                display_text = f"☐ {fname}"
                self.lst_files.insert(tk.END, display_text)
        
        if 0 <= self.current_idx < self.lst_files.size():
            self.lst_files.selection_clear(0, tk.END)
            self.lst_files.selection_set(self.current_idx)
            self.lst_files.see(self.current_idx)

    def setup_data_list(self):
        f = tk.LabelFrame(self.right_panel, text="Recorded Data", padx=5, pady=5)
        f.pack(fill="both", expand=True)
        self.lst_data = tk.Listbox(f, font=("Consolas", 8), height=8)
        self.lst_data.pack(fill="both", expand=True)
        tk.Button(f, text="💾 Save CSV", command=self.save_csv, bg="#2980B9", fg="white").pack(fill="x", pady=5)

    # --- VIEW CONTROL ---
    def reset_view(self):
        self.zoom_level = 1.0; self.pan_x = 0; self.pan_y = 0; self.refresh_image()
    def on_mousewheel(self, event):
        if event.num == 5 or event.delta < 0: self.zoom_out()
        else: self.zoom_in()
    def zoom_in(self, event=None): self.zoom_level *= 1.2; self.refresh_image()
    def zoom_out(self, event=None): 
        self.zoom_level /= 1.2; 
        if self.zoom_level < 0.1: self.zoom_level = 0.1
        self.refresh_image()
    def start_pan(self, event): self.drag_start_x = event.x; self.drag_start_y = event.y
    def do_pan(self, event):
        self.pan_x += event.x - self.drag_start_x; self.pan_y += event.y - self.drag_start_y
        self.drag_start_x = event.x; self.drag_start_y = event.y; self.refresh_image()

    # --- NAV & IMAGE LOAD ---
    def on_file_select(self, event):
        sel = self.lst_files.curselection(); 
        if sel: self.current_idx = sel[0]; self.refresh_image()
    def prev_img(self, e=None):
        if self.current_idx > 0: self.current_idx -= 1; self.refresh_image()
    def next_img(self, e=None):
        if self.current_idx < len(self.image_list) - 1: self.current_idx += 1; self.refresh_image()

    def refresh_image(self):
        if not self.image_list: return
        
        # Only select if index is valid (prevents errors during rapid changes)
        if self.lst_files.size() > 0:
            self.lst_files.selection_clear(0, tk.END)
            if 0 <= self.current_idx < self.lst_files.size():
                self.lst_files.selection_set(self.current_idx)
                self.lst_files.see(self.current_idx)

        path = self.image_list[self.current_idx]
        img = cv2.imread(path) 
        if img is None: return
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        self.current_img_w = w 
        
        cw = self.canvas.winfo_width(); ch = self.canvas.winfo_height()
        if cw < 50: cw, ch = 800, 600
        base_scale = min(cw/w, ch/h)
        self.current_scale = base_scale * self.zoom_level
        nw = int(w * self.current_scale); nh = int(h * self.current_scale)
        self.offset_x = (cw - nw) // 2 + self.pan_x; self.offset_y = (ch - nh) // 2 + self.pan_y
        
        pil = Image.fromarray(img).resize((nw, nh), Image.Resampling.NEAREST)
        self.tk_img = ImageTk.PhotoImage(pil)
        self.canvas.delete("all")
        self.canvas.create_image(self.offset_x, self.offset_y, anchor="nw", image=self.tk_img)
        
        # DIRECTION LABEL
        d_str = self.scan_direction.get()
        col_d = "cyan" if d_str == "L->R" else "orange"
        self.canvas.create_text(10, 30, anchor="nw", text=f"Scan: {d_str}", fill=col_d, font=("Arial", 10, "bold"))
        self.canvas.create_text(10, 10, anchor="nw", text=f"Frame: {self.current_idx} ({self.zoom_level:.1f}x)", fill="white", font=("Arial", 12))
        
        # Draw Overlays
        show_lines = (self.marking_state < 4)
        if show_lines:
            for key, col in [('t1', 'cyan'), ('t2', 'magenta')]:
                if key in self.temp_points:
                    fid, x, y = self.temp_points[key]
                    cx = self.offset_x + x * self.current_scale; cy = self.offset_y + y * self.current_scale
                    if not (key == 't1' and self.marking_state == 3): self.canvas.create_line(self.offset_x, cy, cx, cy, fill=col, width=2)
                    self.canvas.create_oval(cx-5, cy-5, cx+5, cy+5, outline=col, width=2)
                    label = f"{key.upper()}" + (f" (Fr:{fid})" if fid != self.current_idx else "")
                    self.canvas.create_text(cx+10, cy-10, text=label, fill=col, font=("Arial", 10, "bold"), anchor="w")
        
        if 'dia_end' in self.temp_points:
            x1, y1 = self.temp_points['dia_start'][1:]; x2, y2 = self.temp_points['dia_end'][1:]
            cx1 = self.offset_x + x1 * self.current_scale; cy1 = self.offset_y + y1 * self.current_scale
            cx2 = self.offset_x + x2 * self.current_scale; cy2 = self.offset_y + y2 * self.current_scale
            if self.temp_points['dia_start'][0] == self.current_idx:
                self.canvas.create_line(cx1, cy1, cx2, cy2, fill="yellow", width=2)
                self.canvas.create_text((cx1+cx2)/2, (cy1+cy2)/2 - 15, text=f"D={self.measured_dia_um:.1f}µm", fill="yellow", font=("Arial", 10, "bold"))
                
        elif self.dia_start_pt and self.marking_state == 5 and self.dia_start_pt[0] == self.current_idx:
            cx = self.offset_x + self.dia_start_pt[1] * self.current_scale; cy = self.offset_y + self.dia_start_pt[2] * self.current_scale
            self.canvas.create_oval(cx-4, cy-4, cx+4, cy+4, fill="yellow")

    def cycle_state(self):
        if self.marking_state == 0:
            self.temp_points = {}; self.dia_start_pt = None; self.marking_state = 2
            edge = "Left" if self.scan_direction.get() == "L->R" else "Right"
            self.lbl_instruction.config(text=f"1. Ref: {edge}.\n2. Find t1 Frame -> Click Pore.")
            self.btn_action.config(text="Cancel", bg="#95A5A6"); self.lbl_calc_info.config(text=f"Waiting for T1...")
            self.refresh_image()
        elif self.marking_state >= 2: self.reset_state()

    def get_distance_from_ref(self, x_pos):
        return (self.current_img_w - x_pos) if self.scan_direction.get() == "R->L" else x_pos

    def on_canvas_click(self, event):
        self.canvas.focus_set()
        if self.marking_state == 0: return

        x_click = (event.x - self.offset_x) / self.current_scale
        y_click = (event.y - self.offset_y) / self.current_scale
        
        if self.marking_state == 2: # T1
            self.temp_points['t1'] = (self.current_idx, x_click, y_click)
            try: scale = float(self.ent_scale.get()); vel_mm_s = float(self.ent_velocity.get())
            except: scale = 2.9216; vel_mm_s = 800.0
            self.fixed_velocity_mm_s = vel_mm_s
            
            vel_um_us = vel_mm_s / 1000.0 
            if vel_um_us == 0: vel_um_us = 0.001

            self.L1_um = self.get_distance_from_ref(x_click) * scale
            self.t1_calc_us = self.L1_um / vel_um_us # Time = Dist / Speed
            
            self.marking_state = 3
            self.lbl_instruction.config(text="3. Go to t2 frame -> Click Pore.")
            self.lbl_calc_info.config(text=f"t1: {self.t1_calc_us:.0f} µs")
            self.refresh_image()
            
        elif self.marking_state == 3: # T2
            self.temp_points['t2'] = (self.current_idx, x_click, y_click)
            try: scale = float(self.ent_scale.get())
            except: scale = 2.9216
            
            self.L2_um = self.get_distance_from_ref(x_click) * scale
            vel_um_us = self.fixed_velocity_mm_s / 1000.0
            if vel_um_us == 0: vel_um_us = 0.001
            self.t2_calc_us = self.L2_um / vel_um_us
            
            self.lbl_calc_info.config(text=f"t1: {self.t1_calc_us:.0f} µs | t2: {self.t2_calc_us:.0f} µs")
            self.marking_state = 4
            self.lbl_instruction.config(text="4. Measure Diameter: Start Point")
            self.refresh_image() 
                
        elif self.marking_state == 4: # Dia Start
            self.dia_start_pt = (self.current_idx, x_click, y_click); self.marking_state = 5
            self.lbl_instruction.config(text="5. Measure Diameter: End Point"); self.refresh_image()

        elif self.marking_state == 5: # Dia End
            try: scale = float(self.ent_scale.get())
            except: scale = 2.9216
            x1, y1 = self.dia_start_pt[1], self.dia_start_pt[2]
            dist_px = math.sqrt((x_click - x1)**2 + (y_click - y1)**2)
            self.measured_dia_um = dist_px * scale
            self.temp_points['dia_start'] = self.dia_start_pt; self.temp_points['dia_end'] = (self.current_idx, x_click, y_click)
            self.finalize_sequence()

    def finalize_sequence(self):
        fname = os.path.basename(self.image_list[self.temp_points['t1'][0]])
        self.marked_files.add(fname); self.refresh_file_list()
        
        entry = {
            "Filename": fname, "Velocity_mm_s": self.fixed_velocity_mm_s,
            "t1_calc_us": round(self.t1_calc_us, 2), "t2_calc_us": round(self.t2_calc_us, 2),
            "Diameter_um": round(self.measured_dia_um, 2), "Direction": self.scan_direction.get()
        }
        self.collected_data.append(entry)
        self.lst_data.insert(tk.END, f"{len(self.collected_data)}. t1={self.t1_calc_us:.0f} | t2={self.t2_calc_us:.0f} | D={self.measured_dia_um:.1f}")
        self.temp_points = {}; self.dia_start_pt = None; self.marking_state = 0
        self.btn_action.config(text="Start New Sequence", bg="#27AE60"); self.lbl_instruction.config(text="Saved. Ready.")
        self.refresh_image()

    def reset_state(self):
        self.marking_state = 0; self.temp_points = {}; self.dia_start_pt = None
        self.lbl_instruction.config(text="Step 1: Click 'Start Sequence'")
        self.btn_action.config(text="Start Sequence", bg="#27AE60"); self.lbl_calc_info.config(text="Idle")
        self.refresh_image()

    def save_csv(self):
        if not self.collected_data: return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if path:
            with open(path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=["Filename", "Velocity_mm_s", "t1_calc_us", "t2_calc_us", "Diameter_um", "Direction"])
                writer.writeheader(); writer.writerows(self.collected_data)
            messagebox.showinfo("Success", "Data saved.")

    def open_plotter(self):
        plot_win = tk.Toplevel(self.window); plot_win.title("t1 vs t2 Analysis"); plot_win.geometry("900x700")
        ctrl = tk.Frame(plot_win); ctrl.pack(side="top", fill="x", padx=10, pady=5)
        
        tk.Button(ctrl, text="📂 Load CSV", command=lambda: self.load_csv_plot(ax, canvas, ent), bg="#2980B9", fg="white").pack(side="left")
        tk.Label(ctrl, text="| Split (µm):", fg="#E74C3C").pack(side="left", padx=10)
        ent = tk.Entry(ctrl, width=5); ent.insert(0, "100"); ent.pack(side="left")
        tk.Button(ctrl, text="Refresh", command=lambda: self.plot_data(ax, canvas, self.collected_data, ent), bg="#E67E22", fg="white").pack(side="left", padx=10)

        fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
        canvas = FigureCanvasTkAgg(fig, master=plot_win); canvas.get_tk_widget().pack(fill="both", expand=True)
        if self.collected_data: self.plot_data(ax, canvas, self.collected_data, ent)

    def load_csv_plot(self, ax, canvas, ent):
        p = filedialog.askopenfilename(filetypes=[("CSV", "*.csv")]); 
        if not p: return
        try:
            with open(p, 'r') as f: data = list(csv.DictReader(f))
            # Convert strings to floats
            for d in data: 
                d['t1_calc_us'] = float(d['t1_calc_us']); d['t2_calc_us'] = float(d['t2_calc_us']); d['Diameter_um'] = float(d['Diameter_um'])
            self.plot_data(ax, canvas, data, ent)
        except Exception as e: messagebox.showerror("Error", str(e))

    def plot_data(self, ax, canvas, data, ent):
        ax.clear(); 
        try: cut = float(ent.get())
        except: cut = 100.0
        
        x_s, y_s, x_l, y_l = [], [], [], []
        for d in data:
            (x_s if d['Diameter_um'] < cut else x_l).append(d['t1_calc_us'])
            (y_s if d['Diameter_um'] < cut else y_l).append(d['t2_calc_us'])

        if x_s: ax.scatter(x_s, y_s, c='red', alpha=0.6, label=f"Dia < {cut}µm")
        if x_l: ax.scatter(x_l, y_l, c='blue', alpha=0.6, label=f"Dia >= {cut}µm")
        
        all_vals = x_s + x_l + y_s + y_l
        if all_vals:
            mn, mx = min(all_vals), max(all_vals); pad = (mx-mn)*0.1
            ax.plot([mn-pad, mx+pad], [mn-pad, mx+pad], 'k--', alpha=0.3)
            ax.set_xlim(mn-pad, mx+pad); ax.set_ylim(mn-pad, mx+pad)
            
        ax.set_xlabel("t1 (calc) [µs]"); ax.set_ylabel("t2 (calc) [µs]"); ax.legend(); ax.grid(True)
        canvas.draw()