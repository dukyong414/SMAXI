import sys
import os
import webbrowser
import importlib.util
from pathlib import Path

_PACKAGE_DIR = Path(__file__).resolve().parent
sys.path.append(str(_PACKAGE_DIR))


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


def _import_normalize():
    _load_local_module("normalize", "2_normalize.py")
    from normalize import NormalizerApp, ThermalCorrectionApp
    return NormalizerApp, ThermalCorrectionApp


def _import_ai_tools():
    _load_local_module("ai_tools", "3_ai_tools.py")
    from ai_tools import SamLabelerApp, YoloTrainerApp
    return SamLabelerApp, YoloTrainerApp


def _import_analysis():
    _load_local_module("analysis", "4_analysis.py")
    from analysis import DataPlotterApp, EventTaggerApp
    return DataPlotterApp, EventTaggerApp

# =========================================================
# HELPER: SAFE IMAGE IO (trouble shooting for computer using non-English character)
# =========================================================
def imread_safe(path):
    """Reads an image safely, handling non-English paths."""
    try:
        stream = np.fromfile(path, np.uint8)
        return cv2.imdecode(stream, cv2.IMREAD_UNCHANGED)
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return None

def imwrite_safe(path, img):
    """Writes an image safely, handling non-English paths."""
    try:
        ext = os.path.splitext(path)[1]
        result, n = cv2.imencode(ext, img)
        if result:
            with open(path, mode='wb') as f:
                n.tofile(f)
            return True
        return False
    except Exception as e:
        print(f"Error writing {path}: {e}")
        return False

# =========================================================
# CONFIG APP
# =========================================================
class ConfigApp:
    def __init__(self, root, on_complete):
        self.root = root
        self.on_complete = on_complete
        self.root.title("System Configuration")
        self.root.geometry("900x250")
        self.root.configure(bg="#ECF0F1")       
        
        cx = self.root.winfo_screenwidth() // 2
        cy = self.root.winfo_screenheight() // 2
        self.root.geometry(f"900x250+{cx-300}+{cy-125}")

        tk.Label(self.root, text="Select master project folder", font=("Arial", 16, "bold"), bg="#ECF0F1", fg="#2C3E50").pack(pady=20)
        
        f_entry = tk.Frame(self.root, bg="#ECF0F1")
        f_entry.pack(fill="x", padx=20)
        
        # DEFAULT PATH & RECALL LOGIC
        default_path = r"C:\Users\X-ray image processing software"
        if os.path.exists("saved_path.txt"):
            with open("saved_path.txt", "r") as f:
                default_path = f.read().strip()
                
        self.path_var = tk.StringVar(value=default_path)
        self.remember_var = tk.BooleanVar(value=True) # Checkbox variable
        
        tk.Entry(f_entry, textvariable=self.path_var, font=("Consolas", 10), width=50).pack(side="left", fill="x", expand=True)
        tk.Button(f_entry, text="Browse...", command=self.browse, bg="#BDC3C7").pack(side="left", padx=5)
        tk.Checkbutton(self.root, text="Remember this folder for next time", variable=self.remember_var, bg="#ECF0F1", fg="#2C3E50", font=("Arial", 10)).pack(pady=5)
        tk.Button(self.root, text="Start SMAXI", command=self.confirm, bg="#2ECC71", fg="white", font=("Arial", 12, "bold"), width=20, height=2).pack(pady=30)
        
    def browse(self):
        d = filedialog.askdirectory()
        if d: self.path_var.set(d)
        
    def confirm(self):
        p = self.path_var.get()
        if not os.path.exists(p):
            messagebox.showerror("Error", "Folder does not exist!\nPlease select the valid root folder.")
            return
        if self.remember_var.get():
            with open("saved_path.txt", "w") as f:
                f.write(p)
        # UPDATE GLOBALS IN UTILS
        utils.BASE_DIR = p
        utils.LOGO_FILE_PATH = os.path.join(utils.BASE_DIR, "2. [Image file] Logo image for GUI setup", "Logo.png")
        utils.SAM_CHECKPOINT_DEFAULT = os.path.join(utils.BASE_DIR, r"1. Root folder\models", "sam_vit_b_01ec64.pth")
        utils.BG_IMAGE_PATH = os.path.join(utils.BASE_DIR, "2. [Image file] Background image for GUI setup", "Background image.png")
        
        for widget in self.root.winfo_children():
            widget.destroy()
            
        self.on_complete()

# =========================================================
# SPLASH SCREEN
# =========================================================
class SplashScreen:
    def __init__(self, root, switch_callback):
        self.root = root
        self.switch_callback = switch_callback
        self.root.title("FAST-AM Lab | SMAXI")
        
        w = self.root.winfo_screenwidth()
        h = self.root.winfo_screenheight()
        
        self.root.geometry(f"{w}x{h}+0+0")
        self.root.update_idletasks()
        try: self.root.state('zoomed')
        except: self.root.attributes('-fullscreen', True)
        
        self.canvas = tk.Canvas(self.root, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        self.final_bg = None

        # Use utils.BG_IMAGE_PATH
        if os.path.exists(utils.BG_IMAGE_PATH):
            try:
                pil_img = Image.open(utils.BG_IMAGE_PATH).convert("RGBA")
                pil_img = resize_aspect_fill(pil_img, w, h)
                overlay = Image.new('RGBA', pil_img.size, (255, 255, 255, 0))
                draw = ImageDraw.Draw(overlay)

                box_w, box_h = 900, 800
                left = (w - box_w) // 2
                top = (h - box_h) // 2
                right = left + box_w
                bottom = top + box_h
                
                draw.rectangle((left, top, right, bottom), fill=(255, 255, 255, 235), outline="#BDC3C7", width=3)
                combined_img = Image.alpha_composite(pil_img, overlay)
                self.final_bg = ImageTk.PhotoImage(combined_img)
                self.canvas.create_image(0, 0, image=self.final_bg, anchor="nw")
            except Exception as e:
                print(f"Error processing background: {e}")
                self.canvas.config(bg="white")
        else:
            self.canvas.config(bg="white")

        cx = w // 2
        cy_start = (h - 600) // 2 + 60 

        if os.path.exists(utils.LOGO_FILE_PATH):
            try:
                self.logo_img = tk.PhotoImage(file=utils.LOGO_FILE_PATH)
                self.canvas.create_image(cx, cy_start, image=self.logo_img, anchor="center")
            except: pass
        
        self.canvas.create_text(cx, cy_start + 200, text="ML-powered x-ray image analysis software", 
                                font=("Helvetica", 20, "bold"), fill="#2C3E50", width=650, justify="center")

        github_url = "https://github.com/dukyong414/SMAXI"
        self.canvas.create_text(
            cx,
            cy_start + 245,
            text="View on GitHub: dukyong414/SMAXI",
            font=("Arial", 11, "underline"),
            fill="#2E86C1",
            tags="github_link",
        )
        self.canvas.tag_bind("github_link", "<Button-1>", lambda _e: webbrowser.open(github_url))
        self.canvas.tag_bind("github_link", "<Enter>", lambda _e: self.canvas.config(cursor="hand2"))
        self.canvas.tag_bind("github_link", "<Leave>", lambda _e: self.canvas.config(cursor=""))
        
        self.canvas.create_text(cx, cy_start + 285, text="Developed by FAST-AM Lab, Northwestern University", 
                                font=("Helvetica", 15, "bold"), fill="#4E058E")

        desc_text = (
            "Welcome to SMAXI, an ML-powered full-field x-ray image analysis software.\n\n" 
            "An end-to-end workflow for pre-processing, segmenting, tracking, and analyzing"
            " multi-dimensional full-field x-ray data (2D images and 3D CT data).\n\n"
            "1. Image pre-processor: Thermal drift correction and background removal.\n"
            "2. ML-powered object segmenter: SAM-powered 2D and 3D segmentation.\n"
            "3. ML-powered object tracker: YOLO-powered continuous tracking.\n"
            "4. Interactive geometry feature analyzer: LLM-powered feature analysis tool.\n"
        )
        self.canvas.create_text(cx, cy_start + 420, text=desc_text, font=("Arial", 14), fill="#34495E", 
                                justify="left", width=850)

        btn_start = tk.Button(self.root, text="Click to start SMAXI", font=("Arial", 14, "bold"), 
                              bg="#4E058E", fg="white", width=25, height=2, 
                              command=self.start_app, relief="raised", cursor="hand2")
        
        self.canvas.create_window(cx, cy_start + 560, window=btn_start, anchor="center")
        self.canvas.create_text(cx, cy_start + 610, text="© 2026 Northwestern University", font=("Arial", 10), fill="#2C3E50")

    def start_app(self):
        self.canvas.destroy()
        self.switch_callback()  

# =========================================================
# CLASS: APP LAUNCHER (Main Menu)
# =========================================================
class AppLauncher:
    def __init__(self, root):
        self.root = root
        self.root.title("FAST-AM Lab | SMAXI") 
        self.root.geometry("1200x900") 
        self.root.update_idletasks()
        try: self.root.state('zoomed') 
        except: self.root.attributes('-fullscreen', True) 
        
        self.main_container = tk.Frame(root)
        self.main_container.pack(fill="both", expand=True)
        self.show_home()

    def go_to_splash(self):
        self.main_container.destroy()
        def restart_program():
            AppLauncher(self.root)
        SplashScreen(self.root, restart_program)

    def _refresh_hardware_label(self):
        if not hasattr(self, "lbl_hardware"):
            return
        try:
            if not self.lbl_hardware.winfo_exists():
                return
        except tk.TclError:
            return
        utils.init_gpu_normalizer()
        status_color = "black" if utils.HAS_GPU_NORMALIZER else "gray"
        self.lbl_hardware.config(
            text=f"Hardware: {utils.GPU_NAME}",
            fg=status_color,
        )

    def show_home(self):
        for w in self.main_container.winfo_children(): w.destroy()
        
        self.canvas = tk.Canvas(self.main_container, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.final_bg = None
        
        w = self.root.winfo_width()
        h = self.root.winfo_height()
        if w < 100: w, h = 1200, 900

        if os.path.exists(utils.BG_IMAGE_PATH):
            try:
                pil_img = Image.open(utils.BG_IMAGE_PATH).convert("RGBA")
                pil_img = resize_aspect_fill(pil_img, w, h)
                overlay = Image.new('RGBA', pil_img.size, (255, 255, 255, 0))
                draw = ImageDraw.Draw(overlay)
                box_w, box_h = 1000, 900
                left = (w - box_w) // 2
                top = (h - box_h) // 2
                draw.rectangle((left, top, left + box_w, top + box_h), fill=(255, 255, 255, 235), outline="#BDC3C7", width=3)
                combined_img = Image.alpha_composite(pil_img, overlay)
                self.final_bg = ImageTk.PhotoImage(combined_img)
                self.canvas.create_image(0, 0, image=self.final_bg, anchor="nw")
            except: self.canvas.config(bg="white")
        else:
            self.canvas.config(bg="white")

        cx = w // 2
        cy_start = (h - 700) // 2 + 50 

        btn_back = tk.Button(self.canvas, text="← Back to Welcome", command=self.go_to_splash, 
                             font=("Arial", 11, "bold"), bg="#95A5A6", fg="white")
        self.canvas.create_window(30, 30, window=btn_back, anchor="nw")
        
        self.lbl_hardware = tk.Label(
            self.canvas,
            text="Hardware: detecting…",
            fg="gray",
            bg="white",
            font=("Arial", 10, "bold"),
            padx=10,
            pady=5,
            relief="ridge",
            borderwidth=2,
        )
        self.canvas.create_window(w - 30, 30, window=self.lbl_hardware, anchor="ne")
        self.root.after_idle(self._refresh_hardware_label)

        if os.path.exists(utils.LOGO_FILE_PATH):
            try:
                self.logo_img = tk.PhotoImage(file=utils.LOGO_FILE_PATH)
                self.canvas.create_image(cx, cy_start, image=self.logo_img, anchor="center")
            except: pass
        else:
            self.canvas.create_text(cx, cy_start, text="FAST-AM Lab", font=("Arial", 24, "bold"), fill="gray")

        self.canvas.create_text(cx, cy_start + 150, text="Select Tool", font=("Helvetica", 22, "bold"), fill="#2C3E50")
        
        note_text = "Note: The processing speed of each tool depends on your PC's specifications."
        self.canvas.create_text(cx, cy_start + 200, text=note_text, font=("Arial", 15, "italic"), fill="black")
        
        specs_text = f"Your System: {get_system_specs()}"
        # Added width=600 to force the text to wrap into two lines when it gets too long
        self.canvas.create_text(cx, cy_start + 250, text=specs_text, font=("Consolas", 15, "bold"), fill="#2E86C1", width=600, justify="center")

        btn_frame = tk.Frame(self.canvas, bg="white")
        btn_style = {
            "font": ("Arial", 13, "bold"),
            "fg": "white",
            "width": 22,
            "height": 4,
            "wraplength": 200,
            "cursor": "hand2",
            "relief": "flat",
            "borderwidth": 0,
            "activeforeground": "white",
        }
        section_box_style = {
            "bg": "#F7F9FA",
            "relief": "solid",
            "borderwidth": 2,
            "highlightthickness": 0,
        }
        section_label_style = {
            "font": ("Arial", 13, "bold"),
            "fg": "#2C3E50",
            "justify": "center",
            "padx": 18,
            "pady": 22,
        }

        def _build_tool_section(parent, row, title, label_bg, buttons):
            section = tk.Frame(parent, **section_box_style)
            section.grid(row=row, column=0, padx=8, pady=(0, 14 if row == 0 else 0), sticky="w")

            tk.Label(
                section,
                text=title,
                bg=label_bg,
                **section_label_style,
            ).grid(row=0, column=0, sticky="ns")

            tk.Frame(section, width=1, bg="#C5CDD5").grid(row=0, column=1, sticky="ns", pady=10)

            button_area = tk.Frame(section, bg=section_box_style["bg"])
            button_area.grid(row=0, column=2, padx=(14, 16), pady=14, sticky="w")

            for idx, (text, color, command, active_bg) in enumerate(buttons):
                btn_row, btn_col = divmod(idx, 2)
                colspan = 2 if len(buttons) == 1 else 1
                tk.Button(
                    button_area,
                    text=text,
                    bg=color,
                    activebackground=active_bg,
                    command=command,
                    **btn_style,
                ).grid(row=btn_row, column=btn_col, columnspan=colspan, padx=10, pady=8, sticky="ew")

        _build_tool_section(
            btn_frame,
            row=0,
            title="Major\nfeatures",
            label_bg="#E8EEF2",
            buttons=[
                ("1. Image pre-processor", "#16A085", self.launch_normalizer, "#138D75"),
                ("2. ML-powered \nobject segmenter", "#2980B9", self.launch_labeler, "#21618C"),
                ("3. ML-powerd \nobject tracker", "#8E44AD", self.launch_trainer, "#7D3C98"),
                ("4. Interactive geometry\nfeature analyzer", "#E67E22", self.launch_plotter, "#CA6F1E"),
            ],
        )
        _build_tool_section(
            btn_frame,
            row=1,
            title="Auxiliary\nfeatures",
            label_bg="#ECEFF1",
            buttons=[
                ("5. Event Tagger", "#34495E", self.launch_tagger, "#2C3E50"),
                ("6. Add your own\nPython tool", "#7F8C8D", self.show_custom_tool_guide, "#5D6D7E"),
            ],
        )

        self.canvas.create_window(cx, cy_start + 500, window=btn_frame, anchor="center")
        self.canvas.create_text(cx, cy_start + 600, text="© 2026 Northwestern University", font=("Arial", 10), fill="#BDC3C7")

    def show_others_menu(self):
        for w in self.main_container.winfo_children(): w.destroy()
        
        self.canvas = tk.Canvas(self.main_container, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        
        w = self.root.winfo_width()
        h = self.root.winfo_height()
        if w < 100: w, h = 1200, 900
        
        if os.path.exists(utils.BG_IMAGE_PATH):
            try:
                pil_img = Image.open(utils.BG_IMAGE_PATH).convert("RGBA")
                pil_img = resize_aspect_fill(pil_img, w, h)
                overlay = Image.new('RGBA', pil_img.size, (255, 255, 255, 0))
                draw = ImageDraw.Draw(overlay)
                box_w, box_h = 800, 600
                left = (w - box_w) // 2
                top = (h - box_h) // 2
                draw.rectangle((left, top, left + box_w, top + box_h), fill=(255, 255, 255, 235), outline="#BDC3C7", width=3)
                combined_img = Image.alpha_composite(pil_img, overlay)
                self.final_bg = ImageTk.PhotoImage(combined_img)
                self.canvas.create_image(0, 0, image=self.final_bg, anchor="nw")
            except: self.canvas.config(bg="white")
        else:
            self.canvas.config(bg="white")

        btn_back = tk.Button(self.canvas, text="← Back to Menu", command=self.show_home, 
                             font=("Arial", 11, "bold"), bg="#95A5A6", fg="white")
        self.canvas.create_window(30, 30, window=btn_back, anchor="nw")
        
        cx = w // 2
        cy_start = (h - 600) // 2 + 50 
        
        self.canvas.create_text(cx, cy_start + 100, text="Other Tools", font=("Helvetica", 28, "bold"), fill="#2C3E50")
        self.canvas.create_text(cx, cy_start + 140, text="Select a supplementary module:", font=("Arial", 12, "italic"), fill="#34495E")
        
        btn_frame = tk.Frame(self.canvas, bg="white")
        btn_style = {"font": ("Arial", 14, "bold"), "fg": "white", "width": 30, "height": 4, "wraplength": 280, "cursor": "hand2"}
        
        tk.Button(btn_frame, text="1. Transient Event Analyzer\n(Includes Advanced Functions)", bg="#C0392B", command=self.launch_tagger, **btn_style).pack(pady=15)
        
        self.canvas.create_window(cx, cy_start + 300, window=btn_frame, anchor="center")    

    def launch_normalizer(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("Pre-Processing Check")
        dialog.geometry("550x250")
        dialog.resizable(False, False)
        dialog.configure(bg="white")
        
        try:
            x = self.root.winfo_x() + (self.root.winfo_width() // 2) - 275
            y = self.root.winfo_y() + (self.root.winfo_height() // 2) - 125
            dialog.geometry(f"+{x}+{y}")
        except:
            dialog.geometry("+100+100")

        dialog.transient(self.root)
        dialog.grab_set()

        lbl_icon = tk.Label(dialog, text="?", font=("Arial", 40, "bold"), fg="#3498DB", bg="white")
        lbl_icon.place(x=20, y=20)

        msg = ("Do you want to check cine/image files for \n"
               "thermal drift correction first?")
        lbl_msg = tk.Label(dialog, text=msg, font=("Arial", 12), bg="white", justify="left")
        lbl_msg.place(x=80, y=30)

        self.user_choice = None 

        def on_drift():
            self.user_choice = True
            dialog.destroy()

        def on_norm():
            self.user_choice = False
            dialog.destroy()

        def on_cancel():
            self.user_choice = None
            dialog.destroy()

        btn_drift = tk.Button(dialog, text="Go to Thermal Drift Tool", command=on_drift, 
                              bg="#2ECC71", fg="white", font=("Arial", 10, "bold"), width=25, height=2)
        btn_drift.place(x=50, y=100)

        btn_norm = tk.Button(dialog, text="Go directly to Normalizer", command=on_norm, 
                             bg="#E67E22", fg="white", font=("Arial", 10, "bold"), width=25, height=2)
        btn_norm.place(x=280, y=100)

        btn_cancel = tk.Button(dialog, text="Cancel", command=on_cancel, 
                               bg="#95A5A6", fg="white", font=("Arial", 10), width=10)
        btn_cancel.place(x=230, y=180)

        dialog.protocol("WM_DELETE_WINDOW", on_cancel)
        self.root.wait_window(dialog)

        if self.user_choice is None: 
            return 
        
        for w in self.main_container.winfo_children(): w.destroy()

        NormalizerApp, ThermalCorrectionApp = _import_normalize()
        if self.user_choice is True:
            ThermalCorrectionApp(
                self.main_container,
                on_back=self.show_home,
                on_proceed_to_norm=self.launch_normalizer_direct,
            )
        else:
            NormalizerApp(self.main_container, on_back=self.show_home)

    def launch_normalizer_direct(self):
        for w in self.main_container.winfo_children(): w.destroy()
        NormalizerApp, _ = _import_normalize()
        NormalizerApp(self.main_container, on_back=self.show_home)

    def launch_trainer(self):
        for w in self.main_container.winfo_children(): w.destroy()
        _, YoloTrainerApp = _import_ai_tools()
        YoloTrainerApp(self.main_container, on_back=self.show_home)

    def launch_plotter(self):
        for w in self.main_container.winfo_children(): w.destroy()
        DataPlotterApp, _ = _import_analysis()
        DataPlotterApp(self.main_container, on_back=self.show_home)

    def launch_tagger(self):
        for w in self.main_container.winfo_children(): w.destroy()
        _, EventTaggerApp = _import_analysis()
        EventTaggerApp(self.main_container, on_back=self.show_home)

    def launch_labeler(self):
        for w in self.main_container.winfo_children(): w.destroy()
        SamLabelerApp, _ = _import_ai_tools()
        SamLabelerApp(self.main_container, on_back=self.show_home)

    def show_custom_tool_guide(self):
        """Public guide: how to wire a stand-alone Python script into SMAXI."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Add your own Python tool")
        dialog.geometry("720x560")
        dialog.configure(bg="white")
        dialog.transient(self.root)
        try:
            x = self.root.winfo_x() + (self.root.winfo_width() // 2) - 360
            y = self.root.winfo_y() + (self.root.winfo_height() // 2) - 280
            dialog.geometry(f"+{x}+{y}")
        except Exception:
            pass

        tk.Label(
            dialog,
            text="Integrate a stand-alone Python app into SMAXI",
            font=("Arial", 14, "bold"),
            bg="white",
            fg="#2C3E50",
        ).pack(anchor="w", padx=18, pady=(16, 8))

        guide = (
            "You can launch your own .py tool from the Auxiliary features row "
            "without rewriting it as an embedded SMAXI panel.\n\n"
            "Recommended steps\n"
            "─────────────────\n"
            "1. Put your script somewhere stable, e.g.\n"
            "   1. Root folder / 1. Python code / my_custom_tool.py\n"
            "   or keep it outside the package and use an absolute path.\n\n"
            "2. Open 1_main.py and add a path constant near the top:\n\n"
            "   MY_TOOL_SCRIPT = Path(__file__).resolve().parent / \"my_custom_tool.py\"\n"
            "   # or: MY_TOOL_SCRIPT = Path(r\"C:\\\\path\\\\to\\\\my_tool.py\")\n\n"
            "3. Add a launcher method on AppLauncher (same pattern as Event Tagger):\n\n"
            "   def launch_my_tool(self):\n"
            "       import subprocess, sys\n"
            "       script = MY_TOOL_SCRIPT\n"
            "       if not script.is_file():\n"
            "           messagebox.showerror(\"Missing\", f\"Not found:\\n{script}\")\n"
            "           return\n"
            "       subprocess.Popen([sys.executable, str(script)], cwd=str(script.parent))\n\n"
            "4. Register a button in show_home() under Auxiliary features:\n\n"
            "   (\"6. My custom tool\", \"#7F8C8D\", self.launch_my_tool, \"#5D6D7E\")\n\n"
            "Notes\n"
            "─────\n"
            "• Stand-alone tk.Tk() apps should be started with subprocess.Popen so they "
            "get their own process/window (do not nest a second tk.Tk inside SMAXI).\n"
            "• If your tool is a class that takes (parent_frame, on_back=...), you can "
            "embed it like NormalizerApp instead of using subprocess.\n"
            "• Keep secrets and machine-specific paths out of the public repo; use "
            "relative paths under the SMAXI package when possible.\n"
        )

        text = scrolledtext.ScrolledText(
            dialog, wrap="word", font=("Consolas", 10), bg="#F8F9F9", height=22
        )
        text.pack(fill="both", expand=True, padx=18, pady=(0, 10))
        text.insert("1.0", guide)
        text.config(state="disabled")

        tk.Button(
            dialog,
            text="Close",
            command=dialog.destroy,
            bg="#34495E",
            fg="white",
            font=("Arial", 10, "bold"),
            width=12,
        ).pack(pady=(0, 16))
        
if __name__ == "__main__":
    multiprocessing.freeze_support() 
    root = tk.Tk()
    
    def start_program():
        def launch_main_menu(): 
            AppLauncher(root)
        SplashScreen(root, launch_main_menu)

    ConfigApp(root, start_program)
    root.mainloop()