import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


from utils import *
import utils

# Import the following tools from different .py file
from normalize import NormalizerApp, ThermalCorrectionApp
from ai_tools import SamLabelerApp, YoloTrainerApp, TrackingViewer
from analysis import DataPlotterApp, EventTaggerApp

# =========================================================
# HELPER: SAFE IMAGE IO (trouble shooting for computer using non-English character)
# =========================================================
def imread_safe(path):
    """Reads an image safely, handling non-English paths (e.g., Korean)."""
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

        tk.Label(self.root, text="Select Master Project Folder", font=("Arial", 16, "bold"), bg="#ECF0F1", fg="#2C3E50").pack(pady=20)
        
        f_entry = tk.Frame(self.root, bg="#ECF0F1")
        f_entry.pack(fill="x", padx=20)
        
        # DEFAULT PATH
        default_path = r"C:\Users\X-ray image processing software"
        self.path_var = tk.StringVar(value=default_path)
        
        tk.Entry(f_entry, textvariable=self.path_var, font=("Consolas", 10), width=50).pack(side="left", fill="x", expand=True)
        tk.Button(f_entry, text="Browse...", command=self.browse, bg="#BDC3C7").pack(side="left", padx=5)
        
        tk.Button(self.root, text="START PROGRAM", command=self.confirm, bg="#2ECC71", fg="white", font=("Arial", 12, "bold"), width=20, height=2).pack(pady=30)
        
    def browse(self):
        d = filedialog.askdirectory()
        if d: self.path_var.set(d)
        
    def confirm(self):
        p = self.path_var.get()
        if not os.path.exists(p):
            messagebox.showerror("Error", "Folder does not exist!\nPlease select the valid root folder.")
            return
            
        # UPDATE GLOBALS IN UTILS
        utils.BASE_DIR = p
        utils.LOGO_FILE_PATH = os.path.join(utils.BASE_DIR, "2. [Image file] Logo image for GUI setup", "Logo.png")
        utils.SAM_CHECKPOINT_DEFAULT = os.path.join(utils.BASE_DIR, "1. Root folder\models", "sam_vit_b_01ec64.pth")
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
        self.root.title("X-ray Image Analysis Program")
        
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
        
        self.canvas.create_text(cx, cy_start + 200, text="Graphical User Interface for X-Ray Image Analysis", 
                                font=("Helvetica", 20, "bold"), fill="#2C3E50", width=650, justify="center")
        
        self.canvas.create_text(cx, cy_start + 300, text="Developed by FAST-AM Lab, Northwestern University", 
                                font=("Helvetica", 12, "bold"), fill="#4E058E")

        desc_text = (
            "Welcome to the X-ray Image Processing GUI.\n\n"
            "This software provides an end-to-end workflow for processing high-speed X-ray imaging data.\n"
            "Applicable field includes various x-ray imaging technology.\n"
            "\n1. Normalizer: Compensate for thermal drift effect and convert raw X-ray images into high-contrast grey-scale or binary video.\n"
            "2. AI Labeler: Annotate objects using the novel Segment Anything Model (SAM).\n"
            "3. YOLO Trainer: Aggregate datasets and train custom detection models automatically.\n"
            "4. Data Plotter: Interact and visualize geometric features (width, depth) from tracking data.\n"
            "5. Event Tagger: Mark the existence of defects (Pore/Bubble/Spatter) per frame."
        )
        
        self.canvas.create_text(cx, cy_start + 400, text=desc_text, font=("Arial", 11), fill="#34495E", 
                                justify="left", width=850)

        btn_start = tk.Button(self.root, text="Click to Start Program", font=("Arial", 14, "bold"), 
                              bg="#4E058E", fg="white", width=25, height=2, 
                              command=self.start_app, relief="raised", cursor="hand2")
        
        self.canvas.create_window(cx, cy_start + 550, window=btn_start, anchor="center")
        self.canvas.create_text(cx, cy_start + 600, text="© 2026 Northwestern University", font=("Arial", 10), fill="#2C3E50")

    def start_app(self):
        self.canvas.destroy()
        self.switch_callback()  

# =========================================================
# CLASS: APP LAUNCHER (Main Menu)
# =========================================================
class AppLauncher:
    def __init__(self, root):
        self.root = root
        self.root.title("FAST-AM Lab | Data Processing Program") 
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
        
        status_color = "black" if utils.HAS_GPU_NORMALIZER else "gray"
        lbl_hardware = tk.Label(self.canvas, text=f"Hardware: {utils.GPU_NAME}", 
                                fg=status_color, bg="white", 
                                font=("Arial", 10, "bold"), 
                                padx=10, pady=5, relief="ridge", borderwidth=2)
        self.canvas.create_window(w - 30, 30, window=lbl_hardware, anchor="ne")

        if os.path.exists(utils.LOGO_FILE_PATH):
            try:
                self.logo_img = tk.PhotoImage(file=utils.LOGO_FILE_PATH)
                self.canvas.create_image(cx, cy_start, image=self.logo_img, anchor="center")
            except: pass
        else:
            self.canvas.create_text(cx, cy_start, text="FAST-AM Lab", font=("Arial", 24, "bold"), fill="gray")

        self.canvas.create_text(cx, cy_start + 200, text="Select Tool", font=("Helvetica", 28, "bold"), fill="#2C3E50")
        
        note_text = "Note: The processing speed of each tool depends on your PC's specifications."
        self.canvas.create_text(cx, cy_start + 240, text=note_text, font=("Arial", 10, "italic"), fill="black")
        
        specs_text = f"Your System: {get_system_specs()}"
        self.canvas.create_text(cx, cy_start + 270, text=specs_text, font=("Consolas", 9, "bold"), fill="#2E86C1")

        btn_frame = tk.Frame(self.canvas, bg="white")
        btn_style = {"font": ("Arial", 14, "bold"), "fg": "white", "width": 18, "height": 4, "wraplength": 180}
        
        tk.Button(btn_frame, text="1. Normalize Data\n(GPU)", bg="#16A085", command=self.launch_normalizer, **btn_style).grid(row=0, column=0, padx=10, pady=10)
        tk.Button(btn_frame, text="2. Label Data\n(SAM AI)", bg="#2980B9", command=self.launch_labeler, **btn_style).grid(row=0, column=1, padx=10, pady=10)
        tk.Button(btn_frame, text="3. Train & Track\n(YOLO)", bg="#8E44AD", command=self.launch_trainer, **btn_style).grid(row=0, column=2, padx=10, pady=10)
        tk.Button(btn_frame, text="4. Plot Geometry\n(Data Vis)", bg="#E67E22", command=self.launch_plotter, **btn_style).grid(row=0, column=3, padx=10, pady=10)
        tk.Button(btn_frame, text="5. Event Tagger\n(Pore/Bubble)", bg="#C0392B", command=self.launch_tagger, **btn_style).grid(row=1, column=0, columnspan=4, padx=10, pady=10)

        self.canvas.create_window(cx, cy_start + 480, window=btn_frame, anchor="center")
        self.canvas.create_text(cx, cy_start + 550, text="© 2026 Northwestern University", font=("Arial", 10), fill="#BDC3C7")

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
               "THERMAL DRIFT correction first?")
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

        btn_drift = tk.Button(dialog, text="Go to Thermal Drift Tool\n(Recommended)", command=on_drift, 
                              bg="#2ECC71", fg="white", font=("Arial", 10, "bold"), width=25, height=2)
        btn_drift.place(x=50, y=100)

        btn_norm = tk.Button(dialog, text="Go directly to Normalizer\n(Skip Correction)", command=on_norm, 
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

        if self.user_choice is True: 
            ThermalCorrectionApp(self.main_container, 
                                 on_back=self.show_home, 
                                 on_proceed_to_norm=self.launch_normalizer_direct)
        else:
            NormalizerApp(self.main_container, on_back=self.show_home)

    def launch_normalizer_direct(self):
        for w in self.main_container.winfo_children(): w.destroy()
        NormalizerApp(self.main_container, on_back=self.show_home)

    def launch_labeler(self):
        for w in self.main_container.winfo_children(): w.destroy()
        SamLabelerApp(self.main_container, on_back=self.show_home)

    def launch_trainer(self):
        for w in self.main_container.winfo_children(): w.destroy()
        YoloTrainerApp(self.main_container, on_back=self.show_home)
        
    def launch_plotter(self):
        for w in self.main_container.winfo_children(): w.destroy()
        DataPlotterApp(self.main_container, on_back=self.show_home)

    def launch_tagger(self):
        for w in self.main_container.winfo_children(): w.destroy()
        EventTaggerApp(self.main_container, on_back=self.show_home)

if __name__ == "__main__":
    multiprocessing.freeze_support() 
    root = tk.Tk()
    
    def start_program():
        def launch_main_menu(): 
            AppLauncher(root)
        SplashScreen(root, launch_main_menu)

    ConfigApp(root, start_program)
    root.mainloop()