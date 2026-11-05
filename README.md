**SMAXI (Software for Machine-learning Analysis of X-ray Images): An Open-source, End-to-End, AI-Powered Software**

Developed by **FAST-AM Lab**, Northwestern University, Evanston, IL (Department of Mechanical Engineering)  
[Visit Lab Website](https://fast-am.mech.northwestern.edu/)

<img width="1784" height="487" alt="Logo" src="Logo.png" />

<!-- Video: clickable YouTube thumbnail -->
[![SMAXI Tutorial Video](https://img.youtube.com/vi/sCgA6VVVTUc/maxresdefault.jpg)](https://youtu.be/sCgA6VVVTUc)

## 📖 Overview

**SMAXI (Software for Machine-learning Analysis of X-ray Images)** is an open-source software designed to address the "Big Data" bottlenecks inherent in modern full-field X-ray imaging techniques. While advancements in both laboratory-scale x-ray sources and high-energy synchrotron facilities enable rapid, high-resolution data acquisition, the manual processing of these massive and multidimensional datasets has become prohibitively tedious.

SMAXI solves this by providing a robust, user-friendly software for processing static 2D x-ray image, transient high-speed 2D x-ray video, and 3D computed tomography (CT) data.

Unlike closed-source commercial alternatives that restrict customization, SMAXI is fully extensible and integrates state-of-the-art machine learning foundational models (SAM 1 & 2, YOLO v8 & v12, Llama 3) to automate and accelerate image pre-processing, precise object segmentation, dynamic object tracking, and complex geometric feature analysis.

## 🖼️ Diagram
The diagram below summarizes the core architecture and key modules of SMAXI.

<img width="1789" height="1091" alt="Graphical abstract" src="Abstract.png" />

---

## ✨ Key Features

SMAXI is structured into four primary analytical modules, supplemented by advanced auxiliary tools:

### 1. Image Pre-processor (Image Stabilization & Normalization)
* **Thermal Drift Correction:** Computer vision-assisted automated stabilization of high-speed x-ray images to track and correct for thermal drift-triggered surface shifts (e.g., substrate's thermal expansion during laser powder bed fusion).
* **Image Normalization:** Background removal and contrast enhancement designed to resolve vague features and isolate phenomena like keyholes and melt pools. Supports both continuous grayscale and binary normalization, with optional CLAHE illumination flattening and a temporal moving median.
* **Frame review:** Step through frames one at a time in the normalizer and in a comparison window. The comparison window can hide the original video and show only the normalized result.
* **Batch keyhole pipeline:** Point at a folder of `.cine` files to normalize each one and then run YOLO keyhole tracking.

### 2. ML-Powered Object Segmenter
* **Zero-Shot Detection:** Utilizes foundational **Segment Anything Models (SAM 1 & SAM 2)** for robust, prompt-based 2D, 3D, and 4D object segmentation. This drastically reduces the tedious manual labeling overhead required by traditional polygon methods, replacing it with efficient point-and-click or bounding-box interactions.
* **Optional frame propagation:** In 2D segmentation, **Propagate frames** carries the current mask across the image sequence. SAM re-segments each next frame from the previous mask. Leave it unused if you prefer to label frames yourself.

### 3. ML-Powered Object Tracker
* **Object Tracking:** Implements YOLOv8 and YOLOv12 architectures trained directly on SAM-generated masks for automated, continuous high-speed tracking of dynamic target objects (e.g., rapidly evolving keyhole vapor depressions during laser powder bed fusion process or movement of tool pin during friction stir welding process).

### 4. Interactive Geometry Feature Analyser
* **Geometric Quantification:** Automatically extracts, quantifies, and plots temporal geometric metrics—such as object depth, width, area, and aspect ratio—derived directly from the tracking data.
* **LLM Integration:** A built-in conversational assistant powered by Llama 3 (via Ollama). Users can query extracted datasets in natural language for statistical summaries.

### Auxiliary feature
* **Transient Event Tagger:** A dedicated interface for manually identifying, tagging, and exporting localized transient events (e.g., spatter, bubble formation, or pore generation) with precise spatial and temporal coordinates.

---

## 📦 Download

Get the software from this GitHub repository. Use **Code → Download ZIP**, or clone it:

```bash
git clone https://github.com/dukyong414/SMAXI.git
```

That download is the source code (`1. Python code/1_main.py`) and `2. Installation/requirements.txt`. You do not need a Google Drive zip to get the program.

Large files stay on Google Drive because GitHub is a poor place for model weights, sample cine/CT volumes, and the tutorial video. Open the folder below and download only the pieces you need. Avoid **Download all**, which packs every sample and model into one large zip.

### 🔗 [Optional large files (Google Drive folder)](https://drive.google.com/drive/folders/1woC6zvyxjAKdQ0fuNuI6pKW8LH5Ltft8?usp=sharing)

Download these folders individually:

| What you need | Drive folder | Required? |
| --- | --- | --- |
| SAM weights | `1. Root folder/models` | Yes, for segmentation |
| GUI background and logo | `2. [Image file] ...` | Yes, for the welcome screen |
| Sample cine, CT, and YOLO labels | `3. [Sample file for input] ...` | No |
| Example outputs and a trained YOLO weight | `4. [Output file of the software] ...` | No |
| Tutorial slides and video | files at the top of the Drive folder | No |

Place the GitHub `1. Python code` folder inside your `1. Root folder` (replace the older copy if you already downloaded one from Drive).

**The Drive folder also contains:**

* 📂 **1. Root folder:** The core software directory containing the main Python scripts, README.txt, requirements.txt, SAM 1&2 models 
* 📂 **2. GUI Assets:**
    * Background image for GUI setup
    * Logo image for GUI setup
* 📂 **3. Sample Input Files:**
    * High-speed X-ray image (cine file)
    * High-speed X-ray image (thermal drift correction demo)
    * Labeled mask data for YOLO training
    * X-ray tomography 2D images (images from TomoBank)
    * X-ray tomography 3D image (human head opendata, MSD Cardiac dataset)
* 📂 **4. Sample Output Files:**
    * CSV file for geometry analysis
    * High-speed X-ray image (normalized output)
    * Trained YOLO model (`.pt` file)
* 📚 **Tutorials:**
    * Tutorial Slides (`.pdf` and `.pptx`)
    * X-ray image analysis program tutorial video (`.mp4` - Recorded June 11, 2026)


---

## 🛠️ Software Installation

**Target System:** Windows / Linux / macOS  
**Recommended Environment:** Conda (Anaconda or Miniconda)

### 1. Prerequisites & File Structure
Before installing, ensure your project folder is organized as follows:
* **`1. Python code/`**: Source code. Start the program with `1_main.py`.
* **`models/`**: SAM weights (for example `sam_vit_b_01ec64.pth`), downloaded from Drive.
* **`requirements.txt`**: Python libraries. In this repository the file is `2. Installation/requirements.txt`.

### 2. Installation Steps (Conda)

**Step 1: Create the Environment** Open your terminal (Anaconda Prompt) and run the following command to create a new environment named `xray_image_software` with Python 3.12:
```bash
conda create -n xray_image_software python=3.12 pip spyder
```
**Step 2: Activate the Environment** 
```bash
conda activate xray_image_software
```

**Step 3: Install Python Libraries** From this repository, install `2. Installation/requirements.txt`. If you are using the full Drive package, the same file is also in `1. Root folder`.

```bash
cd /d "PATH_TO_SMAXI\2. Installation"
pip install -r requirements.txt
```

**Step 4: Install Llama 3 (Optional)** This software uses Ollama for the local LLM chat feature.

(1) Download Ollama from this link. 
[Visit Ollama website](https://ollama.com/download)

(2) Once installed, run this command in your terminal:
```bash
ollama pull llama3
```
## 🚀 Usage

Follow the steps below to launch the software using your preferred IDE.

### [Option A] Using Visual Studio Code
1. **Open the Project:** Launch VS Code and open the folder `1. Python code`.
2. **Select Interpreter:** * *Note: Ensure Python is installed prior to this step.*
   * Press `Ctrl+Shift+P` (Windows/Linux) or `Cmd+Shift+P` (Mac) to open the Command Palette.
   * Type **"Python: Select Interpreter"** and select it.
   * Choose the `xray_image_software` environment from the list.
3. **Run the Application:**
   * Open `1_main.py` from the file explorer.
   * Click the **Play** button in the top-right corner of the editor.

### [Option B] Using Spyder
1. **Activate the Environment:** Open your terminal or Anaconda Prompt and run:
   ```bash
   conda activate xray_image_software
   ```
2. **Launch Spyder:** Start the IDE by typing
   ```bash
   spyder
   ```
3. **Run the Software**
   * Once Spyder opens, open `1_main.py` from the file menu.
   * Press F5 or click the green Run icon in the toolbar.
   

<br />
<br />
Below is the tutorial video.
<br />

[![SMAXI Tutorial Video](https://img.youtube.com/vi/hc3nLJa31L8/maxresdefault.jpg)](https://youtu.be/hc3nLJa31L8)



## 📬 Contact

Dukyong Kim: kdy0414@u.northwestern.edu

## 🧠 AI Models & Acknowledgments

This software leverages state-of-the-art artificial intelligence models to achieve high-precision analysis:

* **SAM (Segment Anything Model)** by Meta AI – Integrated for zero-shot image segmentation and annotation.
* **YOLO-seg (You Only Look Once)** by Ultralytics – Integrated for high-speed object tracking and instance segmentation.
* **Llama 3** by Meta AI – Powering the local "Chat with Data" and geometric feature analysis.
* **Google Gemini** – Utilized for software architecture planning, code development, and optimization.
