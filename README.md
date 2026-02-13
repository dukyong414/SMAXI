# SynchroVision

**An Open-source & End-to-End & AI-Powered Software for Multi-Dimensional X-ray Image Analysis**

Developed by **FAST-AM Lab**, Northwestern University, Evanston, IL (Department of Mechanical Engineering)  
[Visit Lab Website](https://fast-am.mech.northwestern.edu/)

*Research was conducted in active collaboration with Advanced Photon Source, Argonne National Laboratory, Lemont, IL*

<img width="1784" height="487" alt="Logo" src="https://github.com/user-attachments/assets/ef9646a3-3f87-4641-9bc1-13122e4ac1c4" />

---

## 📖 Overview

**SynchroVision** is an open-source software designed to address the "Big Data" challenges posed by next-generation high-energy synchrotron facilities, such as the upgraded Advanced Photon Source (APS-U).

While upgraded storage rings enable high-frame-rate acquisition over extended exposure times and allows a wide variety of x-ray experiment, they create massive data volumes and brighter beam often result in reduced absorption contrast in the collected images. Our developed SynchroVision solves this by providing a robust, user-friendly pipeline for **processing transient high-speed X-ray video**, **2D static images**, and **3D tomography data**.

Unlike closed-source commercial alternatives, SynchroVision is fully customizable and integrates state-of-the-art machine learning models (SAM 2, YOLO-seg, Llama 3) to facilitate accurate and fast object segmentation, object tracking, and transient geometric analysis.

## 🖼️ Diagram
The below diagram summarizes the key features of SynchroVision software.

<img width="1789" height="1091" alt="Graphical abstract" src="https://github.com/user-attachments/assets/c6ba1649-75ec-42e6-8798-60729ea82247" />

---

## ✨ Key Features

SynchroVision software is structured into four core modules:

### 1. Image Processing (Stabilization & Normalization)
* **Thermal Drift Correction:** Computer vision-assisted automated stabilization of high-speed X-ray images to correct for thermal drift-triggered surface line changes during laser powder bed fusion (L-PBF) experiments.
* **Image Normalization:** Background removal and contrast enhancement to resolve vague image features (Grey-scale or Binary normalization are both available).

### 2. ML-Powered Object Annotation & Tracking
* **Zero-Shot Detection:** Utilizes **SAM 1 and 2 (Segment Anything Model)** for robust and accurate image annotation without extensive pre-training. This is highly beneficial in reducing the data labeling cost and time, where many commercial labeling apps require subscription. 
* **Object Tracking:** Implements **YOLO-seg** for high-speed tracking of dynamic objects (e.g., keyholes) in both 2D X-ray images and 3D computer tomography (CT) scans.

### 3. Interactive Geometry Feature Analysis
* **Geometric Quantification:** Automatically extracts and plots object's transient geometricl features such as keyhole depth, width, area, and aspect ratio over time.
* **LLM Integration:** Features a built-in chatbot powered by **Llama 3**. Users can "chat" with extracted data to understand advanced insights and statistical summaries interactively.

### 4. Transient Advanced Image Analysis
* **Event Logging:** A dedicated interface for tagging transient events (e.g., spatter, bubble formation, pore generation).
* **In-situ & Ex-situ Label-time Difference Calculator:** Interactive GUI for calculating the difference in In-situ and Ex-situ labeled time to compensate for inherent time lags.

---

## 📦 Installation Package Explanation

Due to large file sizes, the complete dataset, source code package, and tutorial materials are hosted externally.
Python files can be found in the repository. If any modifications are made to Python files in repository, users can replace them with Python files in the Google Drive below. 


### 🔗 [Download All Assets Here (Google Drive Link)](https://drive.google.com/drive/folders/1woC6zvyxjAKdQ0fuNuI6pKW8LH5Ltft8?usp=sharing)

**The external repository contains:**

* 📂 **1. Root folder:** The core software directory containing the main Python scripts, README.txt, requirements.txt, SAM 1&2 models 
* 📂 **2. GUI Assets:**
    * Background image for GUI setup
    * Logo image for GUI setup
* 📂 **3. Sample Input Files:**
    * High-speed X-ray image (cine file)
    * High-speed X-ray image (thermal drift correction demo)
    * Labeled mask data for YOLO training
    * X-ray tomography 2D images (images from APS XSD-IMG group)
    * X-ray tomography 3D image (human head opendata, MSD Cardiac dataset)
* 📂 **4. Sample Output Files:**
    * CSV file for geometry analysis
    * High-speed X-ray image (normalized output)
    * Trained YOLO model (`.pt` file)
* 📚 **Tutorials:**
    * Tutorial Slides (`.pdf` and `.pptx`)
    * X-ray image analysis program tutorial video (`.mp4` - Recorded Jan 22, 2026)
 
**Setup Instruction:**
Download all folders (`1.` through `4.`) and place them in your workspace. Ensure the folder structure remains exactly as downloaded so the software can locate dependencies automatically.
<br />
Below is the tutorial video.
[![Watch the video](https://img.youtube.com/vi/K2RnguNqBVk/maxresdefault.jpg)](https://youtu.be/K2RnguNqBVk)

---

## 🛠️ Software Installation

**Target System:** Windows / Linux / macOS  
**Recommended Environment:** Conda (Anaconda or Miniconda)

### 1. Prerequisites & File Structure
Before installing, ensure your project folder is organized as follows:
* **`1. Python code/`**: Contains the source code (e.g., `main.py`).
* **`models/`**: Stores SAM models (e.g., `sam_vit_b_01ec64.pth`).
* **`requirements.txt`**: List of required Python libraries.

### 2. Installation Steps (Conda)

**Step 1: Create the Environment** Open your terminal (Anaconda Prompt) and run the following command to create a new environment named `xray_image_software` with Python 3.12:
```bash
conda create -n xray_image_software python=3.12 pip spyder
```
**Step 2: Activate the Environment** 
```bash
conda activate xray_image_software
```

**Step 3: Install Python Libraries** Navigate to the root folder (where requirements.txt is located) and install dependencies:
```bash
cd /d "Modify this path (PATH_TO_YOUR_SOFTWARE_FOLDER)_(file path must end with/1. Root folder)"
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
(1) Launch the Software: Run the main Python script in your IDE or terminal.
```bash
python main.py
```
(2) Select Workspace: Upon launch, select the root folder inside the software package when prompted.
(3) Main Menu Navigation: The dashboard provides access to the four main tools:

* Normalize Data (GPU): For drift correction and background removal.

* Label Data (SAM AI): For annotating regions of interest.

* Train & Track (YOLO): For training the model and tracking objects in video files.

* Plot Geometry (Data Vis): For visualizing results and using the LLM chatbot.

* Event Tagger: For manual logging of specific experimental events.


## 📄 Citation
If you use SynchroVision in your research, please cite:

Kim, D., et al. "SynchroVision: An end-to-end open-source artificial intelligence-powered software for multi-dimensional x-ray image analysis."

## 📬 Contact

Dukyong Kim: kdy0414@u.northwestern.edu

## 🧠 AI Models & Acknowledgments

This software leverages state-of-the-art artificial intelligence models to achieve high-precision analysis:

* **SAM (Segment Anything Model)** by Meta AI – Integrated for zero-shot image segmentation and annotation.
* **YOLO-seg (You Only Look Once)** by Ultralytics – Integrated for high-speed object tracking and instance segmentation.
* **Llama 3** by Meta AI – Powering the local "Chat with Data" and geometric feature analysis.
* **Google Gemini** – Utilized for software architecture planning, code development, and optimization.
