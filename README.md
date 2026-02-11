# SynchroVision

**An End-to-End AI-Powered Software for Multi-Dimensional X-ray Image Analysis**

Developed by _FAST_-AM Lab, Northwestern University (Department of Mechanical Engineering) [link](https://fast-am.mech.northwestern.edu/)

Research was conducted in active collaboration with Argonne National Laboratory (Advanced Photon Source)*

---

## 📖 Overview

**SynchroVision** is an open-source software designed to address the "Big Data" challenges posed by next-generation high-energy synchrotron facilities, such as the upgraded Advanced Photon Source (APS-U).

While upgraded storage rings enable high-frame-rate acquisition over extended exposure times, they create massive data volumes and often result in reduced absorption contrast. SynchroVision solves this by providing a robust, user-friendly pipeline for **processing transient high-speed X-ray video**, **2D static images**, and **3D tomography data**.

Unlike closed-source commercial alternatives, SynchroVision is fully customizable and integrates state-of-the-art machine learning models (SAM 2, YOLO-seg, Llama 3) to facilitate accurate and fast object segmentation, object tracking, and transient geometric analysis.

## 🖼️ Diagram
The below diagram summarizes the key features of SynchroVision software.
<img width="1789" height="1091" alt="Graphical abstract" src="https://github.com/user-attachments/assets/c6ba1649-75ec-42e6-8798-60729ea82247" />

---

## ✨ Key Features

SynchroVision software is structured into four core modules:

### 1. Image Processing (Stabilization & Normalization)
* **Thermal Drift Correction:** Computer vision-assisted automated stabilization of high-speed x-ray images to correct for thermal drift-triggered surface line change during laser powder bed fusion (L-PBF) experiments.
* **Image Normalization:** Background removal and contrast enhancement to resolve vague image features (Grey-scale or Binary normalization avaiable)

### 2. ML-Powered Object Annotation & Tracking
* **Zero-Shot Detection:** Utilizes **SAM 1 and 2 (Segment Anything Model)** for robust and accurate image annotation without extensive pre-training.
* **Object Tracking:** Implements **YOLO-seg** for high-speed tracking of dynamic objects (e.g., keyholes) in both 2-D x-ray images and 3-D computer tomography (CT) scans.

### 3. Interactive Geometry Feature Analysis
* **Geometric Quantification:** Automatically extracts and plots features such as keyhole depth, width, area, and aspect ratio over time. (features can be changed with respect to different object of interest)
* **LLM Integration:** Features a built-in chatbot powered by **Llama 3**. Users can "chat" with their data to extract advanced insights and statistical summaries interactively.

### 4. Transient Advanced Image Analysis
* **Event Logging:** A dedicated interface for tagging transient events (e.g., spatter, bubble formation, pore generation).
* **In-situ & Ex-situ Label-time Difference Calculator:** Interactive GUI for calculating difference in In-situ and Ex-situ labeled time to compensate for time difference.

---

## 🛠️ Installation
**Target System:** Windows / Linux / macOS  
**Recommended Environment:** Conda (Anaconda or Miniconda)

## 1. Prerequisites & File Structure
---
Before installing, ensure your project folder is organized as follows:
* **`1. Python code/`**: Contains the source code (e.g., `main.py`).
* **`models/`**: Stores SAM models (e.g., `sam_vit_b_01ec64.pth`).
* **`requirements.txt`**: List of required Python libraries.
---
## 2. Installation Steps (Conda)

### Step 1: Create the Environment
Open your terminal (Anaconda Prompt) and run the following command to create a new environment named `xray_image_software` with Python 3.12:

```bash
conda create -n xray_image_software python=3.12 pip spyder
    ```

---

## 🚀 Usage

1.  **Launch the Software:**
    Run the main Python script in your IDE or terminal.
    ```bash
    python main.py
    ```

2.  **Select Workspace:**
    Upon launch, select the root folder inside the software package when prompted.

3.  **Main Menu Navigation:**
    The dashboard provides access to the four main tools:
    * **Normalize Data (GPU):** For drift correction and background removal.
    * **Label Data (SAM AI):** For annotating regions of interest.
    * **Train & Track (YOLO):** For training the model and tracking objects in video files.
    * **Plot Geometry (Data Vis):** For visualizing results and using the LLM chatbot.
    * **Event Tagger:** For manual logging of specific experimental events.

---

## 📄 Citation

If you use SynchroVision in your research, please cite:

> **Kim, D.**, et al. "SynchroVision: An end-to-end open-source artificial intelligence-powered software for multi-dimensional x-ray image analysis."

---

## 📬 Contact

* **Dukyong Kim:** kdy0414@u.northwestern.edu
