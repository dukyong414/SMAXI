# SynchroVision

**An End-to-End AI-Powered Software for Multi-Dimensional X-ray Image Analysis**

Developed by **Dukyong Kim** and **Tao Sun** *FAST-AM Lab, Northwestern University & Advanced Photon Source, Argonne National Laboratory*

---

## 📖 Overview

**SynchroVision** is an open-source software designed to address the "Big Data" challenges posed by next-generation high-energy synchrotron facilities, such as the upgraded Advanced Photon Source (APS-U).

While upgraded storage rings enable high-frame-rate acquisition over extended exposure times, they create massive data volumes and often result in reduced absorption contrast. SynchroVision solves this by providing a robust, user-friendly pipeline for **processing transient high-speed X-ray video**, **2D static images**, and **3D tomography data**.

Unlike closed-source commercial alternatives, SynchroVision is fully customizable and integrates state-of-the-art machine learning models (SAM 2, YOLO-seg, Llama 3) to automate segmentation, tracking, and geometric analysis.

---

## ✨ Key Features

SynchroVision is structured into four core modules:

### 1. Computer-Vision Assisted Image Processing
* **Thermal Drift Correction:** Automated stabilization of images to correct for thermal drift during experiments.
* **Image Normalization:** Background removal and contrast enhancement to resolve vague image features.

### 2. ML-Powered Object Annotation & Tracking
* **Zero-Shot Detection:** Utilizes **SAM 2 (Segment Anything Model)** for robust image annotation without extensive pre-training.
* **Object Tracking:** Implements **YOLO-seg** for high-speed tracking of dynamic objects (e.g., keyholes, pores) in both 2D X-ray images and 3D CT scans.

### 3. Interactive Geometry Feature Analysis
* **Geometric Quantification:** Automatically extracts and plots features such as keyhole depth, width, area, and aspect ratio over time.
* **LLM Integration:** Features a built-in chatbot powered by **Llama 3**. Users can "chat" with their data to extract advanced insights and statistical summaries interactively.

### 4. Transient Event Log Tagger
* **Event Logging:** A dedicated interface for tagging transient events (e.g., spatter, bubble formation, pore generation).
* **Data Export:** Exports tagged events with precise timestamps and coordinate locations to CSV for further analysis.

---

## 🛠️ Installation

### Prerequisites
* **Python 3.x**
* **GPU:** Recommended for ML training and tracking modules (CUDA-enabled device).

### Setup
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YourUsername/SynchroVision.git](https://github.com/YourUsername/SynchroVision.git)
    cd SynchroVision
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment (Conda or venv).
    ```bash
    pip install -r requirements.txt
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

* **Dukyong Kim:** [Insert Email]
* **Tao Sun:** taosun@northwestern.edu
* **Department:** Mechanical Engineering, Northwestern University
