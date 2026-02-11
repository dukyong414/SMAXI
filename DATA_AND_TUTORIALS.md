# 📥 Data & Tutorials

Due to the large file size of high-speed X-ray video and 3D tomography data, the sample datasets and tutorial videos for **SynchroVision** are hosted externally on Google Drive.

### 🔗 [Download All Assets Here (Google Drive Link)](INSERT_YOUR_GOOGLE_DRIVE_LINK_HERE)

---

## 📂 What is Included?

The external repository contains the following folders and files required to run the tutorials and demo the software:

### 2. GUI Assets
* **Background image for GUI setup:** Essential for the software's visual interface.
* **Logo image for GUI setup:** Required for the main menu display.

### 3. Sample Input Data
* **High speed x-ray image (cine file):** Raw video data for testing the normalization and playback features.
* **High speed x-ray image (thermal drift):** A specific dataset designed to demonstrate the "Thermal Drift Correction" module.
* **Labeled mask data for YOLO training:** Pre-annotated data to test the machine learning training interface.
* **X-ray tomography 2D images:** Static slice images for 2D analysis.
* **X-ray tomography 3D image (human head phantom):** Large dataset for testing 3D visualization tools.

### 4. Sample Output Files
* **CSV file for geometry analysis:** Example output showing how geometric features (keyhole depth, width) are exported.
* **High speed x-ray image (normalized):** The result of running the stabilization/normalization module.
* **Trained YOLO model:** A pre-trained `.pt` model file so you can test tracking without waiting for training.

### 📚 Tutorials & Guides
* **Tutorial Slide (PDF/PPT):** Comprehensive slides explaining the software architecture and usage.
* **Video Tutorial (`.mp4`):** A step-by-step walkthrough of the software features (Recorded Jan 22, 2026).

---

## ⚙️ Setup Instructions

After downloading the files from Google Drive:

1.  **Unzip/Copy** the folders (`2.`, `3.`, `4.`) into the **root directory** of the SynchroVision software.
2.  Ensure the folder names match exactly as listed above, or the software may not find the sample files automatically.
3.  **Run `main.py`** and select the root folder when prompted.DATA_AND_TUTORIALS
