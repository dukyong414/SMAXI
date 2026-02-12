================================================================================
          FAST-AM X-RAY IMAGE PROCESSING SOFTWARE | INSTALLATION GUIDE
================================================================================
Author: Dukyong Kim (PhD 1st year student)
Affiliation: FAST-AM Lab, Northwestern University, Department of Mechanical Engineering
Target System: Windows / Linux / macOS
Recommended Environment: Conda (Anaconda or Miniconda)

================================================================================
1. OVERVIEW
================================================================================
This software provides an end-to-end workflow for processing high-speed X-ray 
imaging data. It includes modules for image normalization, AI-based 
segmentation (SAM), object detection (YOLO), and LLM-powered data visualization.

================================================================================
2. FILE STRUCTURE (for 1. Root folder)
================================================================================
* 1. Python code/ ..... Contains the source code (main.py, etc.)
* models/ ............. Contains various SAM model weights (sam_vit_b_01ec64.pth).
* requirements.txt .... List of required Python libraries.

================================================================================
3. INSTALLATION (CONDA)
================================================================================
Please follow these steps to set up the environment.

STEP 1: Create the Environment
Open your terminal (Anaconda Prompt) and run:
   conda create -n xray_image_software python=3.12 pip spyder

STEP 2: Activate the Environment
   conda activate xray_image_software

STEP 3: Install Python Libraries
Navigate to this folder and install dependencies:
   cd /d "PATH_TO_YOUR_SOFTWARE_FOLDER" #SOFTWARE_FOLDER should end with file path as /1. Root folder"
   pip install -r requirements.txt

STEP 4: Install Llama 3 (For AI Chat) *this is an optional feature
This software uses Ollama for the local LLM.
1. Download Ollama from https://ollama.com/download (macOS, Linux, and windows available)
2. Once installed, run this command in your terminal:
   ollama pull llama3

================================================================================
4. HOW TO RUN THE SOFTWARE
================================================================================

[Option A: Using VS Code]
1. Open Visual Studio Code.
2. Open the folder "1. Python code".
3. Select Interpreter:  #INSTALL PYTHON prior to this step.
   - Press Ctrl+Shift+P (or Cmd+Shift+P).
   - Type "Python: Select Interpreter". 
   - Choose the 'xray_image_software' environment we created above.
4. Open 'main.py' and press the Play button.

[Option B: Using Spyder]
1. Open the terminal and make sure the environment is active:
   conda activate xray_image_software
2. Launch Spyder by typing:
   spyder
3. Open 'main.py' inside Spyder and run it.

================================================================================
5. TROUBLESHOOTING
================================================================================
* "ModuleNotFoundError: No module named 'utils'":
  This happens if the IDE is looking in the wrong folder.
  Fix: Ensure you are running the script exactly from the "1. Python code" folder, 
  not the root directory.

* "Ollama connection failed":
  Ensure the Ollama app is running in the background and that you have run 
  'ollama pull llama3' successfully.

================================================================================
FAST-AM lab, Northwestern University
================================================================================