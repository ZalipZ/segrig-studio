# 🎬 SegRig Studio

> ⚠️ **Work In Progress:** This project is currently in active development. Some features, systems, or fallback mechanisms might be incomplete, unstable, or subject to change.
>> 🤖 **AI-Assisted Project:** The core logic, memory optimization, and UI structure of this project were developed with the assistance of [Claude AI](https://claude.ai/).

**SegRig Studio** is a local web application designed for automated 2D character rigging prep. It leverages AI to segment characters from an image into individual transparent layers and reconstructs the missing background, streamlining the workflow for Motion Graphics and Virtual Production.

## ✨ Features
*   **Text-Prompted Segmentation:** Automatically separate layers using text prompts (e.g., `hair, face, arms, shirt`) powered by the LangSAM model.
*   **Generative Fill Background:** Seamlessly reconstruct the missing background (where the character was extracted) using Stable Diffusion 1.5 Inpainting.
*   **AE-Ready PSD Export:** Export as a `.psd` file where each layer is cropped to a tight bounding box, specifically designed for importing into After Effects using the **"Retain Layer Sizes"** option.
*   **ZIP & Manifest Export:** Export a ZIP file containing transparent PNGs and a `manifest.json` file for offsets, ideal for planar setups in 3D software like C4D.
*   **VRAM Optimized:** The script strictly manages memory by loading and tearing down heavy models (LangSAM and SD) one at a time, ensuring it runs safely on 8 GB VRAM GPUs like the RTX 2060 Super.

## 💻 Hardware Target
*   **OS:** Windows 10 / 11
*   **CPU:** i5-14400F or similar.
*   **RAM:** 32 GB.
*   **GPU:** NVIDIA RTX 2060 Super (8 GB VRAM) or higher.

---

## 📁 Project Setup

Before installing, ensure you have the following two files in your project directory alongside `segrig_studio.py`.

### 1. `requirements.txt`
Create a file named `requirements.txt` and paste the following dependencies:

```text
# ---- PyTorch (CUDA 12.1 build for RTX 2060 Super / Turing) -------------------
torch>=2.3.1
torchvision>=0.18.1

# ---- Diffusion stack --------------------------------------------------------
diffusers==0.31.0
transformers==4.45.2
accelerate==1.0.1           # REQUIRED for enable_model_cpu_offload()
safetensors>=0.4.5
huggingface_hub>=0.25.0

# ---- Segmentation & Background Removal --------------------------------------
rembg==2.0.59
onnxruntime-gpu==1.19.2

# ---- Imaging & Processing ---------------------------------------------------
opencv-python-headless==4.10.0.84
pillow>=10.4.0
numpy<2.0                   
scipy>=1.14.0
scikit-image>=0.24.0

# ---- UI ---------------------------------------------------------------------
gradio==4.44.1
gradio-client==1.3.0

# ---- Export -----------------------------------------------------------------
pytoshop>=1.2.1
psd-tools==1.9.34

# ---- Utilities --------------------------------------------------------------
tqdm
fastapi==0.115.0
starlette==0.38.6
uvicorn==0.30.6
pydantic==2.9.2

```

## 🛠️ Installation

This project requires **Python 3.10 or 3.11** (SAM2 does not build cleanly on Python 3.12). Installing via Miniconda or Anaconda is highly recommended.

**1. Create the Environment**
Open `Anaconda Prompt` or `Miniconda Prompt` and run:
```bash
conda create -n segrig python=3.11 -y
conda activate segrig
```

**2. Install PyTorch**
You must install the `CUDA 12.1` build of PyTorch first and separately:
```bash
pip install torch==2.4.1 torchvision==0.19.1 --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
```

**3. Install LangSAM**
Install `Lang-SAM (which pulls in GroundingDINO and SAM2)` directly from the source:
```bash
pip install git+[https://github.com/luca-medeiros/lang-segment-anything.git](https://github.com/luca-medeiros/lang-segment-anything.git)
```

**4. Install Remaining Dependencies**
Install the rest of the required libraries using the `requirements.txt` file created earlier:
```bash
pip install -r requirements.txt
```

## 🚀 How to Run

**Start_SegRig.bat**
Create a file named `Start_SegRig.bat` (this will be your one-click launcher)::
```bash
@echo off
:: Locate Conda path to initialize the environment
IF EXIST "%USERPROFILE%\miniconda3\Scripts\activate.bat" (
    call "%USERPROFILE%\miniconda3\Scripts\activate.bat"
) ELSE IF EXIST "%USERPROFILE%\anaconda3\Scripts\activate.bat" (
    call "%USERPROFILE%\anaconda3\Scripts\activate.bat"
) ELSE IF EXIST "C:\ProgramData\miniconda3\Scripts\activate.bat" (
    call "C:\ProgramData\miniconda3\Scripts\activate.bat"
) ELSE (
    echo "Conda path not found! Please open Anaconda Prompt manually."
    pause
    exit
)

:: Activate the environment and run the server
call conda activate segrig
cd /d C:\AI\app
python segrig_studio.py
pause
```

# Simply double-click the Start_SegRig.bat file you created.
The script will automatically locate your Conda installation, activate the segrig environment, and launch the local server
The Gradio UI will be accessible through your web browser at http://127.0.0.1:7860/
Note: On the very first run, the system will download the AI models (LangSAM and Stable Diffusion 1.5), which may take some time depending on your internet connection

## 🎨 After Effects Workflow

#Once processing is complete and the `.psd` file is exported:
1.Open After Effects and go to File → Import → File.
2.Select the exported PSD file.
3.In the Import Kind dialogue, ensure you select Composition – Retain Layer Sizes.
4.The anchor point of each layer will be perfectly centered on its respective part, ready for immediate rigging.



