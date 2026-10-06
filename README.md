# 🎬 SegRig Studio

> ⚠️ **Work In Progress:** This project is currently in active development. Some features, systems, or fallback mechanisms might be incomplete, unstable, or subject to change.

**SegRig Studio** is a local web application designed for automated 2D character rigging prep. It leverages AI to segment characters from an image into individual transparent layers and reconstructs the missing background, streamlining the workflow for Motion Graphics and Virtual Production.

## ✨ Features
*   **Text-Prompted Segmentation:** Automatically separate layers using text prompts (e.g., `hair, face, arms, shirt`) powered by the LangSAM model.
*   **Generative Fill Background:** Seamlessly reconstruct the missing background (where the character was extracted) using Stable Diffusion 1.5 Inpainting.
*   **AE-Ready PSD Export:** Export as a `.psd` file where each layer is cropped to a tight bounding box, specifically designed for importing into After Effects using the **"Retain Layer Sizes"** option[cite: 18].
*   **ZIP & Manifest Export:** Export a ZIP file containing transparent PNGs and a `manifest.json` file for offsets, ideal for planar setups in 3D software like C4D[cite: 18].
*   **VRAM Optimized:** The script strictly manages memory by loading and tearing down heavy models (LangSAM and SD) one at a time, ensuring it runs safely on 8 GB VRAM GPUs like the RTX 2060 Super[cite: 18].

## 💻 Hardware Target
*   **OS:** Windows 10 / 11
*   **CPU:** i5-14400F or similar[cite: 18].
*   **RAM:** 32 GB[cite: 18].
*   **GPU:** NVIDIA RTX 2060 Super (8 GB VRAM) or higher[cite: 18].

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
