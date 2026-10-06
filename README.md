# segrig-studio
local web application designed for automated 2D character rigging prep. It leverages AI to segment characters from an image into individual transparent layers and reconstructs the missing background, streamlining the workflow for Motion Graphics and Virtual Production.

# 🎬 SegRig Studio

> ⚠️ **Work In Progress:** This project is currently in active development. Some features, systems, or fallback mechanisms might be incomplete, unstable, or subject to change.

**SegRig Studio** is a local web application designed for automated 2D character rigging prep. It leverages AI to segment characters from an image into individual transparent layers and reconstructs the missing background, streamlining the workflow for Motion Graphics and Virtual Production.

## ✨ Features
*   **Text-Prompted Segmentation:** Automatically separate layers using text prompts (e.g., `hair, face, arms, shirt`) powered by the LangSAM model.
*   **Generative Fill Background:** Seamlessly reconstruct the missing background (where the character was extracted) using Stable Diffusion 1.5 Inpainting[cite: 18].
*   **AE-Ready PSD Export:** Export as a `.psd` file where each layer is cropped to a tight bounding box, specifically designed for importing into After Effects using the **"Retain Layer Sizes"** option[cite: 18].
*   **ZIP & Manifest Export:** Export a ZIP file containing transparent PNGs and a `manifest.json` file for offsets, ideal for planar setups in 3D software like C4D[cite: 18].
*   **VRAM Optimized:** The script strictly manages memory by loading and tearing down heavy models (LangSAM and SD) one at a time, ensuring it runs safely on 8 GB VRAM GPUs like the RTX 2060 Super[cite: 18].

## 💻 Hardware Target
*   **OS:** Windows 10 / 11
*   **CPU:** i5-14400F or similar[cite: 18].
*   **RAM:** 32 GB[cite: 18].
*   **GPU:** NVIDIA RTX 2060 Super (8 GB VRAM) or higher[cite: 18].

---

## 🛠️ Installation

This project requires **Python 3.10 or 3.11** (SAM2 does not build cleanly on Python 3.12). Installing via Miniconda or Anaconda is highly recommended.

**1. Create the Environment**
Open `Anaconda Prompt` or `Miniconda Prompt` and run:
```bash
conda create -n segrig python=3.11 -y
conda activate segrig
