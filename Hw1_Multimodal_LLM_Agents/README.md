# HW1: Multimodal LLM Agents for Image Captioning & Image Style Transfer

This repository contains code for evaluating image captioning and performing image style transfer using multimodal LLM agents. The tasks are implemented in Google Colab using Python 3 and are designed to run on an A100 GPU.

---

## Table of Contents

- [Environment Details](#environment-details)
- [How to Run the Code](#how-to-run-the-code)
- [Task 1: Image Captioning Evaluation](#task-1-image-captioning-evaluation)
  - [0. Environment Setting](#0-environment-setting)
  - [1. BLIP: `Salesforce/blip-image-captioning-base`](#1-blip-salesforceblip-image-captioning-base)
  - [2. Phi‑4: `microsoft/Phi-4-multimodal-instruct`](#2-phi-4-microsoftphi-4-multimodal-instruct)
  - [3. Load Datasets for Evaluation](#3-load-datasets-for-evaluation)
  - [4. Load Evaluation Metrics](#4-load-evaluation-metrics)
  - [5. Evaluation Function for Image Captioning](#5-evaluation-function-for-image-captioning)
  - [6. Run Evaluations](#6-run-evaluations)
  - [7. Case Study – Qualitative Analysis](#7-case-study–-qualitative-analysis-of-interesting-samples)
- [Task 2-1: MLLM Image Style Transfer (Text-to-Image)](#task-2-1-mllm-image-style-transfer-text-to-image)
  - [0. Environment Setting](#0-environment-setting-1)
  - [1. Download the 100 Content Images](#1-download-the-100-content-images)
  - [Instruction Strategy 1: With Question](#instruction-strategy-1-with-question)
    - [2. Load the MLLM: Phi‑4 (multimodal instruct)](#2-load-the-mllm-phi-4-multimodal-instruct)
    - [3. Load the T2I Model: Stable Diffusion 3 (medium)](#3-load-the-t2i-model-stable-diffusion-3-medium)
    - [4. Create Output Folder](#4-create-output-folder)
    - [5. Loop Over the 100 Images](#5-loop-over-the-100-images)
  - [Instruction Strategy 2: Without Question](#instruction-strategy-2-without-question)
    - [2. Load the MLLM: Phi‑4 (multimodal instruct)](#2-load-the-mllm-phi-4-multimodal-instruct-1)
    - [3. Load the T2I Model: Stable Diffusion 3 (medium)](#3-load-the-t2i-model-stable-diffusion-3-medium-1)
    - [4. Create Output Folder](#4-create-output-folder-1)
    - [5. Loop Over the 100 Images](#5-loop-over-the-100-images-1)
  - [Style Transfer on Your Profile Photo](#style-transfer-on-your-profile-photo)
- [Task 2-2: MLLM Image Style Transfer (Image-to-Image)](#task-2-2-mllm-image-style-transfer-image-to-image)
  - [1. Download the 100 Content Images](#1-download-the-100-content-images-1)
  - [2. Load Phi‑4 (multimodal instruct) to Generate Text Prompts](#2-load-phi-4-multimodal-instruct-to-generate-text-prompts)
  - [3. Load Stable Diffusion v1.5 (Image-to-Image)](#3-load-stable-diffusion-v15-image-to-image)
  - [4. Create a Function to Generate Long Prompt Embeddings](#4-create-a-function-to-generate-long-prompt-embeddings)
  - [5. Create the Output Folder](#5-create-the-output-folder)
  - [6. Process Each Content Image](#6-process-each-content-image)
    - [Parameters & Variants:](#parameters--variants)
      - Short Prompts: `strength=0.75, guidance_scale=9.5`
      - Long Prompt: `strength=0.75, guidance_scale=9.5`
      - Short Prompt Variant: `strength=0.95, guidance_scale=8.5`
  - [Style Transfer on Your Profile](#style-transfer-on-your-profile)

---

## Environment Details

- **Device:** A100 GPU  
- **Platform:** Google Colab  
- **Python Version:** Python 3.x  
- **Key Libraries:**
  - [diffusers](https://github.com/huggingface/diffusers) (for Stable Diffusion models)
  - [transformers](https://github.com/huggingface/transformers) (for Phi‑4 and BLIP)
  - [accelerate](https://github.com/huggingface/accelerate)
  - [safetensors](https://github.com/huggingface/safetensors)
  - [Pillow](https://python-pillow.org/) (for image processing)
  - [gdown](https://github.com/wkentaro/gdown) (for downloading datasets)

**Installation Command (Colab):**

```bash
!pip install diffusers accelerate safetensors Pillow gdown
