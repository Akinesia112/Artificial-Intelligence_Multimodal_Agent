# HW1: Multimodal LLM Agents for Image Captioning & Image Style Transfer

This repository contains code for evaluating image captioning and performing image style transfer using multimodal LLM agents. The tasks are implemented in Google Colab using Python 3 and are designed to run on an A100 GPU.

---

## Table of Contents

- Environment Details
- How to Run the Code
- **Task 1: Image Captioning Evaluation**
  - 0. Environment Setting
  - 1. BLIP: `Salesforce/blip-image-captioning-base`
  - 2. Phi‑4: `microsoft/Phi-4-multimodal-instruct`
  - 3. Load Datasets for Evaluation
  - 4. Load Evaluation Metrics
  - 5. Evaluation Function for Image Captioning
  - 6. Run Evaluations
  - 7. Case Study – Qualitative Analysis
- **Task 2-1: MLLM Image Style Transfer (Text-to-Image)**
  - 0. Environment Setting
  - 1. Download the 100 Content Images
  - **Instruction Strategy 1: With Question and Substeps**
    - 2. Load the MLLM: Phi‑4 (multimodal instruct)
    - 3. Load the T2I Model: Stable Diffusion 3 (medium)
    - 4. Create Output Folder
    - 5. Loop Over the 100 Images
  - **Instruction Strategy 2: Without Question and Substeps**
    - 2. Load the MLLM: Phi‑4 (multimodal instruct)
    - 3. Load the T2I Model: Stable Diffusion 3 (medium)
    - 4. Create Output Folder
    - 5. Loop Over the 100 Images
  - **Style Transfer on Your Profile Photo**
- **Task 2-2: MLLM Image Style Transfer (Image-to-Image)**
  - 1. Download the 100 Content Images
  - 2. Load Phi‑4 (multimodal instruct) to Generate Text Prompts
  - 3. Load Stable Diffusion v1.5 (Image-to-Image)
  - 4. Create a Function to Generate Long Prompt Embeddings
  - 5. Create the Output Folder
  - 6. Process Each Content Image
    - Parameters & Variants:
      - Short Prompts: `strength=0.75, guidance_scale=9.5`
      - Long Prompt: `strength=0.75, guidance_scale=9.5`
      - Short Prompt Variant: `strength=0.95, guidance_scale=8.5`
  - **Style Transfer on Your Profile**

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
```
