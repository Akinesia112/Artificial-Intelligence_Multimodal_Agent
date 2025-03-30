# HW1: Multimodal LLM Agents for Image Captioning & Image Style Transfer (Text-to-Image & Image-to-Image)

This repository contains code for evaluating image captioning and performing image style transfer (Text-to-Image & Image-to-Image) using multimodal LLM agents. The tasks are implemented in Google Colab using Python 3 and are designed to run on an A100 GPU.

---

## Table of Contents

- [Environment Details](#environment-details)
- How to Run the Code: all in one `.ipynb` file.
- **Task 1: Image Captioning Evaluation**
- **Task 1: Evaluation Details**
   - Models (Restricted Using Model Card from huggingface):
      - [BLIP](https://huggingface.co/Salesforce/blip-image-captioning-base), 
      - [Phi-4](https://huggingface.co/microsoft/Phi-4-multimodal-instruct)
    - Datasets (Restricted): 
      - [MSCOCO-Test (5k)](https://huggingface.co/datasets/nlphuji/mscoco_2014_5k_test_image_text_retrieval), 
      - [flickr30k](https://huggingface.co/datasets/nlphuji/flickr30k)
    - Metrics ([intro](https://avinashselvam.medium.com/llm-evaluation-metrics-bleu-rogue-and-meteor-explained-a5d2b129e87f), [implementation](https://huggingface.co/docs/evaluate/index)) :
      - BLEU, ROUGE-1, ROUGE-2, METEOR 
  - 0.Environment Setting
  - 1.BLIP: `Salesforce/blip-image-captioning-base`
  - 2.Phi‑4: `microsoft/Phi-4-multimodal-instruct`
  - 3.Load Datasets for Evaluation
  - 4.Load Evaluation Metrics
  - 5.Evaluation Function for Image Captioning
  - 6.Run Evaluations
  - 7.Case Study – Qualitative Analysis
       
- **Task 2-1: MLLM Image Style Transfer (Text-to-Image)**
  - 0.Environment Setting
  - 1.Download the 100 Content Images
  - **Instruction Strategy 1: With Question and Substeps**
    - 2.Load the MLLM: Phi‑4 (multimodal instruct)
    - 3.Load the T2I Model: Stable Diffusion 3 (medium)
    - Login Huggingface with your own acess token (choose 'write'), and fill the token in `token="Your Token"` below:
      ```
      from huggingface_hub import login
      login()
      ```
      ```
      # ---------------------------
      # 3) Load the T2I Model: Stable Diffusion 3 (medium)
      # ---------------------------
      from diffusers import StableDiffusion3Pipeline

      pipe = StableDiffusion3Pipeline.from_pretrained(
          "stabilityai/stable-diffusion-3-medium-diffusers", 
          torch_dtype=torch.float16,
          token="Your Token")
      pipe = pipe.to("cuda")
      ```
    - 4.Create Output Folder
    - 5.Loop Over the 100 Images
  - **Instruction Strategy 2: Without Question and Substeps**
    - 2.Load the MLLM: Phi‑4 (multimodal instruct)
    - 3.Load the T2I Model: Stable Diffusion 3 (medium)
    - Login Huggingface with your own acess token (choose `write`), and fill the token in `token="Your Token"` below:
      ```
      from huggingface_hub import login
      login()
      ```
      ```
      # ---------------------------
      # 3) Load the T2I Model: Stable Diffusion 3 (medium)
      # ---------------------------
      from diffusers import StableDiffusion3Pipeline

      pipe = StableDiffusion3Pipeline.from_pretrained(
          "stabilityai/stable-diffusion-3-medium-diffusers", 
          torch_dtype=torch.float16,
          token="Your Token")
      pipe = pipe.to("cuda")
      ```
    - 4.Create Output Folder
    - 5.Loop Over the 100 Images
  - **Style Transfer on Your Profile Photo**
- **Task 2-2: MLLM Image Style Transfer (Image-to-Image)**
  - 1.Download the 100 Content Images
  - 2.Load Phi‑4 (multimodal instruct) to Generate Text Prompts
  - 3.Load Stable Diffusion v1.5 (Image-to-Image)
  - Install `!pip install compel` to solve the token length constraint problem `The following part of your input was truncated because CLIP can only handle sequences up to 77 tokens:`:
      ```
      from compel import Compel, ReturnedEmbeddingsType

      # Use stable-diffusion-v1-5 tokenizer and weight initialization compel
      compel = Compel(tokenizer=pipe.tokenizer , text_encoder=pipe.text_encoder, returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED, requires_pooled=[False, True])
      ```
  - 4.Create a Function to Generate Long Prompt Embeddings
  - 5.Create the Output Folder
  - 6.Process Each Content Image
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
  - [Pillow](https://github.com/python-pillow/Pillow) (for image processing)
  - [gdown](https://github.com/wkentaro/gdown) (for downloading datasets)

**Installation Command (Colab):**
❗To avoid Phi-4 with `TypeError: bad operand type for unary -: ‘NoneType’`
Plese set transformers==4.48.2（see an [external site](https://huggingface.co/microsoft/Phi-4-multimodal-instruct/discussions/36Links)).

```bash
!pip install diffusers accelerate safetensors Pillow gdown backoff evaluate rouge_score datasets transformers==4.48.2
```
