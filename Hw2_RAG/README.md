# HW2 RAG System Notebooks

This repository contains two notebooks for implementing and evaluating a Retrieval-Augmented Generation (RAG) system in Google Colab.

### **Task 1**: Retrieval-Augmented Generation (RAG) on Resume Infomation Retrieval and Summary
- Implement a Retrieval-Augmented Generation system using:
  - LLM: Phi-2
  - Embedding Model: all-MiniLM-L6-v2
- Generate:
  - Response without RAG
  - Response with RAG
- Analyze and compare the two responses.

### **Task 2**: RAG-based Page Retrieval on Slides
- Build a RAG system to retrieve pages from AI.pdf (463 pages) based on query questions.
- Each query's answer must be a single page number.
- Join the [Kaggle competition](https://www.kaggle.com/t/e5a90293e822445b98a7d60be57aa67c).
- Submit predictions in the format of HW2_template.csv(results.csv).
- Using OCR and Captioning models to improve information retrieval.
  - OCR: pytesseract
  - LLM Captioning: Phi-4-multimodal-instruct
  - Embedding Model: all-MiniLM-L6-v2

```bash
├── Hw2_task1.ipynb       # Notebook for initial RAG implementation and analysis in Resume
├── Hw2_task2.ipynb       # Notebook for enhanced RAG with OCR/captioning and page-level reasoning
└── README.md             
```

## Environment Details

- **Python Version**: 3.8 or 3.9
- **GPU (Optional)**: NVIDIA GPU (I used A100) with CUDA 11.x for accelerated model inference

## Run Notebooks

### Hw2_task1.ipynb
Open Hw2_task1.ipynb in Google Colab.

Follow the cells to load data, build the RAG pipeline, and evaluate retrieval vs. non-RAG responses.

Key Python Libraries:

```bash
!pip install langchain==0.3.23 transformers sentence-transformers chromadb torch accelerate
!pip install -U langchain-community pypdf
!pip install gdown
!gdown --id 1VEzd8186UlJ2-Ah5EepPnCqJ1C-LqAqm
```

### Hw2_task2.ipynb
Open Hw2_task2.ipynb in Google Colab.

In the first cell, install and download dependencies & data:

Hw2_task2 (**Datasets**): https://drive.google.com/drive/folders/1C7IN3IaNFRbLcny9AWfC1gkuEhznEDxC?usp=sharing

pages (**Converted Page Images for OCR**): https://drive.google.com/drive/folders/14OcRVpXzTy7wAVWvBj_HWYpryCcsts1R?usp=sharing


```bash
# 0. install gdown
pip install -q gdown
# 1. download HW2_task2 folder (Dataset)
!gdown --folder 'https://drive.google.com/drive/folders/1C7IN3IaNFRbLcny9AWfC1gkuEhznEDxC?usp=sharing' -O ./Hw2_task2
# 2. download pages folder (page images)
!gdown --folder 'https://drive.google.com/drive/folders/14OcRVpXzTy7wAVWvBj_HWYpryCcsts1R?usp=sharing' -O ./pages

# install needed libraries
!pip install pdf2image torch accelerate transformers==4.48.2 sentence-transformers chromadb pandas langchain faiss-cpu pillow pytesseract

# system utilities
!apt-get update && apt-get install -y poppler-utils tesseract-ocr
!pip install -U langchain-community pypdf
!pip install --upgrade flash-attn
!pip install backoff evaluate rouge_score datasets
```

Execute all cells to set up:
- OCR + Captioning: merges OCR text with image captions
- RAG Embedder: uses all-MiniLM-L6-v2 for vector indexing
