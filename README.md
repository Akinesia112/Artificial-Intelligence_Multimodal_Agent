# Artificial-Intelligence_Multimodal_Agent
Artificial Intelligence (NTU CSIE 5400)

This repository contains all homework projects for the Artificial Intelligence (AI) course, covering multimodal LLM agents, retrieval-augmented generation (RAG) systems, and multi-agent systems.

Artificial Intelligence (AI) refers to the sophisticated capabilities of machines that mimic human cognitive functions, including reasoning, learning, planning, and creativity.

Through these projects, we aim to:

- Construct and comprehend both classic and agentic AI principles.
- Apply AI methodologies to tackle complex real-world challenges across various fields.

- **Topics**:
  - Foundations of **classic** and **agentic** AI paradigms
  - **Knowledge representation**, **reasoning**, and **learning** methodologies
  - **Multi-agent system** design and interaction principles
- **Objective**:
  - Identify and apply appropriate AI techniques to solve complex real-world problems across multiple domains.

---

## HW1: Multimodal LLM Agents for Image Captioning & Style Transfer

This project focuses on evaluating image captioning and performing style transfer tasks using multimodal large language model (LLM) agents.

- **Tasks**:
  - **Image Captioning**: Generate descriptive captions for input images.
  - **Text-to-Image Style Transfer**: Modify images based on textual style instructions.
  - **Image-to-Image Style Transfer**: Transform images by transferring styles from one image to another.

---

## HW2: Retrieval-Augmented Generation (RAG) System

This project involves building and evaluating RAG systems using Google Colab.

### Task 1: Resume Information Retrieval and Summarization

- **Goal**: Implement a RAG system that retrieves and summarizes resume information.
- **Models Used**:
  - **LLM**: [Phi-2](https://huggingface.co/microsoft/phi-2)
  - **Embedding Model**: [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- **Deliverables**:
  - Response without RAG
  - Response with RAG
  - Analysis comparing both responses

### Task 2: RAG-based Page Retrieval on Lecture Slides

- **Goal**: Build a RAG system to retrieve specific pages from `AI.pdf` (463 pages) based on query questions.
- **Requirements**:
  - Each query must be answered with a **single page number**.
  - Participate in a **Kaggle competition** for ranking.
  - Submit predictions in the format `HW2_template.csv` (results.csv).
- **Enhancements**:
  - Use **OCR** and **Captioning** to improve document retrieval:
    - **OCR**: `pytesseract`
    - **Captioning LLM**: `Phi-4-multimodal-instruct`
    - **Embedding Model**: `all-MiniLM-L6-v2`

---

## HW3: Multi-Agent Systems

This project introduces the construction and understanding of multi-agent artificial intelligence systems.

---
