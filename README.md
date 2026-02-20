# Multimodal RAG Engine for Industrial Knowledge Extraction

This repository contains the development of my Master’s Thesis in Electronic Engineering at the University of Genoa, conducted in collaboration with **ABB**. The project focuses on the automated extraction of knowledge from multimodal data (video, audio, text) within complex industrial environments.

The system architecture is designed for modularity and scalability, implementing two distinct pipelines distributed across dedicated branches to address different performance and integration requirements.



## 🏗️ System Architecture

The project compares and implements two different approaches to Retrieval-Augmented Generation (RAG):

| Feature | **Industrial Implementation** (`ABB`) | **Academic Baseline** (`University`) |
| :--- | :--- | :--- |
| **Primary Goal** | Industrial scalability and precision on technical video | Conceptual validation and academic portability  |
| **Embeddings** | **Nomic** (Text) + **CLIP** (Visual) | **CLIP** (Vision-Text Alignment)  |
| **Data Ingestion** | Whisper (ASR) + Multimodal Ingestion | OCR-based text extraction from documents  |
| **Vector Store** | **Qdrant** (Hybrid Search) | In-memory / Simple Vector Store  |
| **LLM Inference** | **Qwen / Llama** (Optimized for reasoning)  | **Mistral** (Baseline performance)  |

---

## 🚀 Implementations

### 1. Industrial Pipeline (`branch: ABB`)
This version represents the core of the collaboration with ABB. It is optimized to handle industrial video datasets where timestamp accuracy and correlation between speech and visual cues are critical.
* **Multimodal Ingestion**: An asynchronous pipeline for audio extraction (Whisper) and intelligent frame-sampling.
* **Hybrid Retrieval**: Leverages **Qdrant** for high-density vector search combined with semantic filters to minimize hallucinations.
* **State-of-the-Art Models**: Integration with **Qwen** and **Nomic Embeddings** to maximize technical domain understanding.

### 2. Academic Baseline (`branch: University`)
A streamlined version developed to test core retrieval and inference concepts in less resource-intensive contexts.
* **OCR Integration**: Focuses on extracting textual metadata from static documents and slides.
* **Mistral Integration**: Utilizes Mistral-7B to generate responses based on contexts retrieved via CLIP.
* **Lightweight GUI**: A Streamlit-based dashboard for direct interaction and retrieval analysis.

---

## 🛠️ Tech Stack
* **Languages**: Python (NumPy, PyTorch).
* **GenAI**: Whisper (OpenAI), CLIP, Llama, Qwen, Mistral.
* **Vector DB**: Qdrant.
* **Frameworks**: Streamlit, Transformers, HuggingFace, Git.

## 👨‍💻 Author
**Riccardo Suma**
* MSc Student in Electronic Engineering - University of Genoa.
* [LinkedIn Profile](https://linkedin.com/in/riccardo-suma-823913260).
* [GitHub Portfolio](https://github.com/RiccardoSuma).

---