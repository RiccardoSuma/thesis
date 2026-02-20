# Multimodal RAG Engine for Industrial Knowledge Extraction

[cite_start]This repository contains the development of my Master’s Thesis in Electronic Engineering at the University of Genoa, conducted in collaboration with **ABB**[cite: 19, 52]. [cite_start]The project focuses on the automated extraction of knowledge from multimodal data (video, audio, text) within complex industrial environments[cite: 19, 52].

[cite_start]The system architecture is designed for modularity and scalability, implementing two distinct pipelines distributed across dedicated branches to address different performance and integration requirements[cite: 41, 52].



## 🏗️ System Architecture

[cite_start]The project compares and implements two different approaches to Retrieval-Augmented Generation (RAG):

| Feature | **Industrial Implementation** (`industrial-abb`) | **Academic Baseline** (`university-baseline`) |
| :--- | :--- | :--- |
| **Primary Goal** | [cite_start]Industrial scalability and precision on technical video [cite: 19, 52] | [cite_start]Conceptual validation and academic portability  |
| **Embeddings** | [cite_start]**Nomic** (Text) + **CLIP** (Visual) [cite: 19, 52] | [cite_start]**CLIP** (Vision-Text Alignment)  |
| **Data Ingestion** | [cite_start]Whisper (ASR) + Multimodal Ingestion [cite: 19, 52] | [cite_start]OCR-based text extraction from documents  |
| **Vector Store** | [cite_start]**Qdrant** (Hybrid Search) [cite: 19, 52] | [cite_start]In-memory / Simple Vector Store  |
| **LLM Inference** | [cite_start]**Qwen / Llama** (Optimized for reasoning)  | [cite_start]**Mistral** (Baseline performance)  |

---

## 🚀 Implementations

### 1. Industrial Pipeline (`branch: industrial-abb`)
[cite_start]This version represents the core of the collaboration with ABB[cite: 19]. [cite_start]It is optimized to handle industrial video datasets where timestamp accuracy and correlation between speech and visual cues are critical.
* [cite_start]**Multimodal Ingestion**: An asynchronous pipeline for audio extraction (Whisper) and intelligent frame-sampling[cite: 19, 52].
* [cite_start]**Hybrid Retrieval**: Leverages **Qdrant** for high-density vector search combined with semantic filters to minimize hallucinations[cite: 19, 52].
* [cite_start]**State-of-the-Art Models**: Integration with **Qwen** and **Nomic Embeddings** to maximize technical domain understanding.

### 2. Academic Baseline (`branch: university-baseline`)
[cite_start]A streamlined version developed to test core retrieval and inference concepts in less resource-intensive contexts.
* [cite_start]**OCR Integration**: Focuses on extracting textual metadata from static documents and slides.
* [cite_start]**Mistral Integration**: Utilizes Mistral-7B to generate responses based on contexts retrieved via CLIP.
* [cite_start]**Lightweight GUI**: A Streamlit-based dashboard for direct interaction and retrieval analysis.

---

## 🛠️ Tech Stack
* [cite_start]**Languages**: Python (NumPy, PyTorch)[cite: 21, 54].
* [cite_start]**GenAI**: Whisper (OpenAI), CLIP, Llama, Qwen, Mistral[cite: 19, 23, 52, 55].
* [cite_start]**Vector DB**: Qdrant[cite: 19, 23, 55].
* [cite_start]**Frameworks**: Streamlit, Transformers, HuggingFace, Git[cite: 25, 52, 56].

## 👨‍💻 Author
[cite_start]**Riccardo Suma** [cite: 1, 36]
* [cite_start]MSc Student in Electronic Engineering - University of Genoa[cite: 8, 43].
* [cite_start][LinkedIn Profile](https://linkedin.com/in/riccardo-suma-823913260)[cite: 2, 37].
* [cite_start][GitHub Portfolio](https://github.com/RiccardoSuma)[cite: 37].

---