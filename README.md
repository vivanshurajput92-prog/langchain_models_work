# LangChain & Hugging Face Models

This repository contains a collection of Python scripts demonstrating how to integrate **Hugging Face** models (both downloaded locally and accessed via the Inference API) with **LangChain**. It covers text generation using Large Language Models (LLMs) and document similarity using embedding models.

## 🚀 Features & File Structure

* **`models/chat_model_hf_local.py`**: Demonstrates how to download and run a Hugging Face text-generation model (like TinyLlama) entirely locally using `HuggingFacePipeline`. Includes local caching configurations (`HF_HOME`) so the model only downloads once.
* **`models/embedding_model.py`**: Shows how to generate vector embeddings for text using Hugging Face models.
* **`models/document_similarity.py`**: A practical implementation that compares a user's query against a list of documents using Cosine Similarity to find the closest matching sentence.
* **`models/chat_model_demo.py`**: General chat model experimentation.

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/vivanshurajput92-prog/langchain_models_work.git](https://github.com/vivanshurajput92-prog/langchain_models_work.git)
   cd langchain_models_work
