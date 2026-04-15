# Evaluated Hybrid RAG Pipeline

An advanced, two-stage Retrieval-Augmented Generation (RAG) architecture built with LangChain, local LLMs (Mistral), and an automated LLM-as-a-Judge evaluation suite. 

## Motivation: Accelerating Academic Research
This project was developed to streamline the literature review process for my M.Tech research project. Reviewing dozens of dense academic papers is incredibly time-consuming. I built this custom RAG pipeline to allow researchers to instantly extract highly specific, scientifically grounded answers from their own curated library of papers. By using local models and strict faithfulness evaluation, it ensures the extracted information is accurate and reliable for academic use.

Unlike standard "Chat with PDF" tutorials, this project is engineered for **Information Retrieval accuracy** and **hallucination prevention**, utilizing Ensemble Retrieval (BM25 + Dense Vectors) and Cross-Encoder Re-ranking.

## Key Features

* **Privacy-First Local Generation:** Utilizes `Ollama` running the `Mistral` model locally, ensuring sensitive research data never leaves the machine.
* **Hybrid Search (Ensemble Retrieval):** Combines Semantic Vector Search (`all-MiniLM-L6-v2`) with Sparse Keyword Search (`BM25`) to accurately retrieve both conceptual information and specific technical acronyms.
* **Two-Stage Re-Ranking:** Implements a Contextual Compression Retriever using a Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) to mathematically re-score and filter the retrieved chunks, passing only the highest-fidelity context to the LLM.
* **Automated Evaluation (LLM-as-a-Judge):** Features a custom evaluation pipeline that dynamically scores generated answers in real-time based on two critical ML metrics:
    * **Faithfulness:** Measures if the LLM hallucinated or stuck strictly to the provided context.
    * **Answer Relevance:** Measures how effectively the generated answer addresses the user's original query.

## Architecture Details

The project follows a separation of concerns, divided into three modular components:
1.  **`ingest.py`**: Handles data pipeline ingestion, recursive character chunking (preserving paragraph/sentence boundaries), and ChromaDB vector store initialization.
2.  **`main.py`**: The core application logic containing the Ensemble Retriever, Cross-Encoder reranking, and generation prompt.
3.  **`evaluate.py`**: The MLOps testing suite containing the zero-shot prompting logic for automated pipeline grading.

## Installation & Setup

* **1. Prerequisites**
You must have [Ollama](https://ollama.com/) installed and running on your machine. Pull the Mistral model:

```bash
ollama run mistral
```

* **2. Clone the repository**
* **3. Install dependencies**
  
```bash
pip install -r requirements.txt
```
* **4. Usage Workflow**
  * **4.1 Step 1: Build the Vector Database**
    
  ```bash
  python3 ingest.py
  ```
  * **4.2 Step 2: Start the RAG Engine**
    
  ```bash
  python3 main.py
  ```

