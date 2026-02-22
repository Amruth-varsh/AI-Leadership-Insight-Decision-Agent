# Adobe AI Leadership Analyst

A Retrieval-Augmented Generation (RAG) agent that indexes Adobe's official financial documents (10-K, 10-Q, Earnings Releases) and answers leadership questions with grounded, citation-backed responses.

## Features

- **PDF ingestion** — recursive chunking with configurable size and overlap
- **FAISS vector store** — cosine similarity search over embedded document chunks
- **HuggingFace embeddings** — `all-MiniLM-L6-v2` runs fully offline after first download
- **Groq LLM** — fast inference via `openai/gpt-oss-120b` (configurable)
- **Strict RAG prompt** — answers sourced exclusively from documents, never outside knowledge
- **Auto-save** — every Q&A pair saved to `outputs/` with timestamps

## Tech Stack

The project leverages a modern AI stack focused on high-speed inference, local embedding generation, and robust document orchestration.

### 🧠 Core Intelligence & Orchestration
| Component | Technology | Role |
|---|---|---|
| **LLM** | [Groq API](https://groq.com/) | Real-time inference using models like `llama-3.3-70b-versatile`. |
| **Framework** | [LangChain](https://www.langchain.com/) | Orchestrates the RAG pipeline, prompt templates, and LLM interactions. |


### 🔍 Retrieval & Vector Management
| Component | Technology | Role |
|---|---|---|
| **Vector Database** | [FAISS](https://github.com/facebookresearch/faiss) | Efficient similarity search for dense vectors (Cosine Similarity). |
| **Embeddings** | [Hugging Face](https://huggingface.co/sentence-transformers) | Local generation of embeddings using `all-MiniLM-L6-v2`. |
| **Retrieval Strategy** | Dense Retrieval | Top-K similarity search to fetch the most relevant document chunks. |

### 🛠️ Data Processing & Infrastructure
| Component | Technology | Role |
|---|---|---|
| **PDF Parsing** | [PyPDF](https://pypdf.readthedocs.io/) | Handles recursive extraction and cleaning of text from PDF documents. |
| **Chunking** | Recursive Character Splitter | Splits documents into meaningful chunks with configurable overlap. |
| **Environment** | Python 3.10+ | Core runtime environment. |
| **Config** | `python-dotenv` | Securely manages API keys and environment variables via `.env`. |

## Project Structure

```
├── config.py           # All settings loaded from .env
├── main.py             # CLI entry point
├── requirements.txt    # Python dependencies
├── .env                # API keys and configuration 
├── data/               # Place your PDF documents here
├── faiss_index/        # Auto-generated FAISS vector store
├── outputs/            # Saved Q&A pairs (timestamped)
└── src/
    ├── __init__.py
    ├── ingest.py       # PDF loading and chunking
    ├── embeddings.py   # HuggingFace embeddings + FAISS build/load
    └── agent.py        # Prompt template + Groq LLM call
```

## Setup

### 1. Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure `.env`

#### Steps to Get Your Groq API Key

1. Go to [console.groq.com](https://console.groq.com)
2. Click **Sign Up** if you don't have an account (sign up with Google or Email)
3. After logging in, click **API Keys** in the left sidebar
4. Click **Create API Key** (top right)
5. Give it a name like `adobe-project` and click **Submit**
6. Copy your key immediately — it starts with `gsk_...` and **will not be shown again**

#### How to Enable `llama-3.3-70b-versatile` in Groq Console

1. Log in to [console.groq.com](https://console.groq.com)
2. Click **Settings** in the left sidebar
3. Click **Limits**
4. Scroll down to the **Allowed Models** section
5. Click the model selection dropdown/search box
6. Type `llama-3.3-70b-versatile` and select it from the list
7. It will appear as a tag/chip in the allowed models box
8. Click **Save**


```env
GROQ_API_KEY=your_groq_api_key_here
MODEL_NAME=llama-3.3-70b-versatile
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
TEMPERATURE=0
TOP_K_RESULTS=10
CHUNK_SIZE=1200
CHUNK_OVERLAP=200
DATA_FOLDER=./data
OUTPUT_FOLDER=./outputs
```

### 4. Add documents

Place PDF files (10-K, earnings releases, etc.) in the `data/` folder.

## Usage

```bash
python main.py             # Start the agent (builds index on first run)
python main.py --rebuild   # Force re-index all documents from scratch
```

Type your question at the prompt. Type `exit` or `quit` to stop.

### Example Questions

- "How did Adobe's net income perform in fiscal year 2025 compared to 2024?"
- "What is Adobe's revenue breakdown by segment for FY2024?"
- "How did the Publishing and Advertising segment perform compared to the other segments?"
- "What is Adobe's FY2026 earnings per share guidance?"
- "How did Digital Media perform compared to last year?"




## Notes

- Do **not** commit `.env` — add it to `.gitignore`
- The FAISS index is built automatically on first run; use `--rebuild` after adding new documents
- The embedding model is cached locally after the first download — no internet required after that
- Set `LOG_LEVEL=DEBUG` for verbose output: `set LOG_LEVEL=DEBUG && python main.py`