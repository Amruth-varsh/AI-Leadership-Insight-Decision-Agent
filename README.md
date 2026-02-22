<div align="center">

```
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║    ██╗      ███████╗ █████╗ ██████╗ ███████╗██████╗ ███████╗██╗  ██╗ ║
║    ██║      ██╔════╝██╔══██╗██╔══██╗██╔════╝██╔══██╗██╔════╝██║  ██║ ║
║    ██║      █████╗  ███████║██║  ██║█████╗  ██████╔╝███████╗███████║ ║
║    ██║      ██╔══╝  ██╔══██║██║  ██║██╔══╝  ██╔══██╗╚════██║██╔══██║ ║
║    ███████╗ ███████╗██║  ██║██████╔╝███████╗██║  ██║███████║██║  ██║ ║
║    ╚══════╝ ╚══════╝╚═╝  ╚═╝╚═════╝ ╚══════╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝ ║
║                                                                      ║
║   ◆  A I  L E A D E R S H I P  I N S I G H T  D E C I S I O N  ◆   ║
║                    ◆  A G E N T  ◆                                   ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
```

### *Ask anything. Get answers grounded in real financial documents.*

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-Orchestration-1C3C3C?style=for-the-badge&logo=chainlink&logoColor=white)](https://langchain.com)
[![Groq](https://img.shields.io/badge/Groq-Lightning%20Fast-F55036?style=for-the-badge)](https://groq.com)
[![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-0064A5?style=for-the-badge)](https://github.com/facebookresearch/faiss)
[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Embeddings-FFD21E?style=for-the-badge)](https://huggingface.co)

</div>

---

## ◈ What Is This?

> **A RAG-powered analyst that lives inside financial documents.**

Drop in company documents (e.g.: Annual reports , Quarterly reports , Strategy notes , Operational updates) — and this agent transforms dense PDFs into instant, citation-backed answers. No hallucinations. No guessing. Every response is anchored to the documents you provide.

```
You ask   →  "How did Adobe's net income change in FY2025?"
Agent     →  Scans embedded document chunks in milliseconds
Response  →  Grounded answer + exact source context. Every time.
```

---

## ◈ System Architecture

<div align="center">

![System Architecture](https://raw.githubusercontent.com/Amruth-varsh/AI-Leadership-Insight-Decision-Agent/main/System%20Architecture.png)

</div>

---

## ◈ Tech Stack

| Layer | Tool | Why It's Here |
|---|---|---|
| 🧠 **LLM** | Groq `llama-3.3-70b-versatile` | Blazing inference. Real-time responses. |
| 🔗 **Orchestration** | LangChain | RAG pipeline, prompt templates, LLM wiring |
| 🗄️ **Vector DB** | FAISS | Facebook's gold-standard similarity search |
| 🔡 **Embeddings** | HuggingFace `all-MiniLM-L6-v2` | Runs 100% offline after first download |
| 📄 **PDF Parsing** | PyPDF | Recursive extraction and text cleaning |
| ✂️ **Chunking** | Recursive Character Splitter | Configurable size + overlap |
| 🔐 **Config** | python-dotenv | Secure API key management via `.env` |

---

## ◈ Project Structure

```
AI-Leadership-Insight-Decision-Agent/
│
├── 📄  main.py              ← CLI entry point. Start here.
├── ⚙️  config.py            ← All settings loaded from .env
├── 📋  requirements.txt     ← Python dependencies
├── 🔐  .env                 ← Your API keys
│
├── 🗄️  faiss_index/         ← Auto-generated vector store
├── 💾  outputs/             ← Saved Q&A pairs (timestamped)
│
└── 📁  src/
    ├── 🔧  __init__.py
    ├── 📥  ingest.py        ← PDF loading & chunking logic
    ├── 🔍  embeddings.py    ← FAISS index build/load
    └── 🤖  agent.py         ← Prompt template + Groq LLM call
```

---

## ◈ Setup

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

**Steps to Get Your Groq API Key**

1. Go to [console.groq.com](https://console.groq.com)
2. Click **Sign Up** if you don't have an account (sign up with Google or Email)
3. After logging in, click **API Keys** in the left sidebar
4. Click **Create API Key** (top right)
5. Give it a name like `adobe-project` and click **Submit**
6. Copy your key immediately — it starts with `gsk_...` and **will not be shown again**

**How to Enable `llama-3.3-70b-versatile` in Groq Console**

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

---

## ◈ Usage

```bash
# First run — builds the FAISS index automatically
python main.py

# Force re-index (after adding new documents)
python main.py --rebuild
```

Type your question at the prompt. Type `exit` or `quit` to stop.

---

## ◈ Example Questions to Try

```
◆  "How did Adobe's net income perform in FY2025 vs FY2024?"
◆  "What is Adobe's revenue breakdown by segment for FY2024?"
◆  "How did the Publishing and Advertising segment compare to others?"
◆  "What is Adobe's FY2026 earnings per share guidance?"
◆  "How did Digital Media ARR grow year-over-year?"
◆  "What risks did Adobe highlight in their most recent 10-K?"
```

---

<div align="center">

```
Built with  ⚡ Groq  ·  🦜 LangChain  ·  🤗 HuggingFace  ·  🔍 FAISS
```

*Answers grounded in documents. Never outside them.*

</div>
