<div align="center">

```
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║    ░█████╗░██████╗░░█████╗░██████╗░███████╗                 ║
║    ██╔══██╗██╔══██╗██╔══██╗██╔══██╗██╔════╝                 ║
║    ███████║██║  ██║██║  ██║██████╦╝█████╗                   ║
║    ██╔══██║██║  ██║██║  ██║██╔══██╗██╔══╝                   ║
║    ██║  ██║██████╔╝╚█████╔╝██████╦╝███████╗                 ║
║    ╚═╝  ╚═╝╚═════╝  ╚════╝ ╚═════╝ ╚══════╝                 ║
║                                                              ║
║         ◆  A I   L E A D E R S H I P   A N A L Y S T  ◆     ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
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

> **A RAG-powered analyst that lives inside Adobe's financials.**

Drop in any 10-K, 10-Q, or Earnings Release — and this agent transforms dense PDFs into instant, citation-backed answers. No hallucinations. No guessing. Every response is anchored to the documents you provide.

```
You ask   →  "How did Adobe's net income change in FY2025?"
Agent     →  Scans embedded document chunks in milliseconds
Response  →  Grounded answer + exact source context. Every time.
```

---

## ◈ How It Works

'''
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
adobe-ai-analyst/
│
├── 📄  main.py              ← CLI entry point. Start here.
├── ⚙️  config.py            ← All settings loaded from .env
├── 📋  requirements.txt     ← Python dependencies
├── 🔐  .env                 ← Your API keys (never commit this)
│
├── 📁  data/                ← Drop your PDFs in here
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

## ◈ Setup in 4 Steps

### `1` — Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### `2` — Install dependencies

```bash
pip install -r requirements.txt
```

### `3` — Configure your `.env`

Create a `.env` file in the root directory:

```env
# ── LLM ────────────────────────────────────────────
GROQ_API_KEY=gsk_your_key_here
MODEL_NAME=llama-3.3-70b-versatile

# ── Embeddings ─────────────────────────────────────
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# ── Retrieval ──────────────────────────────────────
TOP_K_RESULTS=10
TEMPERATURE=0

# ── Chunking ───────────────────────────────────────
CHUNK_SIZE=1200
CHUNK_OVERLAP=200

# ── Paths ──────────────────────────────────────────
DATA_FOLDER=./data
OUTPUT_FOLDER=./outputs
```

> **Getting your Groq API Key**
> 1. Go to [console.groq.com](https://console.groq.com) → Sign Up
> 2. Sidebar → **API Keys** → **Create API Key**
> 3. Name it (e.g. `adobe-project`) → Copy immediately
> 4. Key starts with `gsk_...` — it's shown **only once**

> **Enabling `llama-3.3-70b-versatile`**
> Console → **Settings** → **Limits** → **Allowed Models** → search & select the model → **Save**

### `4` — Add your PDFs

```bash
# Place your financial documents here:
data/
  ├── adobe-10k-2024.pdf
  ├── adobe-10q-q3-2024.pdf
  └── adobe-earnings-q4-2024.pdf
```

---

## ◈ Usage

```bash
# First run — builds the FAISS index automatically
python main.py

# Force re-index (after adding new documents)
python main.py --rebuild

# Verbose debug output
set LOG_LEVEL=DEBUG && python main.py     # Windows
LOG_LEVEL=DEBUG python main.py           # Mac/Linux
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

## ◈ Important Notes

```
⚠️  Never commit .env — add it to .gitignore
⚠️  FAISS index auto-builds on first run
⚠️  Embedding model caches locally — no internet needed after first download
⚠️  Use --rebuild whenever you add new PDFs to data/
```

---

<div align="center">

```
Built with  ⚡ Groq  ·  🦜 LangChain  ·  🤗 HuggingFace  ·  🔍 FAISS
```

*Answers grounded in documents. Never outside them.*

</div>
