import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root, override any existing system env vars
_env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(_env_path, override=True)

# --- Model settings ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-oss-120b")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
TEMPERATURE = int(os.getenv("TEMPERATURE", "0"))

# --- Chunking parameters ---
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "5"))

# --- Folder configuration ---
DATA_FOLDER = os.getenv("DATA_FOLDER", "./data")
OUTPUT_FOLDER = os.getenv("OUTPUT_FOLDER", "./outputs")
FAISS_INDEX_PATH = "./faiss_index"
