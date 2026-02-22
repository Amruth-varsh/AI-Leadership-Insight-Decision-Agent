import logging
import os
import shutil
import sys
import time

from config import DATA_FOLDER, OUTPUT_FOLDER, FAISS_INDEX_PATH
from src.ingest import load_documents
from src.embeddings import build_vector_store, load_vector_store
from src.agent import answer_question

log_level = os.environ.get("LOG_LEVEL", "WARNING").upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.WARNING),
    format="%(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def save_answer(question, answer_text):
    """Save the Q&A pair to a timestamped file in the outputs folder."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(OUTPUT_FOLDER, f"answer_{timestamp}.txt")
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"Q: {question}\n\nA: {answer_text}\n")
    except Exception as e:
        logger.warning(f"Failed to save answer: {e}")


def main():
    os.makedirs(DATA_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    if "--rebuild" in sys.argv and os.path.isdir(FAISS_INDEX_PATH):
        shutil.rmtree(FAISS_INDEX_PATH)
        print("Cleared existing index. Rebuilding...")

    # Load or build vector store
    vector_store = None
    if os.path.isdir(FAISS_INDEX_PATH) and os.listdir(FAISS_INDEX_PATH):
        print("Loading existing index...")
        vector_store = load_vector_store()

    if vector_store is None:
        print("Building index from documents...")
        chunks = load_documents(DATA_FOLDER)
        if not chunks:
            print("No documents found. Place PDFs in the data/ folder and restart.")
            return
        vector_store = build_vector_store(chunks)

    print("\nReady for questions. Type 'exit' to quit.\n")
    while True:
        try:
            question = input("Question: ")
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if question.strip().lower() in ("exit", "quit"):
            print("Goodbye!")
            break
        if not question.strip():
            continue

        answer = answer_question(question, vector_store)
        print(f"\n{answer}\n")
        save_answer(question, answer)


if __name__ == "__main__":
    main()
