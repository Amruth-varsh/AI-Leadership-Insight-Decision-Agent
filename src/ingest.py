import logging
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)


def load_documents(folder_path):
    """Load PDF documents from a folder, split into chunks, and return as
    a list of LangChain Document objects with source metadata."""

    if not os.path.isdir(folder_path):
        logger.warning(f"Data folder not found: {folder_path}")
        return []

    pdf_files = sorted(f for f in os.listdir(folder_path) if f.lower().endswith(".pdf"))

    if not pdf_files:
        logger.warning(f"No PDF files found in {folder_path}")
        return []

    print(f"Found {len(pdf_files)} PDF(s) in {folder_path}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    all_chunks = []

    for pdf in pdf_files:
        path = os.path.join(folder_path, pdf)
        try:
            print(f"  Loading: {pdf}")
            loader = PyPDFLoader(path)
            raw_docs = loader.load()
            # split_documents preserves metadata (source, page) automatically
            chunks = splitter.split_documents(raw_docs)
            all_chunks.extend(chunks)
            logger.debug(f"  {pdf}: {len(raw_docs)} pages -> {len(chunks)} chunks")
        except Exception as e:
            msg = str(e)
            logger.warning(f"Error loading {pdf}: {msg}")
            if "cryptography" in msg.lower():
                print(f"  ⚠ {pdf} appears encrypted — install 'cryptography' package")
            continue

    print(f"Total: {len(all_chunks)} chunks from {len(pdf_files)} document(s)")
    return all_chunks
