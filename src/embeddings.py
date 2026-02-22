import logging
import os

from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_huggingface import HuggingFaceEmbeddings
from config import EMBEDDING_MODEL, FAISS_INDEX_PATH

logger = logging.getLogger(__name__)


def _get_embedding_instance():
    """Return a HuggingFace embedding model instance."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def build_vector_store(chunks):
    """Build a FAISS vector store from document chunks and persist to disk."""
    if not chunks:
        logger.warning("No chunks provided — cannot build vector store.")
        return None

    print(f"Building embeddings for {len(chunks)} chunks (this may take a moment)...")
    embed = _get_embedding_instance()

    try:
        vector_store = FAISS.from_documents(
            chunks, embed,
            distance_strategy=DistanceStrategy.COSINE,
        )
        vector_store.save_local(FAISS_INDEX_PATH)
        print(f"Vector store saved to {FAISS_INDEX_PATH}")
        return vector_store
    except Exception as e:
        logger.error(f"Error building vector store: {e}")
        raise


def load_vector_store():
    """Load an existing FAISS vector store from disk."""
    if not os.path.isdir(FAISS_INDEX_PATH):
        logger.debug(f"FAISS index not found at {FAISS_INDEX_PATH}")
        return None

    try:
        embed = _get_embedding_instance()
        vector_store = FAISS.load_local(
            FAISS_INDEX_PATH, embed,
            allow_dangerous_deserialization=True,
            distance_strategy=DistanceStrategy.COSINE,
        )
        logger.debug(f"Loaded vector store from {FAISS_INDEX_PATH}")
        return vector_store
    except Exception as e:
        logger.error(f"Error loading vector store: {e}")
        return None
