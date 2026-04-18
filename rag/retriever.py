"""Simple query interface for the maintenance docs vector DB.

Usage (from your agent code):
    from rag.retriever import get_retriever
    retriever = get_retriever()
    docs = retriever.invoke("worn brakes on old vehicle")
    for d in docs:
        print(d.metadata["source"], "->", d.page_content[:200])
"""
import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"

from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

CHROMA_DIR  = "chroma_db"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _load_store():
    """Load the Chroma store once per process (cached)."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_name="maintenance_guidelines",
    )


def get_retriever(k: int = 4):
    """Return a retriever that fetches the top-k most relevant chunks."""
    store = _load_store()
    return store.as_retriever(search_kwargs={"k": k})


def search(query: str, k: int = 4):
    """Convenience one-shot search (list of Documents)."""
    return get_retriever(k=k).invoke(query)


# Quick test if run directly: python rag/retriever.py
if __name__ == "__main__":
    test_queries = [
        "brakes worn out on an old vehicle",
        "weak battery in a 7-year-old car",
        "multiple reported issues what to do",
        "tire replacement schedule for trucks",
    ]
    for q in test_queries:
        print(f"\n=== Query: {q} ===")
        hits = search(q, k=3)
        for i, d in enumerate(hits, 1):
            snippet = d.page_content.replace("\n", " ")[:160]
            print(f"  [{i}] {d.metadata['source']}")
            print(f"      {snippet}...")