"""Builds a Chroma vector database from the maintenance_docs/ folder.

Run this ONCE whenever the maintenance docs change:
    python rag/build_index.py
"""
import os
# Disable Chroma's anonymous telemetry (silences harmless telemetry errors)
os.environ["ANONYMIZED_TELEMETRY"] = "False"

from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Config 
DOCS_DIR    = Path("maintenance_docs")
CHROMA_DIR  = "chroma_db"
CHUNK_SIZE  = 600        # characters per chunk (~150 words)
CHUNK_OVER  = 80         # overlap so context isn't lost at chunk boundaries
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # small, fast, CPU-friendly


def main():
    # 1. Load every .md file in maintenance_docs/
    docs = []
    md_files = sorted(DOCS_DIR.glob("*.md"))
    if not md_files:
        raise FileNotFoundError(
            f"No .md files found in {DOCS_DIR}/. "
            "Put the 8 maintenance doc files there first."
        )

    for md_path in md_files:
        loaded = TextLoader(str(md_path), encoding="utf-8").load()
        for d in loaded:
            d.metadata["source"] = md_path.name
        docs.extend(loaded)

    print(f"Loaded {len(docs)} documents from {DOCS_DIR}/")

    # 2. Split long documents into smaller chunks for better retrieval
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVER,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " "],
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks")

    # 3. Embed + store in Chroma
    print(f"Embedding with {EMBED_MODEL} ...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name="maintenance_guidelines",
    )
    print(f"Chroma DB saved to ./{CHROMA_DIR}/")
    print("Done! You can now use rag/retriever.py to query it.")


if __name__ == "__main__":
    main()