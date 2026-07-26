"""
05_create_chroma_store.py
===========================
Stage 5 of the RAG pipeline.

Builds a persistent ChromaDB collection from the chunks + their
pre-computed embeddings. This collection is what gets committed to
GitHub and loaded by the deployed Streamlit app (06_retrieve_context.py),
so the app never needs to re-read PDFs or recompute embeddings at runtime.

Input:
    data/chunks.parquet            (from 03_chunking.py)
    data/chunk_embeddings.npy      (from 04_vector_representation.py)
Output:
    chroma_db/   (persistent ChromaDB directory — commit this to GitHub)

Run standalone:
    python 05_create_chroma_store.py --chunks data/chunks.parquet \
        --embeddings data/chunk_embeddings.npy --persist_dir chroma_db
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import chromadb

COLLECTION_NAME = "digital_society_chunks"


def sanitize_metadata_value(value):
    """Chroma metadata only accepts str / int / float / bool (no None)."""
    if value is None:
        return "unknown"
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def build_chroma_store(chunks_df, embeddings, persist_dir="chroma_db",
                        collection_name=COLLECTION_NAME):
    """Create (or overwrite) a persistent Chroma collection from chunks + embeddings."""
    client = chromadb.PersistentClient(path=persist_dir)

    # Start clean so re-running this script doesn't duplicate/stale entries
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    ids = [str(cid) for cid in chunks_df["chunk_id"].tolist()]
    documents = chunks_df["chunk_text"].tolist()

    metadatas = []
    for _, row in chunks_df.iterrows():
        metadatas.append({
            "document_id": sanitize_metadata_value(row["document_id"]),
            "title": sanitize_metadata_value(row["title"]),
            "authors": sanitize_metadata_value(row["authors"]),
            "publication_year": sanitize_metadata_value(row["publication_year"]),
            "source_type": sanitize_metadata_value(row["source_type"]),
            "language": sanitize_metadata_value(row["language"]),
            "is_current": sanitize_metadata_value(row["is_current"]),
        })

    # Chroma has an internal batch-size limit; add in chunks of 500 to be safe
    batch_size = 500
    for start in range(0, len(ids), batch_size):
        end = start + batch_size
        collection.add(
            ids=ids[start:end],
            embeddings=embeddings[start:end].tolist(),
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )

    return collection


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 5: Build the persistent Chroma vector store.")
    parser.add_argument("--chunks", default="data/chunks.parquet")
    parser.add_argument("--embeddings", default="data/chunk_embeddings.npy")
    parser.add_argument("--persist_dir", default="chroma_db")
    args = parser.parse_args()

    chunks_df = pd.read_parquet(args.chunks)
    embeddings = np.load(args.embeddings)

    assert len(chunks_df) == len(embeddings), (
        f"chunks ({len(chunks_df)}) and embeddings ({len(embeddings)}) row counts do not match."
    )

    collection = build_chroma_store(chunks_df, embeddings, persist_dir=args.persist_dir)

    print(f"✅ Chroma collection '{COLLECTION_NAME}' created at {args.persist_dir}")
    print(f"   Total chunks stored: {collection.count()}")
