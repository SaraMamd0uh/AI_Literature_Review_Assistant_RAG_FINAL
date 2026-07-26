"""
05_create_chroma_store.py
==========================
مش موجود في النوت بوك الأصلي (اللي كان بيخزن الـ embeddings كملف .npy خام
ويعمل cosine_similarity يدويًا). السكريبت ده بيبني Chroma persistent
collection من chunks.parquet + chunk_embeddings.npy الجاهزين من
04_vector_representation.py، فتقدري تستخدمي Chroma كـ vector store حقيقي
بدل الاعتماد على numpy مباشرة في التطبيق.

الـ embeddings متحسوبة مسبقًا بموديل SentenceTransformers نفسه، فبنبعتها
لـ Chroma جاهزة (مش بنسيبها تحسب embeddings تانية بموديلها الافتراضي).

تشغيل:
    python 05_create_chroma_store.py --chunks rag_export/chunks.parquet \
        --embeddings rag_export/chunk_embeddings.npy \
        --persist-dir rag_export/chroma_store
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import chromadb

COLLECTION_NAME = "literature_review_chunks"


def _clean_metadata_value(value):
    """Chroma metadata values must be str / int / float / bool (no None/NaN)."""
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return value


def build_chroma_store(chunks_df, embeddings, persist_dir):
    client = chromadb.PersistentClient(path=str(persist_dir))

    # Recreate the collection fresh every time this script runs.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    ids = [str(cid) for cid in chunks_df["chunk_id"]]

    documents = chunks_df["chunk_text"].tolist()

    metadatas = [
        {
            "document_id": _clean_metadata_value(row["document_id"]),
            "title": _clean_metadata_value(row["title"]),
            "authors": _clean_metadata_value(row["authors"]),
            "publication_year": _clean_metadata_value(row["publication_year"]),
            "source_type": _clean_metadata_value(row["source_type"]),
            "language": _clean_metadata_value(row["language"]),
            "chunk_position": _clean_metadata_value(row["chunk_position"]),
        }
        for _, row in chunks_df.iterrows()
    ]

    # Chroma's add() has a practical batch-size limit; chunk it defensively.
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


def main():
    parser = argparse.ArgumentParser(description="Build a persistent Chroma store from chunks + embeddings")
    parser.add_argument("--chunks", default="rag_export/chunks.parquet")
    parser.add_argument("--embeddings", default="rag_export/chunk_embeddings.npy")
    parser.add_argument("--persist-dir", default="rag_export/chroma_store")
    args = parser.parse_args()

    chunks_df = pd.read_parquet(args.chunks)
    embeddings = np.load(args.embeddings)

    assert len(chunks_df) == len(embeddings), (
        f"chunks.parquet has {len(chunks_df)} rows but embeddings has "
        f"{len(embeddings)} rows — rerun 04_vector_representation.py."
    )

    collection = build_chroma_store(chunks_df, embeddings, args.persist_dir)

    print(f"✅ Chroma collection '{COLLECTION_NAME}' created with {collection.count()} chunks")
    print(f"   Persisted at -> {args.persist_dir}")
    print("   Commit this folder to your GitHub repo alongside chunks.parquet, "
          "bm25_tokens.pkl, and tfidf_*.pkl so the Streamlit app can load it directly.")


if __name__ == "__main__":
    main()
