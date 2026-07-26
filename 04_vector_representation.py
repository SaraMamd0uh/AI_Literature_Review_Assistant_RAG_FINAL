"""
04_vector_representation.py
=============================
Stage 4 of the RAG pipeline.

Computes dense vector embeddings for every chunk using a MULTILINGUAL
sentence-transformers model (needed because our corpus mixes Arabic and
English documents, and queries in either language must be able to match
chunks in the other language).

Input:  data/chunks.parquet          (from 03_chunking.py)
Output: data/chunk_embeddings.npy    (float32 array, same row order as chunks.parquet)

Run standalone:
    python 04_vector_representation.py --in data/chunks.parquet --out data/chunk_embeddings.npy
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


def load_embedding_model(model_name=EMBEDDING_MODEL_NAME):
    return SentenceTransformer(model_name)


def compute_embeddings(chunks_df, model=None, model_name=EMBEDDING_MODEL_NAME, batch_size=32):
    """Encode every chunk's search_text into a normalized embedding vector."""
    if model is None:
        model = load_embedding_model(model_name)

    texts = chunks_df["search_text"].tolist()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return embeddings.astype("float32")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 4: Compute chunk embeddings.")
    parser.add_argument("--in", dest="in_path", default="data/chunks.parquet")
    parser.add_argument("--out", default="data/chunk_embeddings.npy")
    parser.add_argument("--model_name", default=EMBEDDING_MODEL_NAME)
    args = parser.parse_args()

    chunks_df = pd.read_parquet(args.in_path)
    model = load_embedding_model(args.model_name)
    embeddings = compute_embeddings(chunks_df, model=model)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, embeddings)

    # Save the model name alongside the embeddings so later stages know
    # exactly which model to reload for queries.
    with open(Path(args.out).parent / "embedding_model_name.txt", "w") as f:
        f.write(args.model_name)

    print(f"✅ Saved embeddings with shape {embeddings.shape} to {args.out}")
