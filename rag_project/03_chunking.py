"""
03_chunking.py
===============
Section 6 من النوت بوك الأصلي: تقسيم نص كل مستند إلى chunks متداخلة
(overlapping) على مستوى الكلمات، مع تجاهل الأجزاء القصيرة جدًا وأقسام
الـ References. يقرأ `documents.parquet` (ناتج 01_documents.py) وينتج
`chunks.parquet`.

تشغيل:
    python 03_chunking.py --documents rag_export/documents.parquet --out rag_export/chunks.parquet
"""

import argparse
from pathlib import Path

import pandas as pd

REFERENCE_HEADERS = [
    "references",
    "reference",
    "bibliography",
    "works cited",
    "acknowledgment",
    "acknowledgements",
]


def chunk_text(text, chunk_size=120, overlap=30):
    """
    Split text into overlapping chunks.

    Parameters
    ----------
    text : str
        Input document text.
    chunk_size : int
        Number of words in each chunk.
    overlap : int
        Number of overlapping words.

    Returns
    -------
    list
        List of text chunks.
    """
    if not isinstance(text, str) or not text.strip():
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap cannot be negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(words):
            break
        start = end - overlap

    return chunks


def build_chunks_dataframe(documents_df, chunk_size=120, overlap=30):
    rows = []
    chunk_id = 0

    for _, doc in documents_df.iterrows():
        text_chunks = chunk_text(doc["text"], chunk_size=chunk_size, overlap=overlap)

        for position, chunk in enumerate(text_chunks):
            chunk = chunk.strip()
            lower_chunk = chunk.lower()

            word_count = len(chunk.split())

            # Skip very small chunks
            if word_count < 40:
                continue

            # Skip References section
            if any(lower_chunk.startswith(h) for h in REFERENCE_HEADERS):
                continue

            search_text = f"{doc['title']} {doc['authors']} {doc['publication_year']} {chunk}"

            rows.append({
                "chunk_id": chunk_id,
                "document_id": doc["document_id"],
                "title": doc["title"],
                "authors": doc["authors"],
                "publication_year": doc["publication_year"],
                "source_type": doc["source_type"],
                "language": doc["language"],
                "is_current": doc["is_current"],
                "chunk_position": position,
                "chunk_word_count": word_count,
                "chunk_char_count": len(chunk),
                "chunk_text": chunk,
                "search_text": search_text,
            })

            chunk_id += 1

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Build chunks.parquet from documents.parquet")
    parser.add_argument("--documents", default="rag_export/documents.parquet")
    parser.add_argument("--out", default="rag_export/chunks.parquet")
    parser.add_argument("--chunk-size", type=int, default=120)
    parser.add_argument("--overlap", type=int, default=30)
    args = parser.parse_args()

    documents_df = pd.read_parquet(args.documents)
    chunks_df = build_chunks_dataframe(documents_df, chunk_size=args.chunk_size, overlap=args.overlap)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    chunks_df.to_parquet(out_path, index=False)

    print("Total chunks:", len(chunks_df))
    print(chunks_df.groupby("document_id").size().rename("num_chunks"))
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
