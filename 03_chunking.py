"""
03_chunking.py
===============
Stage 3 of the RAG pipeline.

Splits each document's full text into overlapping word-based chunks so
that retrieval can return focused passages instead of whole papers.

Input:  data/documents.parquet   (from 01_documents.py)
Output: data/chunks.parquet

Run standalone:
    python 03_chunking.py --in data/documents.parquet --out data/chunks.parquet
"""

import argparse
from pathlib import Path
"""شغلي هذا الملف على لابتوبك للتأكد أن كل شيء جاهز قبل الرفع"""
from pathlib import Path
import sys

checks = [
    (Path("data/chunks.parquet"), "data/chunks.parquet - أهم ملف (3-10 MB) - لازم يترفع"),
    (Path("data/embedding_model_name.txt"), "data/embedding_model_name.txt - صغير - لازم يترفع"),
    (Path("data/chunk_embeddings.npy"), "data/chunk_embeddings.npy - اختياري للتسريع"),
    (Path("data/pdfs/papers"), "data/pdfs/papers/ - فولدر الأوراق (11 ملف PDF) - لا ترفعيه"),
    (Path("data/pdfs/book"), "data/pdfs/book/ - فولدر الكتاب - لا ترفعيه"),
    (Path("chroma_db"), "chroma_db/ - لا ترفعيه، التطبيق يبنيه لوحده"),
    (Path("streamlit_app.py"), "streamlit_app.py - لازم يكون النسخة المصححة FIXED"),
]

print("=== فحص الملفات المحلية ===\n")
ok_to_upload = []
for p, desc in checks:
    if p.exists():
        if p.is_file():
            size_mb = p.stat().st_size / 1024 / 1024
            print(f"✅ موجود: {desc} ({size_mb:.2f} MB)")
            if "chunks.parquet" in str(p) or "embedding_model_name" in str(p):
                ok_to_upload.append(p)
        else:
            count = len(list(p.glob("*.pdf"))) if "pdfs" in str(p) else len(list(p.rglob("*")))
            print(f"✅ موجود: {desc} ({count} ملف)")
    else:
        print(f"❌ غير موجود: {desc}")

print("\n=== الخلاصة ===")
if Path("data/chunks.parquet").exists():
    print("🎉 ممتاز! عندك chunks.parquet - تقدري ترفعيه على GitHub مباشرة بدون Colab")
    print("اذهبي إلى: https://github.com/SaraMamd0uh/AI_Literature_Review_Assistant_RAG_FINAL")
    print("Add file -> Upload files -> اسحبي data/chunks.parquet")
else:
    print("⚠️ لا يوجد chunks.parquet - شغلي:")
    print("python 01_documents.py")
    print("python 02_preprocessing.py")
    print("python 03_chunking.py")
    print("python 04_vector_representation.py")

import pandas as pd


def chunk_text(text, chunk_size=120, overlap=30):
    """Split text into overlapping word-based chunks."""
    words = text.split()
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap cannot be negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start = end - overlap
    return chunks


def build_chunks_dataframe(documents_df, chunk_size=120, overlap=30):
    """Build a chunk-level DataFrame, carrying full metadata from the source document."""
    rows = []
    chunk_id = 0

    for _, doc in documents_df.iterrows():
        text_chunks = chunk_text(doc["text"], chunk_size=chunk_size, overlap=overlap)

        for position, chunk in enumerate(text_chunks):
            # Title + authors are folded into search_text to boost lexical/semantic matching
            search_text = f"{doc['title']} {doc['authors']} {chunk}"

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
                "chunk_text": chunk,
                "search_text": search_text,
            })
            chunk_id += 1

    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 3: Chunk documents into overlapping passages.")
    parser.add_argument("--in", dest="in_path", default="data/documents.parquet")
    parser.add_argument("--out", default="data/chunks.parquet")
    parser.add_argument("--chunk_size", type=int, default=120)
    parser.add_argument("--overlap", type=int, default=30)
    args = parser.parse_args()

    documents_df = pd.read_parquet(args.in_path)
    chunks_df = build_chunks_dataframe(documents_df, chunk_size=args.chunk_size, overlap=args.overlap)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    chunks_df.to_parquet(args.out, index=False)

    print(f"✅ Saved {len(chunks_df)} chunks to {args.out}")
    print(chunks_df.groupby("document_id").size().rename("num_chunks"))
