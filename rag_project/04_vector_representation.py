"""
04_vector_representation.py
============================
Section 9-11 من النوت بوك الأصلي: بناء التمثيلات الثلاثة المستخدمة في
الاسترجاع: TF-IDF، BM25 (tokens)، والـ embeddings الدلالية متعددة اللغات.
يقرأ `chunks.parquet` (ناتج 03_chunking.py) وينتج الملفات اللي هتتحمّل
في 05_create_chroma_store.py و 06_retrieve_context.py.

تشغيل:
    python 04_vector_representation.py --chunks rag_export/chunks.parquet --out-dir rag_export
"""

import argparse
import importlib.util
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


def _load_module(filename, alias):
    """Load a sibling script as a module (filenames start with digits, so a
    plain `import` isn't possible)."""
    path = Path(__file__).parent / filename
    spec = importlib.util.spec_from_file_location(alias, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

preprocessing = _load_module("02_preprocessing.py", "preprocessing")


# ==========================================================
# Section 2 (reused) — lightweight language detection
# ==========================================================

ARABIC_PATTERN = re.compile(r"[\u0600-\u06FF]")


def detect_language(text, arabic_char_threshold=0.15):
    if not isinstance(text, str):
        return "unknown"
    text = text.strip()
    if not text:
        return "unknown"
    letters = re.findall(r"[^\W\d_]", text, re.UNICODE)
    if not letters:
        return "unknown"
    arabic_letters = ARABIC_PATTERN.findall(text)
    ratio = len(arabic_letters) / len(letters)
    return "ar" if ratio >= arabic_char_threshold else "en"


def normalize_lexical_text(text):
    language = detect_language(text)
    return preprocessing.preprocess_text_bilingual(text, language)


def simple_tokenize(text):
    text = normalize_lexical_text(text)
    return re.findall(r"[a-z0-9]+|[\u0600-\u06FF]+", text.lower())


# ==========================================================
# Section 9 — TF-IDF
# ==========================================================

def build_tfidf(chunks_df):
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(chunks_df["search_text"].map(normalize_lexical_text))
    return vectorizer, matrix


# ==========================================================
# Section 10 — BM25 tokens
# ==========================================================

def build_bm25_tokens(chunks_df):
    return [simple_tokenize(text) for text in chunks_df["search_text"]]


# ==========================================================
# Section 11 — Multilingual embeddings
# ==========================================================

def build_embeddings(chunks_df, model_name=EMBEDDING_MODEL_NAME):
    model = SentenceTransformer(model_name)
    chunk_texts = chunks_df["search_text"].tolist()
    embeddings = model.encode(
        chunk_texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return embeddings


def main():
    parser = argparse.ArgumentParser(description="Build TF-IDF / BM25 / embeddings artifacts")
    parser.add_argument("--chunks", default="rag_export/chunks.parquet")
    parser.add_argument("--out-dir", default="rag_export")
    args = parser.parse_args()

    chunks_df = pd.read_parquet(args.chunks)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Building TF-IDF...")
    tfidf_vectorizer, tfidf_matrix = build_tfidf(chunks_df)
    with open(out_dir / "tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(tfidf_vectorizer, f)
    sp.save_npz(out_dir / "tfidf_matrix.npz", tfidf_matrix)
    print("TF-IDF matrix shape:", tfidf_matrix.shape)

    print("\nBuilding BM25 tokens...")
    tokenized_chunks = build_bm25_tokens(chunks_df)
    with open(out_dir / "bm25_tokens.pkl", "wb") as f:
        pickle.dump(tokenized_chunks, f)
    print("Tokenized chunks:", len(tokenized_chunks))

    print("\nBuilding embeddings (this can take a while)...")
    chunk_embeddings = build_embeddings(chunks_df)
    np.save(out_dir / "chunk_embeddings.npy", chunk_embeddings)
    with open(out_dir / "embedding_model_name.txt", "w") as f:
        f.write(EMBEDDING_MODEL_NAME)
    print("Embeddings shape:", chunk_embeddings.shape)

    print(f"\n✅ Saved all vector artifacts -> {out_dir}")


if __name__ == "__main__":
    main()
