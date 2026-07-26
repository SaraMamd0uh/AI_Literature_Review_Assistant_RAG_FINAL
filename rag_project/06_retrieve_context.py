"""
06_retrieve_context.py
=======================
Section 8-12 و 14 من النوت بوك الأصلي، مُعاد بناؤها بحيث تشتغل من الملفات
المُصدَّرة (rag_export/) بدل ما تعيد التدريب/الحساب من الصفر. ده الموديول
اللي بيستورده streamlit_app.py مباشرة وقت الطلب.

الفرق عن النوت بوك: الاسترجاع الدلالي (semantic) بقى عن طريق Chroma
(05_create_chroma_store.py) بدل matrix numpy خام + cosine_similarity يدوي.

الاستخدام كـ module:
    from importlib helper في streamlit_app.py:
        artifacts = load_artifacts("rag_export")
        package = build_context_package("سؤالي هنا", artifacts)
"""

import importlib.util
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def _load_module(filename, alias):
    path = Path(__file__).parent / filename
    spec = importlib.util.spec_from_file_location(alias, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

preprocessing = _load_module("02_preprocessing.py", "preprocessing")

CHROMA_COLLECTION_NAME = "literature_review_chunks"


# ==========================================================
# Lightweight language detection (same as 01_documents.py / 04_vector_representation.py)
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
# Loading pre-built artifacts
# ==========================================================

def load_artifacts(export_dir="rag_export"):
    """
    Load everything needed for retrieval from the exported folder:
    chunks.parquet, tfidf_vectorizer.pkl, tfidf_matrix.npz, bm25_tokens.pkl,
    chroma_store/, and the embedding model name.

    Call this once and reuse the returned dict (wrap with st.cache_resource
    in the Streamlit app so it isn't reloaded on every rerun).
    """
    export_dir = Path(export_dir)

    chunks_df = pd.read_parquet(export_dir / "chunks.parquet")

    with open(export_dir / "tfidf_vectorizer.pkl", "rb") as f:
        tfidf_vectorizer = pickle.load(f)
    tfidf_matrix = sp.load_npz(export_dir / "tfidf_matrix.npz")

    with open(export_dir / "bm25_tokens.pkl", "rb") as f:
        tokenized_chunks = pickle.load(f)
    bm25 = BM25Okapi(tokenized_chunks)

    embedding_model_name = (export_dir / "embedding_model_name.txt").read_text().strip()
    embedding_model = SentenceTransformer(embedding_model_name)

    chroma_client = chromadb.PersistentClient(path=str(export_dir / "chroma_store"))
    chroma_collection = chroma_client.get_collection(CHROMA_COLLECTION_NAME)

    return {
        "chunks_df": chunks_df,
        "tfidf_vectorizer": tfidf_vectorizer,
        "tfidf_matrix": tfidf_matrix,
        "bm25": bm25,
        "embedding_model": embedding_model,
        "chroma_collection": chroma_collection,
    }


# ==========================================================
# Section 9 — TF-IDF Retriever
# ==========================================================

def retrieve_top_k_tfidf(query, artifacts, k=3):
    query_vector = artifacts["tfidf_vectorizer"].transform([normalize_lexical_text(query)])
    scores = cosine_similarity(query_vector, artifacts["tfidf_matrix"]).flatten()
    ranking = np.argsort(scores)[::-1][:k]
    results = artifacts["chunks_df"].iloc[ranking].copy()
    results["score"] = scores[ranking]
    return results


# ==========================================================
# Section 10 — BM25 Retriever
# ==========================================================

def retrieve_top_k_bm25(query, artifacts, k=3):
    tokenized_query = simple_tokenize(query)
    scores = artifacts["bm25"].get_scores(tokenized_query)
    ranking = np.argsort(scores)[::-1][:k]
    results = artifacts["chunks_df"].iloc[ranking].copy()
    results["score"] = scores[ranking]
    return results


# ==========================================================
# Section 11 — Semantic Retriever (via Chroma)
# ==========================================================

def retrieve_top_k_semantic(query, artifacts, k=3):
    query_embedding = artifacts["embedding_model"].encode(
        [query], convert_to_numpy=True, normalize_embeddings=True
    )
    result = artifacts["chroma_collection"].query(
        query_embeddings=query_embedding.tolist(),
        n_results=k,
    )
    ids = [int(i) for i in result["ids"][0]]
    # Chroma (cosine space) returns a distance; similarity = 1 - distance
    similarities = [1 - d for d in result["distances"][0]]

    results = artifacts["chunks_df"].set_index("chunk_id").loc[ids].reset_index()
    results["score"] = similarities
    return results


def _semantic_scores_full_corpus(query, artifacts):
    """Semantic similarity for every chunk (needed for hybrid fusion)."""
    total = len(artifacts["chunks_df"])
    query_embedding = artifacts["embedding_model"].encode(
        [query], convert_to_numpy=True, normalize_embeddings=True
    )
    result = artifacts["chroma_collection"].query(
        query_embeddings=query_embedding.tolist(),
        n_results=total,
    )
    ids = [int(i) for i in result["ids"][0]]
    similarities = [1 - d for d in result["distances"][0]]

    scores = np.zeros(total)
    id_to_score = dict(zip(ids, similarities))
    chunk_ids = artifacts["chunks_df"]["chunk_id"].tolist()
    for position, chunk_id in enumerate(chunk_ids):
        scores[position] = id_to_score.get(chunk_id, 0.0)
    return scores


# ==========================================================
# Section 12 — Hybrid Retriever
# ==========================================================

def min_max_normalize(scores):
    scores = np.array(scores, dtype=float)
    min_score, max_score = scores.min(), scores.max()
    if max_score == min_score:
        return np.zeros_like(scores)
    return (scores - min_score) / (max_score - min_score)


def retrieve_top_k_hybrid(query, artifacts, alpha=0.6, k=3):
    tokenized_query = simple_tokenize(query)
    lexical_scores = artifacts["bm25"].get_scores(tokenized_query)
    semantic_scores = _semantic_scores_full_corpus(query, artifacts)

    hybrid_scores = (
        alpha * min_max_normalize(semantic_scores)
        + (1 - alpha) * min_max_normalize(lexical_scores)
    )

    ranking = np.argsort(hybrid_scores)[::-1][:k]
    results = artifacts["chunks_df"].iloc[ranking].copy()
    results["score"] = hybrid_scores[ranking]
    return results


# ==========================================================
# Section 14 — Context Building
# ==========================================================

def build_context_package(
    query,
    artifacts,
    retrieval_k=10,
    alpha=0.6,
    max_context_chunks=5,
    max_chunks_per_document=2,
    word_budget=300,
    min_score_ratio=0.35,
):
    """Build a clean, deduplicated context package for the LLM prompt."""
    import hashlib

    candidates = retrieve_top_k_hybrid(query, artifacts, alpha=alpha, k=retrieval_k)

    if candidates.empty:
        return {
            "context_text": "",
            "selected_df": candidates,
            "used_words": 0,
            "num_sources": 0,
            "num_chunks": 0,
            "retrieval_k": retrieval_k,
        }

    top_score = candidates["score"].max()
    candidates = candidates[candidates["score"] >= top_score * min_score_ratio]

    candidates = (
        candidates.sort_values("score", ascending=False)
        .groupby("document_id")
        .head(max_chunks_per_document)
    )

    candidates = candidates.sort_values("score", ascending=False).head(max_context_chunks)

    selected_rows = []
    used_words = 0
    seen_chunks = set()

    for _, row in candidates.iterrows():
        chunk_text = row["chunk_text"]
        chunk_hash = hashlib.md5(chunk_text.encode("utf-8")).hexdigest()

        if chunk_hash in seen_chunks:
            continue

        chunk_words = len(chunk_text.split())
        if used_words + chunk_words > word_budget:
            continue

        selected_rows.append(row)
        seen_chunks.add(chunk_hash)
        used_words += chunk_words

    selected_df = pd.DataFrame(selected_rows)
    if not selected_df.empty:
        selected_df = selected_df.sort_values(by="score", ascending=False)

    context_lines = []
    for _, row in selected_df.iterrows():
        source = f"[Source: {row['title']} | {row['authors']} | {row['publication_year']} | {row['language']}]"
        context_lines.append(f"{source}\n{row['chunk_text']}")

    context_text = "\n\n".join(context_lines)

    return {
        "context_text": context_text,
        "selected_df": selected_df,
        "used_words": used_words,
        "num_sources": selected_df["document_id"].nunique() if not selected_df.empty else 0,
        "num_chunks": len(selected_df),
        "retrieval_k": retrieval_k,
    }
