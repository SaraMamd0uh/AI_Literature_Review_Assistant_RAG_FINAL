"""
06_retrieve_context.py
========================
Stage 6 of the RAG pipeline (runtime library — imported by streamlit_app.py).

Given a user query, retrieves the most relevant chunks using a HYBRID
strategy:
    - Semantic search against the persistent Chroma collection
      (built by 05_create_chroma_store.py) — captures meaning, and works
      cross-lingually thanks to the multilingual embedding model.
    - Lexical (BM25) search over the same chunks — captures exact keyword
      matches (names, numbers, specific terms) that embeddings can miss.

Then packages the top results into a clean, budget-limited, labeled
context block ready to be dropped into an LLM prompt (07_prompting.py).

This module is meant to be imported, not run as a CLI step, since it
needs the persisted Chroma store + chunks.parquet to already exist
(produced by stages 03-05).
"""

import re

import numpy as np
import pandas as pd
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

COLLECTION_NAME = "digital_society_chunks"


# ---------------------------------------------------------------------------
# Bilingual lexical helpers (kept self-contained here so this module has no
# fragile cross-file imports at runtime)
# ---------------------------------------------------------------------------
def detect_language(text, arabic_char_threshold=0.15):
    arabic_pattern = re.compile(r"[\u0600-\u06FF]")
    letters = re.findall(r"[^\W\d_]", text, re.UNICODE)
    if not letters:
        return "unknown"
    arabic_letters = arabic_pattern.findall(text)
    ratio = len(arabic_letters) / len(letters)
    return "ar" if ratio >= arabic_char_threshold else "en"


def normalize_whitespace(text):
    return re.sub(r"\s+", " ", text).strip()


ARABIC_DIACRITICS = re.compile(r"[\u064B-\u0652\u0670\u0640]")


def normalize_arabic_text(text):
    text = ARABIC_DIACRITICS.sub("", text)
    text = re.sub(r"[إأآ]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"[^\u0600-\u06FF\s0-9]", " ", text)
    return normalize_whitespace(text)


def normalize_english_text(text):
    text = re.sub(r"[^\w\s]", " ", text.lower())
    return normalize_whitespace(text)


def normalize_lexical_text(text):
    language = detect_language(text)
    return normalize_arabic_text(text) if language == "ar" else normalize_english_text(text)


def simple_tokenize(text):
    text = normalize_lexical_text(text)
    return re.findall(r"[a-z0-9]+|[\u0600-\u06FF]+", text.lower())


def min_max_normalize(scores):
    scores = np.array(scores, dtype=float)
    lo, hi = scores.min(), scores.max()
    if hi == lo:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)


# ---------------------------------------------------------------------------
# Loading the pre-built index (cheap — no PDFs, no re-embedding of the corpus)
# ---------------------------------------------------------------------------
class RetrievalIndex:
    """Bundles everything needed to answer retrieval queries: the Chroma
    collection (semantic), a BM25 index (lexical) rebuilt from chunks.parquet,
    and the chunk metadata table used to label results."""

    def __init__(self, chunks_parquet_path, chroma_persist_dir,
                 embedding_model_name="paraphrase-multilingual-MiniLM-L12-v2"):
        self.chunks_df = pd.read_parquet(chunks_parquet_path)

        client = chromadb.PersistentClient(path=chroma_persist_dir)
        self.collection = client.get_collection(COLLECTION_NAME)

        self.embed_model = SentenceTransformer(embedding_model_name)

        tokenized_chunks = [simple_tokenize(t) for t in self.chunks_df["search_text"]]
        self.bm25 = BM25Okapi(tokenized_chunks)

    def retrieve_semantic(self, query, k=10):
        query_embedding = self.embed_model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        )[0].tolist()

        results = self.collection.query(query_embeddings=[query_embedding], n_results=k)

        chunk_ids = [int(cid) for cid in results["ids"][0]]
        # Chroma cosine distance -> similarity score (higher is better)
        distances = results["distances"][0]
        scores = [1 - d for d in distances]

        return chunk_ids, scores

    def retrieve_lexical(self, query):
        tokenized_query = simple_tokenize(query)
        return self.bm25.get_scores(tokenized_query)  # aligned to chunks_df row order

    def retrieve_hybrid(self, query, alpha=0.6, k=10):
        """
        hybrid_score = alpha * semantic_score + (1 - alpha) * lexical_score
        alpha closer to 1 favors meaning/cross-lingual matches;
        alpha closer to 0 favors exact keyword/number matches.
        """
        semantic_chunk_ids, semantic_scores_subset = self.retrieve_semantic(query, k=len(self.chunks_df))
        semantic_scores_full = np.zeros(len(self.chunks_df))
        id_to_row = {cid: i for i, cid in enumerate(self.chunks_df["chunk_id"].tolist())}
        for cid, score in zip(semantic_chunk_ids, semantic_scores_subset):
            row_idx = id_to_row.get(cid)
            if row_idx is not None:
                semantic_scores_full[row_idx] = score

        lexical_scores_full = self.retrieve_lexical(query)

        hybrid_scores = (
            alpha * min_max_normalize(semantic_scores_full)
            + (1 - alpha) * min_max_normalize(lexical_scores_full)
        )

        ranking = np.argsort(hybrid_scores)[::-1][:k]
        results = self.chunks_df.iloc[ranking].copy()
        results["score"] = hybrid_scores[ranking]
        return results


# ---------------------------------------------------------------------------
# Context building (turn raw candidates into a clean, labeled context block)
# ---------------------------------------------------------------------------
def build_context_package(
    query,
    index: RetrievalIndex,
    retrieval_k=10,
    alpha=0.6,
    max_context_chunks=5,
    max_chunks_per_document=2,
    word_budget=300,
    min_score_ratio=0.35,
):
    candidates = index.retrieve_hybrid(query, alpha=alpha, k=retrieval_k)

    if len(candidates) == 0:
        return {"context_text": "", "selected_df": candidates, "used_words": 0, "num_sources": 0}

    top_score = candidates["score"].max()
    candidates = candidates[candidates["score"] >= top_score * min_score_ratio]

    candidates = (
        candidates.sort_values(by="score", ascending=False)
        .groupby("document_id")
        .head(max_chunks_per_document)
    )
    candidates = candidates.sort_values(by="score", ascending=False).head(max_context_chunks)

    selected_rows = []
    used_words = 0
    seen_snippets = set()

    for _, row in candidates.iterrows():
        snippet_key = row["chunk_text"][:80]
        if snippet_key in seen_snippets:
            continue
        word_count = len(row["chunk_text"].split())
        if used_words + word_count > word_budget:
            continue
        selected_rows.append(row)
        seen_snippets.add(snippet_key)
        used_words += word_count

    selected_df = pd.DataFrame(selected_rows)

    context_lines = []
    for _, row in selected_df.iterrows():
        label = f"[Source: {row['title']} — {row['authors']} ({row['publication_year']}), lang={row['language']}]"
        context_lines.append(f"{label}\n{row['chunk_text']}")

    return {
        "context_text": "\n\n".join(context_lines),
        "selected_df": selected_df,
        "used_words": used_words,
        "num_sources": selected_df["document_id"].nunique() if len(selected_df) else 0,
    }
