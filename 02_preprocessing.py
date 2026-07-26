"""
02_preprocessing.py
====================
Stage 2 of the RAG pipeline.

Bilingual (Arabic/English) text-cleaning utilities. These are used later
for lexical (keyword) matching in 06_retrieve_context.py, and applied here
to produce a cleaned copy of the documents for inspection.

Input:  data/documents.parquet   (from 01_documents.py)
Output: data/documents_clean.parquet

Run standalone:
    python 02_preprocessing.py --in data/documents.parquet --out data/documents_clean.parquet
"""

import re
import argparse
from pathlib import Path

import pandas as pd


def detect_language(text, arabic_char_threshold=0.15):
    """Heuristic language detector based on the ratio of Arabic characters."""
    arabic_pattern = re.compile(r"[\u0600-\u06FF]")
    letters = re.findall(r"[^\W\d_]", text, re.UNICODE)
    if not letters:
        return "unknown"
    arabic_letters = arabic_pattern.findall(text)
    ratio = len(arabic_letters) / len(letters)
    return "ar" if ratio >= arabic_char_threshold else "en"


def normalize_whitespace(text):
    return re.sub(r"\s+", " ", text).strip()


def remove_urls(text):
    return re.sub(r"http\S+|www\.\S+", "", text)


def preprocess_english_text(text, lowercase=True, remove_url=True, remove_punct=True,
                             remove_num=False, normalize_space=True):
    """English cleaning pipeline: URL removal, lowercasing, punctuation stripping."""
    if remove_url:
        text = remove_urls(text)
    if lowercase:
        text = text.lower()
    if remove_punct:
        text = re.sub(r"[^\w\s]", " ", text)
    if remove_num:
        text = re.sub(r"\d+", "", text)
    if normalize_space:
        text = normalize_whitespace(text)
    return text


ARABIC_DIACRITICS = re.compile(r"[\u064B-\u0652\u0670\u0640]")


def normalize_arabic_text(text):
    """
    Arabic cleaning pipeline:
    - remove diacritics (تشكيل) and tatweel (تطويل)
    - unify alef forms (أ إ آ -> ا) and alef maksura (ى -> ي)
    - keep taa marbuta (ة) since removing it changes word meaning
    - strip non-Arabic punctuation noise
    """
    text = ARABIC_DIACRITICS.sub("", text)
    text = re.sub(r"[إأآ]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"[^\u0600-\u06FF\s0-9]", " ", text)
    text = normalize_whitespace(text)
    return text


def preprocess_text_bilingual(text, language):
    """Route text to the correct cleaning pipeline based on detected language."""
    if language == "ar":
        return normalize_arabic_text(text)
    return preprocess_english_text(text)


def normalize_lexical_text(text):
    """Detect language on the fly and clean accordingly. Used for BM25/keyword matching."""
    language = detect_language(text)
    return preprocess_text_bilingual(text, language)


def simple_tokenize(text):
    """Bilingual tokenizer: Latin letters/numbers OR Arabic letters as tokens."""
    text = normalize_lexical_text(text)
    return re.findall(r"[a-z0-9]+|[\u0600-\u06FF]+", text.lower())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 2: Clean/normalize document text.")
    parser.add_argument("--in", dest="in_path", default="data/documents.parquet")
    parser.add_argument("--out", default="data/documents_clean.parquet")
    args = parser.parse_args()

    documents_df = pd.read_parquet(args.in_path)
    documents_df["clean_preview"] = documents_df.apply(
        lambda row: normalize_lexical_text(row["text"])[:300], axis=1
    )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    documents_df.to_parquet(args.out, index=False)

    print(f"✅ Saved cleaned documents to {args.out}")
    print(documents_df[["document_id", "title", "language"]])
