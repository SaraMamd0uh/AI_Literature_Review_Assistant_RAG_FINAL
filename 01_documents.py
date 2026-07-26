"""
01_documents.py
================
Stage 1 of the RAG pipeline.

Loads all PDF papers + the book from disk, extracts text (with a PyMuPDF
fallback for pages pdfplumber struggles with), fixes reversed Arabic text
direction (a common PDF-extraction issue), detects each document's language,
and attaches manually-verified metadata (title, authors, publication_year).

Output: data/documents.parquet  (one row per document)

Run standalone:
    python 01_documents.py --papers_dir data/pdfs/papers --book_dir data/pdfs/book --out data/documents.parquet
"""

import os
import re
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import fitz            # PyMuPDF
import pdfplumber
from bidi.algorithm import get_display


# ---------------------------------------------------------------------------
# Metadata (order-matched to sorted(glob("*.pdf")) — NOT matched by filename)
# ---------------------------------------------------------------------------
PAPER_METADATA_LIST = [
    {"title": "Digital Communication and its Impact on the Process of Social Interaction Among Jordanian Youth",
     "authors": "Ibrahim Ahmed Al-Adra, Haya Husain Al-Tarawneh, Enas Ghassan Al-Zeqef, Duha Samir Younes",
     "publication_year": 2026},
    {"title": "The Impact of Digital Communication Devices on Face-to-Face Interactions in Public Spaces: The Case Study of Coffeehouses in Cairo, Egypt",
     "authors": "Ehab Shawkya, Manal Aboub, Hala Elanggari",
     "publication_year": 2020},
    {"title": "Computers and Society in the Past Half Century: The Conquest of Will Revisited",
     "authors": "Abbe Mowshowitz",
     "publication_year": 2024},
    {"title": "The Relationship Between Computers and Society: Impacts, Challenges, and Opportunities",
     "authors": "Sb Joseph, Adeniyi Sunny (Dept. of Computer Science, LAUTECH University)",
     "publication_year": "unknown"},
    {"title": "Virtual Impressions: The Effect of Digital Communication on Millennial Social Interaction",
     "authors": "Caitlin Therese Begg",
     "publication_year": 2016},
    {"title": "Rethinking Social Relationships in Old Age: Digitalization and the Social Lives of Older Adults",
     "authors": "Gizem Hülür, Birthe Macdonald",
     "publication_year": 2020},
    {"title": "The Impact of Digital Technologies on Urban Life Quality and Social Dynamics in Bismayah",
     "authors": "Rafah Zuhair Alshaikh, Mufeed Ehsan Shok, Zahraa Imad Hussain Al-Hussaini, Amer Shakir Alkinani",
     "publication_year": 2024},
    {"title": "The Impact of Digital Technology on Social Relationships and Community Dynamics in Contemporary Society",
     "authors": "Dr. Monika Sharma",
     "publication_year": "unknown"},
    {"title": "The Impact of the Internet on Society: A Global Perspective",
     "authors": "unknown",
     "publication_year": 2024},
    {"title": "The Impact of Digitalization on Social Interaction and Public Space",
     "authors": "Susan J. Drucker, Gary Gumpert",
     "publication_year": 2020},
    {"title": "Social Identities, Group Formation, and the Analysis of Online Communities",
     "authors": "Jillianne R. Code, Nicholas E. Zaparyniuk",
     "publication_year": 2010},
]

BOOK_METADATA_LIST = [
    {"title": "Online Communities and Social Computing",
     "authors": "A. Ant Ozok, Panayiotis Zaphiris (Eds.)",
     "publication_year": "unknown"},
]


# ---------------------------------------------------------------------------
# PDF text extraction
# ---------------------------------------------------------------------------
def extract_text_pdfplumber(pdf_path):
    """Extract text from a PDF using pdfplumber, page by page."""
    pages_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            pages_text.append(page.extract_text() or "")
    return pages_text


def extract_text_pymupdf(pdf_path):
    """Extract text from a PDF using PyMuPDF, page by page. Used as a fallback."""
    pages_text = []
    doc = fitz.open(pdf_path)
    for page in doc:
        pages_text.append(page.get_text())
    doc.close()
    return pages_text


def extract_text_from_pdf(pdf_path, min_chars_per_page=20):
    """
    Extract text from a PDF, trying pdfplumber first and falling back to
    PyMuPDF if the extracted text looks too short (common with some scanned
    or complex PDFs, including some Arabic PDFs).
    """
    pages_text = extract_text_pdfplumber(pdf_path)
    avg_len = np.mean([len(p) for p in pages_text]) if pages_text else 0
    if avg_len < min_chars_per_page:
        pages_text = extract_text_pymupdf(pdf_path)
    return pages_text


def fix_arabic_text_direction(text):
    """
    Fix reversed Arabic text extraction (a common PyMuPDF/pdfplumber issue
    with certain Arabic PDF generators). Applies the Unicode Bidi Algorithm
    line by line, which correctly preserves embedded numbers/Latin text.
    """
    lines = text.split("\n")
    return "\n".join(get_display(line) for line in lines)


def detect_language(text, arabic_char_threshold=0.15):
    """Heuristic language detector based on the ratio of Arabic characters."""
    arabic_pattern = re.compile(r"[\u0600-\u06FF]")
    letters = re.findall(r"[^\W\d_]", text, re.UNICODE)
    if not letters:
        return "unknown"
    arabic_letters = arabic_pattern.findall(text)
    ratio = len(arabic_letters) / len(letters)
    return "ar" if ratio >= arabic_char_threshold else "en"


def load_pdf_as_document(pdf_path, document_id, source_type, meta):
    """Load a single PDF into a document dict with full metadata."""
    file_name = os.path.basename(pdf_path)
    pages = extract_text_from_pdf(pdf_path)
    full_text = "\n".join(pages)

    language = detect_language(full_text)
    if language == "ar":
        full_text = fix_arabic_text_direction(full_text)

    full_text = re.sub(r"\s+", " ", full_text).strip()

    return {
        "document_id": document_id,
        "title": meta.get("title", file_name),
        "authors": meta.get("authors", "Unknown"),
        "publication_year": meta.get("publication_year", None),
        "source_type": source_type,
        "doc_type": "research_paper" if source_type == "paper" else "book_chapter",
        "language": language,
        "is_current": True,
        "file_name": file_name,
        "num_pages": len(pages),
        "text": full_text,
    }


def build_documents(papers_dir, book_dir,
                     paper_metadata_list=PAPER_METADATA_LIST,
                     book_metadata_list=BOOK_METADATA_LIST):
    """Build the full documents DataFrame from a papers folder + a book folder."""
    paper_files = sorted(Path(papers_dir).glob("*.pdf"))
    book_files = sorted(Path(book_dir).glob("*.pdf"))

    assert len(paper_files) == len(paper_metadata_list), (
        f"Found {len(paper_files)} paper PDFs but {len(paper_metadata_list)} "
        "metadata entries. Check PAPERS_DIR and PAPER_METADATA_LIST."
    )
    assert len(book_files) == len(book_metadata_list), (
        f"Found {len(book_files)} book PDFs but {len(book_metadata_list)} "
        "metadata entries. Check BOOK_DIR and BOOK_METADATA_LIST."
    )

    documents = []
    document_id = 0

    for pdf_path, meta in zip(paper_files, paper_metadata_list):
        documents.append(load_pdf_as_document(pdf_path, document_id, "paper", meta))
        document_id += 1

    for pdf_path, meta in zip(book_files, book_metadata_list):
        documents.append(load_pdf_as_document(pdf_path, document_id, "book", meta))
        document_id += 1

    return pd.DataFrame(documents)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1: Load PDFs into a documents DataFrame.")
    parser.add_argument("--papers_dir", default="data/pdfs/papers")
    parser.add_argument("--book_dir", default="data/pdfs/book")
    parser.add_argument("--out", default="data/documents.parquet")
    args = parser.parse_args()

    documents_df = build_documents(args.papers_dir, args.book_dir)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    documents_df.to_parquet(args.out, index=False)

    print(f"✅ Saved {len(documents_df)} documents to {args.out}")
    print(documents_df[["document_id", "title", "authors", "publication_year", "language"]])
