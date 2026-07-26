"""
01_documents.py
================
Section 0-4 من النوت بوك الأصلي: قراءة ملفات الـ PDF (أبحاث + كتب)،
تصحيح اتجاه النص العربي، كشف اللغة، دمج الميتاداتا، وبناء DataFrame
موحّد لكل المستندات. يُشغَّل مرة واحدة (محليًا أو على Colab) وينتج
ملف `documents.parquet` يُستخدم كمدخل لباقي خطوات الـ pipeline.

تشغيل:
    python 01_documents.py --base-path "/path/to/Data" --out rag_export/documents.parquet
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import fitz          # PyMuPDF
import pdfplumber
from bidi.algorithm import get_display


# ==========================================================
# Section 1 — Extract Text from PDF + Fix Arabic Text Direction
# ==========================================================

def extract_text_pdfplumber(pdf_path):
    """Extract text from a PDF using pdfplumber, page by page."""
    pages_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            pages_text.append(text)
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
    PyMuPDF if the extracted text looks too short.
    """
    pages_text = extract_text_pdfplumber(pdf_path)
    avg_len = np.mean([len(p) for p in pages_text]) if pages_text else 0
    if avg_len < min_chars_per_page:
        pages_text = extract_text_pymupdf(pdf_path)
    return pages_text


def fix_arabic_text_direction(text):
    """
    Fix reversed Arabic text extraction (a common PyMuPDF/pdfplumber issue
    with certain Arabic PDF generators, where glyphs are stored in visual
    order instead of logical order). Applies the Unicode Bidi Algorithm
    line by line, which correctly preserves embedded numbers/Latin text
    (e.g. citation years) instead of scrambling them.
    """
    lines = text.split("\n")
    fixed_lines = [get_display(line) for line in lines]
    return "\n".join(fixed_lines)


# ==========================================================
# Section 2 — Language Detection
# ==========================================================

ARABIC_PATTERN = re.compile(r"[\u0600-\u06FF]")


def detect_language(text, arabic_char_threshold=0.15):
    """
    Detect whether text is Arabic or English based on the
    proportion of Arabic Unicode characters.

    Returns "ar", "en", or "unknown".
    """
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


# ==========================================================
# Section 3 — Metadata (Matched by File Order, NOT Filename)
# ==========================================================
# IMPORTANT: Metadata is matched according to the sorted order of the PDF
# files, NOT by filename. If the order of the PDF files changes, this
# metadata list must be updated accordingly.

PAPER_METADATA_LIST = [
    {
        "title": "Digital Communication and its Impact on the Process of Social Interaction Among Jordanian Youth",
        "authors": "Ibrahim Ahmed Al-Adra, Haya Husain Al-Tarawneh, Enas Ghassan Al-Zeqef, Duha Samir Younes",
        "publication_year": 2026,
        "source_type": "paper",
    },
    {
        "title": "The Impact of Digital Communication Devices on Face-to-Face Interactions in Public Spaces: The Case Study of Coffeehouses in Cairo, Egypt",
        "authors": "Ehab Shawkya, Manal Aboub, Hala Elanggari",
        "publication_year": 2020,
        "source_type": "paper",
    },
    {
        "title": "Computers and Society in the Past Half Century: The Conquest of Will Revisited",
        "authors": "Abbe Mowshowitz",
        "publication_year": 2024,
        "source_type": "paper",
    },
    {
        "title": "The Relationship Between Computers and Society: Impacts, Challenges, and Opportunities",
        "authors": "Sb Joseph, Adeniyi Sunny (Dept. of Computer Science, LAUTECH University)",
        "publication_year": None,
        "source_type": "paper",
    },
    {
        "title": "Virtual Impressions: The Effect of Digital Communication on Millennial Social Interaction",
        "authors": "Caitlin Therese Begg",
        "publication_year": 2016,
        "source_type": "paper",
    },
    {
        "title": "Rethinking Social Relationships in Old Age: Digitalization and the Social Lives of Older Adults",
        "authors": "Gizem Hulur, Birthe Macdonald",
        "publication_year": 2020,
        "source_type": "paper",
    },
    {
        "title": "The Impact of Digital Technologies on Urban Life Quality and Social Dynamics in Bismayah",
        "authors": "Rafah Zuhair Alshaikh, Mufeed Ehsan Shok, Zahraa Imad Hussain Al-Hussaini, Amer Shakir Alkinani",
        "publication_year": 2024,
        "source_type": "paper",
    },
    {
        "title": "The Impact of Digital Technology on Social Relationships and Community Dynamics in Contemporary Society",
        "authors": "Dr. Monika Sharma",
        "publication_year": None,
        "source_type": "paper",
    },
    {
        "title": "The Impact of the Internet on Society: A Global Perspective",
        "authors": None,
        "publication_year": 2024,
        "source_type": "paper",
    },
    {
        "title": "The Impact of Digitalization on Social Interaction and Public Space",
        "authors": "Susan J. Drucker, Gary Gumpert",
        "publication_year": 2020,
        "source_type": "paper",
    },
    {
        "title": "Social Identities, Group Formation, and the Analysis of Online Communities",
        "authors": "Jillianne R. Code, Nicholas E. Zaparyniuk",
        "publication_year": 2010,
        "source_type": "paper",
    },
]

BOOK_METADATA_LIST = [
    {
        "title": "Online Communities and Social Computing",
        "authors": "A. Ant Ozok, Panayiotis Zaphiris (Eds.)",
        "publication_year": None,
        "source_type": "book",
    },
]


# ==========================================================
# Section 4 — Build Documents (Text + Metadata)
# ==========================================================

def remove_reference_section(text):
    """Remove the References/Bibliography section from a document."""
    patterns = [
        r"(?im)^references\s*$",
        r"(?im)^bibliography\s*$",
        r"(?im)^works cited\s*$",
        r"(?im)^reference list\s*$",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return text[:match.start()].strip()
    return text


def load_pdf_as_document(pdf_path, document_id, source_type, meta):
    """Load a single PDF and combine its extracted text with metadata."""
    file_name = Path(pdf_path).name

    pages = extract_text_from_pdf(pdf_path)
    full_text = "\n".join(pages)

    language = detect_language(full_text)

    if language == "ar":
        full_text = fix_arabic_text_direction(full_text)

    full_text = remove_reference_section(full_text)

    # Whitespace normalization applies to ALL languages (bug fix from the
    # earlier version, which only normalized Arabic documents).
    full_text = re.sub(r"\s+", " ", full_text).strip()

    return {
        "document_id": document_id,
        "title": meta.get("title", file_name),
        "authors": meta.get("authors") or "Unknown",
        "publication_year": meta.get("publication_year"),
        "source_type": source_type,
        "doc_type": "research_paper" if source_type == "paper" else "book_chapter",
        "language": language,
        "is_current": True,
        "file_name": file_name,
        "num_pages": len(pages),
        "text": full_text,
    }


def build_documents_dataframe(base_path):
    base_path = Path(base_path)
    books_path = base_path / "Book"
    papers_path = base_path / "Papers"

    assert books_path.exists(), f"Books folder not found: {books_path}"
    assert papers_path.exists(), f"Papers folder not found: {papers_path}"

    paper_files = sorted(papers_path.glob("*.pdf"))
    book_files = sorted(books_path.glob("*.pdf"))

    assert len(paper_files) == len(PAPER_METADATA_LIST), (
        f"Number of paper PDFs ({len(paper_files)}) does not match "
        f"PAPER_METADATA_LIST ({len(PAPER_METADATA_LIST)})."
    )
    assert len(book_files) == len(BOOK_METADATA_LIST), (
        f"Number of book PDFs ({len(book_files)}) does not match "
        f"BOOK_METADATA_LIST ({len(BOOK_METADATA_LIST)})."
    )

    documents = []
    document_id = 0

    for pdf_path, meta in zip(paper_files, PAPER_METADATA_LIST):
        documents.append(load_pdf_as_document(pdf_path, document_id, "paper", meta))
        document_id += 1

    for pdf_path, meta in zip(book_files, BOOK_METADATA_LIST):
        documents.append(load_pdf_as_document(pdf_path, document_id, "book", meta))
        document_id += 1

    return pd.DataFrame(documents)


def main():
    parser = argparse.ArgumentParser(description="Build documents.parquet from PDFs")
    parser.add_argument("--base-path", required=True, help="Folder containing 'Book' and 'Papers' subfolders")
    parser.add_argument("--out", default="rag_export/documents.parquet", help="Output parquet path")
    args = parser.parse_args()

    documents_df = build_documents_dataframe(args.base_path)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    documents_df.to_parquet(out_path, index=False)

    print(f"Total documents loaded: {len(documents_df)}")
    print(documents_df[["document_id", "title", "language", "source_type", "num_pages"]])
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
