"""
02_preprocessing.py
====================
Section 5 من النوت بوك الأصلي: دوال تنظيف النص (إنجليزي وعربي)، تُستخدم
لاحقًا في بناء الفهرس اللغوي (TF-IDF / BM25). ده موديول بيتعمله import
من سكريبتات تانية (04_vector_representation.py و 06_retrieve_context.py)
وليس له مخرجات على القرص.
"""

import re

ARABIC_DIACRITICS = re.compile(r"[\u064B-\u0652\u0670\u0640]")


def normalize_whitespace(text):
    """Normalize consecutive whitespace into a single space."""
    return re.sub(r"\s+", " ", text).strip()


def remove_urls(text):
    """Remove URLs from text."""
    return re.sub(r"https?://\S+|www\.\S+", "", text)


def preprocess_english_text(
    text,
    lowercase=True,
    remove_url=True,
    remove_punct=True,
    remove_num=False,
    normalize_space=True,
):
    """Basic preprocessing for English text."""
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


def normalize_arabic_text(text):
    """
    Basic preprocessing for Arabic text.
    Suitable for Retrieval (RAG) while preserving word forms.
    """
    # Remove Arabic diacritics
    text = ARABIC_DIACRITICS.sub("", text)

    # Normalize common letter variants
    text = re.sub(r"[إأآ]", "ا", text)
    text = re.sub(r"ى", "ي", text)

    # Remove punctuation while preserving Arabic, English letters and numbers
    text = re.sub(r"[^\u0600-\u06FFa-zA-Z0-9\s]", " ", text)

    return normalize_whitespace(text)


def preprocess_text_bilingual(text, language):
    """Apply preprocessing according to the detected language."""
    if language == "ar":
        return normalize_arabic_text(text)
    elif language == "en":
        return preprocess_english_text(text)
    else:
        return normalize_whitespace(text)


if __name__ == "__main__":
    # Quick self-test
    sample_en = "Check this out: https://example.com   It's AMAZING!!!"
    sample_ar = "الرَّقْمَنَة غيّرت طريقة تفاعل   الأفراد اجتماعيًا."
    print("EN ->", preprocess_text_bilingual(sample_en, "en"))
    print("AR ->", preprocess_text_bilingual(sample_ar, "ar"))
