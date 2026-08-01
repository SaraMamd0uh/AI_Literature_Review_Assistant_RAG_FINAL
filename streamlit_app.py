"""
streamlit_app.py - FIXED VERSION
يعمل حتى لو chroma_db/ غير موجود. يحتاج فقط data/chunks.parquet
(ويفضل معه data/chunk_embeddings.npy لتسريع أول تشغيل، لكنه ليس ضروري)
"""
import os
import re
from pathlib import Path
import importlib.util

import streamlit as st
import pandas as pd
import numpy as np
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# ---------------- Config ----------------
CHUNKS_PATH = Path("data/chunks.parquet")
CHROMA_DIR = Path("chroma_db")
EMBEDDINGS_PATH = Path("data/chunk_embeddings.npy")
EMBEDDING_MODEL_FILE = Path("data/embedding_model_name.txt")
DEFAULT_EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
COLLECTION_NAME = "digital_society_chunks"

st.set_page_config(
    page_title="مساعد مراجعة الأدبيات — أثر الرقمنة",
    page_icon="📚",
    layout="centered",
)

# ---------------- Helpers - bilingual ----------------
def detect_language(text, arabic_char_threshold=0.15):
    arabic_pattern = re.compile(r"[\u0600-\u06FF]")
    letters = re.findall(r"[^\W\d_]", text, re.UNICODE)
    if not letters:
        return "unknown"
    arabic_letters = arabic_pattern.findall(text)
    ratio = len(arabic_letters) / len(letters) if letters else 0
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
    lang = detect_language(text)
    return normalize_arabic_text(text) if lang == "ar" else normalize_english_text(text)

def simple_tokenize(text):
    text = normalize_lexical_text(text)
    return re.findall(r"[a-z0-9]+|[\u0600-\u06FF]+", text.lower())

def min_max_normalize(scores):
    scores = np.array(scores, dtype=float)
    lo, hi = scores.min(), scores.max()
    if hi == lo:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)

def sanitize_metadata_value(value):
    if value is None:
        return "unknown"
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)

# ---------------- Retrieval Index ----------------
class RetrievalIndex:
    def __init__(self, chunks_df, collection, embed_model, bm25):
        self.chunks_df = chunks_df
        self.collection = collection
        self.embed_model = embed_model
        self.bm25 = bm25

    def retrieve_semantic(self, query, k=10):
        query_embedding = self.embed_model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        )[0].tolist()
        results = self.collection.query(query_embeddings=[query_embedding], n_results=k)
        chunk_ids = [int(cid) for cid in results["ids"][0]]
        distances = results["distances"][0]
        scores = [1 - d for d in distances]
        return chunk_ids, scores

    def retrieve_lexical(self, query):
        tokenized_query = simple_tokenize(query)
        return self.bm25.get_scores(tokenized_query)

    def retrieve_hybrid(self, query, alpha=0.6, k=10):
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

def build_context_package(query, index: RetrievalIndex, retrieval_k=15, alpha=0.6, max_context_chunks=5, max_chunks_per_document=2, word_budget=600, min_score_ratio=0.35):
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

# ---------------- LLM ----------------
def build_strict_prompt(query, context_text):
    return f"""You are a research assistant specialized in digital sociology and the impact of digitalization on social interaction.

Rules:
1. Use ONLY the information inside <context>. Do not use outside knowledge.
2. If the context is insufficient, respond exactly: "لا توجد معلومات كافية في المصادر للإجابة على هذا السؤال."
3. Every claim must be followed by a citation in the format (Author, Year).
4. If sources conflict, mention the disagreement explicitly.
5. Answer in the same language as the question (Arabic question -> Arabic answer, English question -> English answer).

Output format:
- Short direct answer (2-4 sentences)
- "Sources used:" followed by bullet list

<context>
{context_text}
</context>

Question: {query}
"""

def call_gemini(prompt, api_key, model_name="gemini-2.5-flash"):
    from google import genai
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(model=model_name, contents=prompt)
    return response.text

# ---------------- Loader with auto-build ----------------
@st.cache_resource(show_spinner="بنجهّز قاعدة المعرفة... قد تأخذ دقيقة أول مرة لتحميل الموديل 🤖")
def get_index():
    # 1. model name
    model_name = DEFAULT_EMBEDDING_MODEL
    if EMBEDDING_MODEL_FILE.exists():
        try:
            model_name = EMBEDDING_MODEL_FILE.read_text(encoding="utf-8").strip() or model_name
        except:
            pass

    # 2. load chunks
    if not CHUNKS_PATH.exists():
        return None, f"ملف {CHUNKS_PATH} غير موجود. شغّلي المراحل 01-04 محلياً وارفعي data/chunks.parquet على GitHub."

    chunks_df = pd.read_parquet(CHUNKS_PATH)
    if len(chunks_df) == 0:
        return None, "ملف chunks.parquet فاضي."

    # 3. embedding model
    embed_model = SentenceTransformer(model_name)

    # 4. BM25
    tokenized_chunks = [simple_tokenize(t) for t in chunks_df["search_text"]]
    bm25 = BM25Okapi(tokenized_chunks)

    # 5. Chroma - try to reuse, else build
    collection = None
    client = None

    if CHROMA_DIR.exists():
        try:
            client = chromadb.PersistentClient(path=str(CHROMA_DIR))
            collection = client.get_collection(COLLECTION_NAME)
            if collection.count() != len(chunks_df):
                st.warning(f"عدد القطع في chroma_db ({collection.count()}) لا يساوي chunks.parquet ({len(chunks_df)}). سيتم إعادة البناء.")
                collection = None
        except Exception as e:
            collection = None

    if collection is None:
        # Build fresh
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        try:
            client.delete_collection(COLLECTION_NAME)
        except:
            pass
        collection = client.create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

        # Load or compute embeddings
        if EMBEDDINGS_PATH.exists():
            try:
                embeddings = np.load(str(EMBEDDINGS_PATH))
                if len(embeddings) != len(chunks_df):
                    raise ValueError("size mismatch")
            except Exception as e:
                embeddings = embed_model.encode(
                    chunks_df["search_text"].tolist(),
                    batch_size=32,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False
                ).astype("float32")
        else:
            embeddings = embed_model.encode(
                chunks_df["search_text"].tolist(),
                batch_size=32,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False
            ).astype("float32")

        ids = [str(cid) for cid in chunks_df["chunk_id"].tolist()]
        documents = chunks_df["chunk_text"].tolist()
        metadatas = []
        for _, row in chunks_df.iterrows():
            metadatas.append({
                "document_id": sanitize_metadata_value(row["document_id"]),
                "title": sanitize_metadata_value(row["title"]),
                "authors": sanitize_metadata_value(row["authors"]),
                "publication_year": sanitize_metadata_value(row["publication_year"]),
                "source_type": sanitize_metadata_value(row["source_type"]),
                "language": sanitize_metadata_value(row["language"]),
                "is_current": sanitize_metadata_value(row["is_current"]),
            })

        batch_size = 500
        for start in range(0, len(ids), batch_size):
            end = start + batch_size
            collection.add(
                ids=ids[start:end],
                embeddings=embeddings[start:end].tolist(),
                documents=documents[start:end],
                metadatas=metadatas[start:end],
            )

    return RetrievalIndex(chunks_df, collection, embed_model, bm25), None

def get_api_key():
    if "GEMINI_API_KEY" in st.secrets:
        return st.secrets["GEMINI_API_KEY"]
    return os.environ.get("GEMINI_API_KEY")

# ---------------- UI ----------------
st.title("📚 مساعد مراجعة الأدبيات")
st.caption("تلقي اللغة (عربي/إنجليزي) على 11 ورقة بحثية وكتاب — RAG أثر الرقمنة على التفاعلات الاجتماعية — نظام")

index, error_msg = get_index()

if error_msg:
    st.error(error_msg)
    st.markdown("""
    **كيف تحلين المشكلة؟**
    1. شغّلي محلياً أو في Colab:
       ```
       python 01_documents.py
       python 02_preprocessing.py
       python 03_chunking.py
       python 04_vector_representation.py
       ```
    2. ثم ارفعي على GitHub فقط هذين الملفين الصغيرين (يكفي):
       - `data/chunks.parquet`
       - `data/embedding_model_name.txt`
       
       اختياري للتسريع: `data/chunk_embeddings.npy` (استخدمي Git LFS لو حجمه > 100MB)
       
       **لا تحتاجين لرفع `chroma_db/` بعد اليوم - التطبيق سيبنيه تلقائياً.**
    """)
    st.stop()

st.success(f"تم تحميل {len(index.chunks_df)} مقطع من {index.chunks_df['document_id'].nunique()} مصدر. جاهز للأسئلة ✅")

api_key = get_api_key()
if not api_key:
    st.warning("⚠️ مفيش مفتاح Gemini API. أضيفي GEMINI_API_KEY في Streamlit Secrets أو .streamlit/secrets.toml")

with st.form("query_form"):
    query = st.text_area(
        "اكتبي سؤالك (عربي أو إنجليزي):",
        placeholder="مثال: How does digitalization affect older adults' social relationships?\nمثال عربي: كيف تؤثر الرقمنة على التفاعل الاجتماعي للشباب؟",
        height=100,
    )
    col1, col2 = st.columns(2)
    with col1:
        alpha = st.slider("وزن الفهم الدلالي (Semantic) مقابل اللفظي (Lexical)", 0.0, 1.0, 0.6, 0.1,
                          help="0 = يعتمد على تطابق الكلمات فقط، 1 = يعتمد على المعنى فقط")
    with col2:
        show_context = st.checkbox("إظهار النص المسترجع بالتفصيل", value=False)
    submitted = st.form_submit_button("اسأل 🔍", use_container_width=True)

if submitted and query.strip():
    with st.spinner("بنبحث في المصادر ونجهّز الإجابة..."):
        package = build_context_package(query, index, alpha=alpha, retrieval_k=15)

        if not package["context_text"]:
            st.error("لم يتم العثور على مصادر ذات صلة كافية بالسؤال. جربي صياغة أخرى.")
        else:
            if api_key:
                try:
                    prompt = build_strict_prompt(query, package["context_text"])
                    answer = call_gemini(prompt, api_key=api_key)
                except Exception as e:
                    answer = f"⚠️ حصل خطأ في استدعاء Gemini: {e}"
            else:
                answer = "⚠️ مفيش مفتاح API — تم عرض المصادر المسترجعة فقط بدون إجابة مولّدة."

            st.markdown("### 💡 الإجابة")
            st.write(answer)

            st.markdown("### 📖 المصادر المستخدمة")
            if len(package["selected_df"]) > 0:
                sources = package["selected_df"][["title", "authors", "publication_year", "language", "score"]].drop_duplicates()
                st.dataframe(sources, use_container_width=True, hide_index=True)
            else:
                st.write("لا يوجد.")

            if show_context:
                st.markdown("### 🔎 النص المسترجع (Context)")
                st.text(package["context_text"])

            st.caption(f"عدد الكلمات المستخدمة في السياق: {package['used_words']} | عدد المصادر: {package['num_sources']}")

elif submitted:
    st.info("اكتبي سؤالاً أولاً.")

st.divider()
st.caption("مشروع أكاديمي — الإجابات مولّدة تلقائياً من المصادر المرفوعة فقط، وقد تحتوي على أخطاء — راجعي المصادر الأصلية.")
