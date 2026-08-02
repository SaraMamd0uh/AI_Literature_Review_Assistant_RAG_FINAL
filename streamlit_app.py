"""
streamlit_app.py
==================
The deployed web app. Loads the pre-built Chroma store + BM25 index
(no PDFs, no Google Drive needed here — those were only needed once,
offline, to run stages 01-05), and answers user questions using the
hybrid retriever (06_retrieve_context.py) + Gemini (07_prompting.py).
"""

import os
import importlib
from pathlib import Path

import streamlit as st

# Numbered filenames aren't valid Python identifiers, so we import them by
# string name via importlib instead of a normal `import 06_retrieve_context`.
retrieve_context = importlib.import_module("06_retrieve_context")
prompting = importlib.import_module("07_prompting")

RetrievalIndex = retrieve_context.RetrievalIndex
build_context_package = retrieve_context.build_context_package
generate_answer = prompting.generate_answer

CHUNKS_PATH = "data/chunks.parquet"
CHROMA_DIR = "chroma_db"
EMBEDDING_MODEL_FILE = "data/embedding_model_name.txt"
DEFAULT_EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# ---------------------------------------------------------------------------
# Branding
# ---------------------------------------------------------------------------
BRAND_NAME = "Scholaris"
BRAND_TAGLINE = "Literature Review Intelligence — Grounded in Your Sources"

st.set_page_config(
    page_title=f"{BRAND_NAME} | مساعد مراجعة الأدبيات",
    page_icon="📚",
    layout="centered",
    initial_sidebar_state="collapsed",
)


@st.cache_resource(show_spinner="بنجهّز قاعدة المعرفة (أول مرة بس)...")
def get_index():
    model_name = DEFAULT_EMBEDDING_MODEL
    if Path(EMBEDDING_MODEL_FILE).exists():
        model_name = Path(EMBEDDING_MODEL_FILE).read_text().strip()
    return RetrievalIndex(
        chunks_parquet_path=CHUNKS_PATH,
        chroma_persist_dir=CHROMA_DIR,
        embedding_model_name=model_name,
    )


def get_api_key():
    if "GEMINI_API_KEY" in st.secrets:
        return st.secrets["GEMINI_API_KEY"]
    return os.environ.get("GEMINI_API_KEY")


# ---------------------------------------------------------------------------
# Styling (corporate / professional look + logo + brand name)
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Cairo:wght@400;600;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', 'Cairo', sans-serif;
    }

    .block-container {
        padding-top: 1.4rem;
        max-width: 880px;
    }

    /* ---- Hero / brand header ---- */
    .brand-hero {
        background: linear-gradient(135deg, #0b1220 0%, #10233d 45%, #1e3a8a 100%);
        border-radius: 20px;
        padding: 2.2rem 2rem 2rem 2rem;
        margin-bottom: 1.6rem;
        box-shadow: 0 12px 32px rgba(16, 35, 61, 0.28);
        position: relative;
        overflow: hidden;
    }
    .brand-hero::after {
        content: "";
        position: absolute;
        top: -60px; right: -60px;
        width: 220px; height: 220px;
        background: radial-gradient(circle, rgba(99,102,241,0.35), transparent 70%);
    }
    .brand-row {
        display: flex;
        align-items: center;
        gap: 0.85rem;
        margin-bottom: 0.6rem;
    }
    .brand-logo {
        flex-shrink: 0;
        width: 46px; height: 46px;
        border-radius: 12px;
        background: linear-gradient(135deg, #6366f1, #22d3ee);
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 4px 14px rgba(99,102,241,0.45);
    }
    .brand-name {
        color: #ffffff;
        font-size: 1.55rem;
        font-weight: 800;
        letter-spacing: 0.2px;
        line-height: 1;
    }
    .brand-name span {
        color: #7dd3fc;
    }
    .brand-tagline {
        color: rgba(226, 232, 255, 0.75);
        font-size: 0.85rem;
        font-weight: 500;
        margin-top: 0.1rem;
    }
    .brand-title-ar {
        color: #ffffff;
        font-size: 1.15rem;
        font-weight: 700;
        margin-top: 1.1rem;
    }
    .brand-desc-ar {
        color: rgba(226, 232, 255, 0.85);
        font-size: 0.92rem;
        margin-top: 0.35rem;
        line-height: 1.7;
    }
    .brand-pills {
        display: flex;
        gap: 0.5rem;
        margin-top: 1rem;
        flex-wrap: wrap;
    }
    .brand-pill {
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.14);
        color: #e5e9ff;
        font-size: 0.76rem;
        font-weight: 600;
        padding: 0.28rem 0.7rem;
        border-radius: 999px;
    }

    /* ---- Section headers ---- */
    .section-label {
        font-weight: 700;
        font-size: 0.98rem;
        color: #1e293b;
        margin: 1.1rem 0 0.5rem 0;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }

    /* ---- Form card ---- */
    div[data-testid="stForm"] {
        background: #ffffff;
        border: 1px solid #e6e9f2;
        border-radius: 16px;
        padding: 1.4rem 1.4rem 1.1rem 1.4rem;
        box-shadow: 0 2px 14px rgba(15, 23, 42, 0.04);
    }

    div[data-testid="stFormSubmitButton"] button {
        background: linear-gradient(135deg, #4338ca, #2563eb);
        color: white;
        font-weight: 700;
        border: none;
        border-radius: 10px;
        padding: 0.55rem 1.4rem;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.35);
    }
    div[data-testid="stFormSubmitButton"] button:hover {
        filter: brightness(1.08);
    }

    /* ---- Answer card ---- */
    .answer-box {
        background: #ffffff;
        border: 1px solid #e6e9f2;
        border-left: 5px solid #4338ca;
        border-radius: 14px;
        padding: 1.1rem 1.3rem;
        line-height: 1.9;
        font-size: 1rem;
        color: #1e293b;
        box-shadow: 0 2px 14px rgba(15, 23, 42, 0.04);
    }

    footer, #MainMenu { visibility: hidden; }

    .app-footer {
        text-align: center;
        color: #94a3b8;
        font-size: 0.78rem;
        margin-top: 1.6rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Brand hero (logo + name + description) — replaces the old plain title
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="brand-hero">
        <div class="brand-row">
            <div class="brand-logo">
                <svg width="26" height="26" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M4 5.5C4 4.67 4.67 4 5.5 4H11V20H5.5C4.67 20 4 19.33 4 18.5V5.5Z" fill="white" fill-opacity="0.92"/>
                    <path d="M13 4H18.5C19.33 4 20 4.67 20 5.5V18.5C20 19.33 19.33 20 18.5 20H13V4Z" fill="white" fill-opacity="0.65"/>
                </svg>
            </div>
            <div>
                <div class="brand-name">{BRAND_NAME}<span>.</span></div>
                <div class="brand-tagline">{BRAND_TAGLINE}</div>
            </div>
        </div>
        <div class="brand-title-ar">📚 مساعد مراجعة الأدبيات الذكي</div>
        <div class="brand-desc-ar">
            نظام RAG ثنائي اللغة (عربي/إنجليزي) يبحث في 11 ورقة بحثية وكتاب حول أثر الرقمنة
            على التفاعلات الاجتماعية، ويرجع لكِ إجابة دقيقة مع مصدرها الكامل: الباحث، العنوان، وسنة النشر.
        </div>
        <div class="brand-pills">
            <span class="brand-pill">🔎 استرجاع هجين (Semantic + BM25)</span>
            <span class="brand-pill">🌐 عربي / إنجليزي</span>
            <span class="brand-pill">📌 مصادر موثّقة</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not (Path(CHUNKS_PATH).exists() and Path(CHROMA_DIR).exists()):
    st.error(
        "لم يتم العثور على `data/chunks.parquet` أو مجلد `chroma_db/`. "
        "لازم تشغّلي المراحل 01 إلى 05 أولاً (محليًا أو في Colab) وترفعي نواتجها مع الكود."
    )
    st.stop()

index = get_index()
api_key = get_api_key()

if not api_key:
    st.warning(
        "⚠️ مفيش مفتاح Gemini API متاح. اضيفي `GEMINI_API_KEY` في إعدادات Secrets "
        "على Streamlit Cloud، أو في ملف `.streamlit/secrets.toml` محليًا."
    )

st.markdown('<div class="section-label">✍️ اسألي عن أي شيء في المصادر</div>', unsafe_allow_html=True)

with st.form("query_form"):
    query = st.text_area(
        "اكتبي سؤالك (عربي أو إنجليزي):",
        placeholder="مثال: How does digitalization affect older adults' social relationships?",
        height=90,
    )
    col1, col2 = st.columns(2)
    with col1:
        alpha = st.slider("وزن الفهم الدلالي (Semantic) مقابل التطابق اللفظي (Lexical)", 0.0, 1.0, 0.6, 0.1)
    with col2:
        show_context = st.checkbox("إظهار النص المسترجع بالتفصيل", value=False)
    submitted = st.form_submit_button("اسأل 🔍")

if submitted and query.strip():
    with st.spinner("بنبحث في المصادر ونجهّز الإجابة..."):
        package = build_context_package(query, index, alpha=alpha)

        if not package["context_text"]:
            st.error("لم يتم العثور على مصادر ذات صلة كافية بالسؤال.")
        else:
            if api_key:
                try:
                    answer, _ = generate_answer(query, package["context_text"], api_key=api_key)
                except Exception as e:
                    answer = f"⚠️ حصل خطأ في استدعاء الموديل: {e}"
            else:
                answer = "⚠️ مفيش مفتاح API — تم عرض المصادر المسترجعة فقط بدون إجابة مولّدة."

            st.markdown('<div class="section-label">💡 الإجابة</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)

            st.markdown('<div class="section-label">📖 المصادر المستخدمة</div>', unsafe_allow_html=True)
            sources = package["selected_df"][
                ["title", "authors", "publication_year", "language"]
            ].drop_duplicates()
            st.dataframe(sources, use_container_width=True, hide_index=True)

            if show_context:
                st.markdown('<div class="section-label">🔎 النص المسترجع بالتفصيل (Context)</div>', unsafe_allow_html=True)
                st.text(package["context_text"])
elif submitted:
    st.info("اكتبي سؤالًا أولًا.")

st.divider()
st.markdown(
    f"""
    <div class="app-footer">
        {BRAND_NAME} © 2026 — مشروع أكاديمي. الإجابات مولّدة تلقائيًا من المصادر المرفوعة فقط،
        وقد تحتوي على أخطاء — راجعي المصادر الأصلية.
    </div>
    """,
    unsafe_allow_html=True,
)
