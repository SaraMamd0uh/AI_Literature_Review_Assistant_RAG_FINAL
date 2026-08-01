"""
streamlit_app.py
==================
The deployed web app. Loads the pre-built Chroma store + BM25 index
(no PDFs, no Google Drive needed here — those were only needed once,
offline, to run stages 01-05), and answers user questions using the
hybrid retriever (06_retrieve_context.py) + Gemini (07_prompting.py).
"""

import os
import streamlit as st
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions

DATA_PATH = "data/chunks.parquet"
CHROMA_PATH = "chroma_db"

@st.cache_resource
def load_db():
    if not os.path.exists(DATA_PATH):
        st.error(f"ملف {DATA_PATH} مش موجود. اتأكدي انك رفعتيه على GitHub.")
        st.stop()
    
    df = pd.read_parquet(DATA_PATH)
    
    # لو chroma_db مش موجود، ابنِه من جديد
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    # ... باقي كود الـ embedding بتاعك
    return client, df

client, df = load_db()

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

st.set_page_config(
    page_title="مساعد مراجعة الأدبيات — أثر الرقمنة على التفاعل الاجتماعي",
    page_icon="📚",
    layout="centered",
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
# UI
# ---------------------------------------------------------------------------
st.title("📚 مساعد مراجعة الأدبيات")
st.caption("أثر الرقمنة على التفاعلات الاجتماعية — نظام RAG ثنائي اللغة (عربي/إنجليزي) على 11 ورقة بحثية وكتاب")

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

            st.markdown("### 💡 الإجابة")
            st.write(answer)

            st.markdown("### 📖 المصادر المستخدمة")
            sources = package["selected_df"][
                ["title", "authors", "publication_year", "language"]
            ].drop_duplicates()
            st.dataframe(sources, use_container_width=True, hide_index=True)

            if show_context:
                st.markdown("### 🔎 النص المسترجع بالتفصيل (Context)")
                st.text(package["context_text"])
elif submitted:
    st.info("اكتبي سؤالًا أولًا.")

st.divider()
st.caption(
    "مشروع أكاديمي — الإجابات مولّدة تلقائيًا من المصادر المرفوعة فقط، وقد تحتوي على أخطاء — راجعي المصادر الأصلية."
)
