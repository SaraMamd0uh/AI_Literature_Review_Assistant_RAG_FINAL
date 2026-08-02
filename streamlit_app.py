"""
LITE version - يفتح في 20 ثانية بدون torch
يستخدم BM25 فقط (بحث بالكلمات) + Gemini
مثالي للعرض السريع
"""
import os
import re
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
from rank_bm25 import BM25Okapi

CHUNKS_PATH = Path("data/chunks.parquet")

st.set_page_config(page_title="مساعد مراجعة الأدبيات — نسخة سريعة", page_icon="📚", layout="centered")

def detect_language(text, thresh=0.15):
    ar = re.compile(r"[\u0600-\u06FF]")
    letters = re.findall(r"[^\W\d_]", text, re.UNICODE)
    if not letters: return "unknown"
    return "ar" if len(ar.findall(text))/len(letters) >= thresh else "en"

def normalize_whitespace(t): return re.sub(r"\s+"," ",t).strip()
AR_DIAC = re.compile(r"[\u064B-\u0652\u0670\u0640]")
def normalize_ar(t):
    t = AR_DIAC.sub("",t)
    t = re.sub(r"[إأآ]","ا",t)
    t = re.sub(r"ى","ي",t)
    t = re.sub(r"[^\u0600-\u06FF\s0-9]"," ",t)
    return normalize_whitespace(t)
def normalize_en(t): return normalize_whitespace(re.sub(r"[^\w\s]"," ",t.lower()))
def normalize_lex(t): return normalize_ar(t) if detect_language(t)=="ar" else normalize_en(t)
def simple_tokenize(t):
    t = normalize_lex(t)
    return re.findall(r"[a-z0-9]+|[\u0600-\u06FF]+", t.lower())

@st.cache_resource(show_spinner="بنجهز البحث السريع...")
def get_index():
    if not CHUNKS_PATH.exists():
        return None, f"ملف {CHUNKS_PATH} غير موجود"
    df = pd.read_parquet(CHUNKS_PATH)
    tokenized = [simple_tokenize(t) for t in df["search_text"]]
    bm25 = BM25Okapi(tokenized)
    return (df, bm25), None

def retrieve(df, bm25, query, k=10):
    tok_q = simple_tokenize(query)
    scores = bm25.get_scores(tok_q)
    ranking = np.argsort(scores)[::-1][:k]
    res = df.iloc[ranking].copy()
    res["score"] = scores[ranking]
    return res

def build_context(query, df, bm25):
    cands = retrieve(df, bm25, query, k=15)
    # فلترة بسيطة
    top = cands["score"].max()
    cands = cands[cands["score"] >= top*0.2].head(5)
    lines=[]
    rows=[]
    used=0
    for _, r in cands.iterrows():
        if used + len(r["chunk_text"].split()) > 600: continue
        lines.append(f"[Source: {r['title']} — {r['authors']} ({r['publication_year']})]\n{r['chunk_text']}")
        rows.append(r)
        used += len(r["chunk_text"].split())
    import pandas as pd
    return {"context_text":"\n\n".join(lines), "selected_df": pd.DataFrame(rows), "used_words": used}

def build_prompt(q, ctx):
    return f"""You are a research assistant specialized in digital sociology.
Use ONLY context. If insufficient say: لا توجد معلومات كافية.
Cite (Author, Year). Answer in same language as question.

<context>{ctx}</context>

Question: {q}
"""

def call_gemini(prompt, api_key):
    from google import genai
    client = genai.Client(api_key=api_key)
    resp = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    return resp.text

def get_api_key():
    if "GEMINI_API_KEY" in st.secrets: return st.secrets["GEMINI_API_KEY"]
    return os.environ.get("GEMINI_API_KEY")

# UI
st.title("📚 مساعد مراجعة الأدبيات")
st.caption("نسخة سريعة (BM25) - تعمل بدون torch - أثر الرقمنة على التفاعلات")

(index_data, err) = get_index()
if err:
    st.error(err); st.stop()
df, bm25 = index_data
st.success(f"تم تحميل {len(df)} مقطع من {df['document_id'].nunique()} مصدر ✅ - وضع سريع")

api_key = get_api_key()
if not api_key:
    st.warning("⚠️ أضيفي GEMINI_API_KEY في Secrets")

with st.form("f"):
    q = st.text_area("سؤالك (عربي/إنجليزي):", placeholder="كيف تؤثر الرقمنة على كبار السن؟", height=90)
    show_ctx = st.checkbox("إظهار النص المسترجع", False)
    sub = st.form_submit_button("اسأل 🔍", use_container_width=True)

if sub and q.strip():
    with st.spinner("نبحث..."):
        pkg = build_context(q, df, bm25)
        if not pkg["context_text"]:
            st.error("لا يوجد مصادر كافية")
        else:
            if api_key:
                try:
                    ans = call_gemini(build_prompt(q, pkg["context_text"]), api_key)
                except Exception as e:
                    ans = f"خطأ Gemini: {e}"
            else:
                ans = "المصادر فقط (لا يوجد API key)"
            st.markdown("### 💡 الإجابة")
            st.write(ans)
            st.markdown("### 📖 المصادر")
            if len(pkg["selected_df"]):
                st.dataframe(pkg["selected_df"][["title","authors","publication_year"]].drop_duplicates(), use_container_width=True, hide_index=True)
            if show_ctx:
                st.text(pkg["context_text"])
elif sub:
    st.info("اكتبي سؤالاً")

st.divider()
st.caption("نسخة خفيفة سريعة - للعرض. النسخة الكاملة تستخدم بحث دلالي + لفظي.")
