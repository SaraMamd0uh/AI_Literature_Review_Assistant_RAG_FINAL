"""
نسخة Llama فقط - OpenRouter
Section 16 - RAG Chatbot
يستخدم فقط meta-llama/llama-3.3-70b-instruct:free
بدون Gemini - سريع وخفيف
"""
import os
import re
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
from rank_bm25 import BM25Okapi
from openai import OpenAI

CHUNKS_PATH = Path("data/chunks.parquet")
MODEL = "meta-llama/llama-3.3-70b-instruct:free"

st.set_page_config(page_title="مساعد مراجعة الأدبيات - Llama", page_icon="📚", layout="centered")

# --- Helpers للبحث ---
def normalize_whitespace(t): return re.sub(r"\s+"," ",t).strip()
AR_DIAC = re.compile(r"[\u064B-\u0652\u0670\u0640]")
def normalize_ar(t):
    t = AR_DIAC.sub("",t)
    t = re.sub(r"[إأآ]","ا",t)
    t = re.sub(r"ى","ي",t)
    t = re.sub(r"[^\u0600-\u06FF\s0-9]"," ",t)
    return normalize_whitespace(t)
def normalize_en(t): return normalize_whitespace(re.sub(r"[^\w\s]"," ",t.lower()))
def detect_lang(t):
    ar = re.compile(r"[\u0600-\u06FF]")
    letters = re.findall(r"[^\W\d_]", t, re.UNICODE)
    if not letters: return "unknown"
    return "ar" if len(ar.findall(t))/len(letters) >= 0.15 else "en"
def normalize_lex(t): return normalize_ar(t) if detect_lang(t)=="ar" else normalize_en(t)
def simple_tokenize(t):
    t = normalize_lex(t)
    return re.findall(r"[a-z0-9]+|[\u0600-\u06FF]+", t.lower())

@st.cache_resource(show_spinner="بنجهز قاعدة المعرفة...")
def get_index():
    if not CHUNKS_PATH.exists():
        return None, f"ملف {CHUNKS_PATH} غير موجود على GitHub"
    df = pd.read_parquet(CHUNKS_PATH)
    tokenized = [simple_tokenize(t) for t in df["search_text"]]
    bm25 = BM25Okapi(tokenized)
    return (df, bm25), None

def build_context_package(query, df, bm25, k=10, max_chunks=5, budget=600):
    """نفس الاسم اللي في كودك القديم"""
    tok_q = simple_tokenize(query)
    scores = bm25.get_scores(tok_q)
    ranking = np.argsort(scores)[::-1][:k]
    cands = df.iloc[ranking].copy()
    cands["score"] = scores[ranking]
    top = cands["score"].max() if len(cands) else 0
    cands = cands[cands["score"] >= top*0.25].head(max_chunks)
    lines=[]; rows=[]; used=0
    for _, r in cands.iterrows():
        wc = len(r["chunk_text"].split())
        if used + wc > budget: continue
        lines.append(f"[Source: {r['title']} — {r['authors']} ({r['publication_year']})]\n{r['chunk_text']}")
        rows.append(r); used+=wc
    return {"context_text":"\n\n".join(lines), "selected_df": pd.DataFrame(rows), "used_words": used}

def get_openrouter_key():
    if "OPENROUTER_API_KEY" in st.secrets: return st.secrets["OPENROUTER_API_KEY"]
    if "API" in st.secrets: return st.secrets["API"]  # دعم اسم قديم
    return os.environ.get("OPENROUTER_API_KEY") or os.environ.get("API")

# --- UI ---
st.title("📚 مساعد مراجعة الأدبيات")
st.caption("RAG Chatbot - Section 16 - Llama 3.3 70B via OpenRouter")

(index_data, err) = get_index()
if err:
    st.error(err); st.stop()
df, bm25 = index_data
st.success(f"تم تحميل {len(df)} مقطع من {df['document_id'].nunique()} مصدر ✅")

openrouter_key = get_openrouter_key()
if not openrouter_key:
    st.error("⚠️ أضيفي OPENROUTER_API_KEY في Secrets بهذا الشكل:\n\nOPENROUTER_API_KEY = \"sk-or-v1-...\"")
    st.stop()

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=openrouter_key)

query = st.text_input("اكتبي سؤالك (عربي/إنجليزي):", placeholder="كيف تؤثر الرقمنة على كبار السن؟")
show_ctx = st.checkbox("إظهار النص المسترجع", False)

if query:
    with st.spinner("نبحث في المصادر ونولد الإجابة بـ Llama..."):
        package = build_context_package(query, df, bm25)
        context = package["context_text"]
        if not context:
            st.error("لم نجد مصادر كافية")
        else:
            try:
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": f"استخدم السياق فقط للإجابة. إذا لم يكن كافياً قل: لا توجد معلومات كافية في المصادر. اذكر المصدر (المؤلف، السنة). جاوب بنفس لغة السؤال.\n\nالسياق:\n{context}"},
                        {"role": "user", "content": query},
                    ],
                    temperature=0.3,
                )
                answer = response.choices[0].message.content
                st.markdown("### 💡 الإجابة (Llama 3.3)")
                st.write(answer)
                st.markdown("### 📖 المصادر")
                if len(package["selected_df"]):
                    st.dataframe(package["selected_df"][["title","authors","publication_year"]].drop_duplicates(), use_container_width=True, hide_index=True)
                if show_ctx:
                    st.text(context)
            except Exception as e:
                st.error(f"خطأ في OpenRouter: {e}")
                st.info("تأكدي أن المفتاح صحيح وأن عندك رصيد في OpenRouter")

st.divider()
st.caption("يعمل بـ meta-llama/llama-3.3-70b-instruct:free عبر OpenRouter - بدون Gemini")
