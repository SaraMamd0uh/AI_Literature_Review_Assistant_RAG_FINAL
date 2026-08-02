"""
PREMIUM UI - وصال WESAL - Llama فقط
منصة أثر الرقمنة على التفاعلات الاجتماعية
تصميم احترافي جداً - Section 16
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
BRAND_NAME_AR = "وصال"
BRAND_NAME_EN = "WESAL"
TAGLINE = "المساعد الذكي لاستكشاف أثر الرقمنة على التفاعلات الاجتماعية"

st.set_page_config(
    page_title=f"{BRAND_NAME_AR} | {BRAND_NAME_EN} - مساعد الأدبيات الرقمية",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== PREMIUM CSS =====
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@300;400;500;700;800&family=Inter:wght@400;600;700&display=swap');

* {font-family: 'Tajawal', 'Inter', sans-serif;}

.main-header {
    background: linear-gradient(135deg, #0F2167 0%, #1E3A8A 40%, #2EC4B6 100%);
    padding: 2rem 2.5rem;
    border-radius: 20px;
    color: white;
    margin-bottom: 1.5rem;
    box-shadow: 0 10px 30px rgba(15,33,103,0.25);
    position: relative;
    overflow: hidden;
}
.main-header::before {
    content: "";
    position: absolute;
    top: -50%;
    right: -20%;
    width: 60%;
    height: 200%;
    background: radial-gradient(circle, rgba(255,255,255,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.brand-title {
    font-size: 2.8rem;
    font-weight: 800;
    letter-spacing: -1px;
    margin-bottom: 0.2rem;
    display: flex;
    align-items: center;
    gap: 1rem;
}
.brand-subtitle {
    font-size: 1.1rem;
    font-weight: 300;
    opacity: 0.92;
    line-height: 1.6;
    max-width: 700px;
}
.stats-bar {
    display: flex;
    gap: 1rem;
    margin-top: 1.2rem;
    flex-wrap: wrap;
}
.stat-chip {
    background: rgba(255,255,255,0.15);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.2);
    padding: 0.5rem 1rem;
    border-radius: 50px;
    font-size: 0.85rem;
    font-weight: 500;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.chat-container {
    background: #ffffff;
    border-radius: 16px;
    border: 1px solid #E5E7EB;
    box-shadow: 0 4px 20px rgba(0,0,0,0.05);
    padding: 1.5rem;
}
.source-card {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-left: 4px solid #2EC4B6;
    padding: 0.8rem 1rem;
    border-radius: 10px;
    margin-bottom: 0.6rem;
    transition: all 0.2s;
}
.source-card:hover {
    border-left-color: #0F2167;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.answer-box {
    background: linear-gradient(180deg, #ffffff 0%, #F8FAFC 100%);
    border: 1px solid #E5E7EB;
    border-radius: 16px;
    padding: 1.5rem;
    line-height: 1.9;
    font-size: 1.05rem;
}
.stChatMessage {
    border-radius: 16px !important;
}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- Helpers ---
def normalize_ws(t): return re.sub(r"\s+"," ",t).strip()
AR_DIAC = re.compile(r"[\u064B-\u0652\u0670\u0640]")
def norm_ar(t):
    t = AR_DIAC.sub("",t)
    t = re.sub(r"[إأآ]","ا",t); t = re.sub(r"ى","ي",t)
    t = re.sub(r"[^\u0600-\u06FF\s0-9]"," ",t)
    return normalize_ws(t)
def norm_en(t): return normalize_ws(re.sub(r"[^\w\s]"," ",t.lower()))
def detect_lang(t):
    ar = re.compile(r"[\u0600-\u06FF]")
    letters = re.findall(r"[^\W\d_]", t, re.UNICODE)
    if not letters: return "unknown"
    return "ar" if len(ar.findall(t))/len(letters)>=0.15 else "en"
def norm_lex(t): return norm_ar(t) if detect_lang(t)=="ar" else norm_en(t)
def simple_tok(t):
    t = norm_lex(t)
    return re.findall(r"[a-z0-9]+|[\u0600-\u06FF]+", t.lower())

@st.cache_resource(show_spinner=False)
def get_index():
    if not CHUNKS_PATH.exists():
        return None, f"ملف {CHUNKS_PATH} غير موجود"
    df = pd.read_parquet(CHUNKS_PATH)
    tok = [simple_tok(t) for t in df["search_text"]]
    bm25 = BM25Okapi(tok)
    return (df, bm25), None

def build_context_package(query, df, bm25, k=12, max_chunks=5, budget=650):
    tok_q = simple_tok(query)
    scores = bm25.get_scores(tok_q)
    ranking = np.argsort(scores)[::-1][:k]
    cands = df.iloc[ranking].copy()
    cands["score"] = scores[ranking]
    top = cands["score"].max() if len(cands) else 0
    cands = cands[cands["score"] >= top*0.25].head(max_chunks)
    lines=[]; rows=[]; used=0
    for _, r in cands.iterrows():
        wc=len(r["chunk_text"].split())
        if used+wc>budget: continue
        lines.append(f"[Source: {r['title']} — {r['authors']} ({r['publication_year']})]\n{r['chunk_text']}")
        rows.append(r); used+=wc
    return {"context_text":"\n\n".join(lines), "selected_df": pd.DataFrame(rows), "used_words": used}

def get_key():
    if "OPENROUTER_API_KEY" in st.secrets: return st.secrets["OPENROUTER_API_KEY"]
    if "API" in st.secrets: return st.secrets["API"]
    return os.environ.get("OPENROUTER_API_KEY") or os.environ.get("API")

# --- SIDEBAR - BRAND STORY ---
with st.sidebar:
    # Logo
    if Path("logo.png").exists():
        st.image("logo.png", width=180)
    elif Path("data/../logo.png").exists():
        st.image("data/../logo.png", width=180)
    
    st.markdown(f"## {BRAND_NAME_AR} | {BRAND_NAME_EN}")
    st.caption(f"{TAGLINE}")
    st.divider()
    st.markdown("### 💡 عن المنصة")
    st.markdown("""
    **وصال** هي منصة RAG أكاديمية ذكية تحلل **أثر الرقمنة على التفاعلات الاجتماعية** 
    اعتماداً على 11 ورقة بحثية وكتاب متخصص.

    - 🔍 **بحث هجين:** دلالي + لفظي (BM25)
    - 🌐 **ثنائية اللغة:** عربي / إنجليزي
    - 🤖 **Llama 3.3 70B** عبر OpenRouter
    - 📚 **Section 16** - هيكل الدكتور
    """)
    st.divider()
    st.markdown("### 📊 الإحصائيات")
    if CHUNKS_PATH.exists():
        try:
            df_tmp = pd.read_parquet(CHUNKS_PATH)
            c1,c2 = st.columns(2)
            c1.metric("المصادر", f"{df_tmp['document_id'].nunique()}")
            c2.metric("المقاطع", f"{len(df_tmp)}")
        except: pass
    
    st.divider()
    st.markdown("### ⚙️ كيف تعمل؟")
    st.markdown("""
    1.  تكتب سؤال بلغة عربية أو إنجليزية
    2.  النظام يسترجع أكثر 5 مقاطع صلة
    3.  Llama يولد إجابة موثقة بالمصادر
    """)
    st.divider()
    st.caption("مشروع أكاديمي - الإجابات مولدة تلقائياً - راجع المصادر الأصلية")

# --- MAIN HEADER ---
logo_col, title_col = st.columns([1,5])
with logo_col:
    if Path("logo.png").exists():
        st.image("logo.png", width=120)
with title_col:
    st.markdown(f"""
    <div class="main-header">
        <div class="brand-title">
            <span>📖 {BRAND_NAME_AR}</span> 
            <span style="font-weight:300; opacity:0.8">|</span> 
            <span style="font-size:2rem">{BRAND_NAME_EN}</span>
        </div>
        <div class="brand-subtitle">
            {TAGLINE} — نظام RAG ثنائي اللغة على 11 ورقة بحثية وكتاب متخصص في علم الاجتماع الرقمي
            <br>نموذج: <code>meta-llama/llama-3.3-70b-instruct:free</code> عبر OpenRouter
        </div>
        <div class="stats-bar">
            <div class="stat-chip">📚 12 مصدر أكاديمي</div>
            <div class="stat-chip">🧠 RAG + BM25</div>
            <div class="stat-chip">🌐 عربي / إنجليزي</div>
            <div class="stat-chip">⚡ Llama 3.3 70B Free</div>
            <div class="stat-chip">🏛️ Section 16</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- Load Index ---
(index_data, err) = get_index()
if err:
    st.error(err)
    st.stop()
df, bm25 = index_data

# --- CHAT HISTORY ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "مرحباً! أنا **وصال**، مساعدك الذكي لتحليل أثر الرقمنة على المجتمع. 👋\n\nاسألني مثلاً:\n- كيف تؤثر الرقمنة على كبار السن؟\n- What is the impact of digital devices on face-to-face interaction in Cairo coffeehouses?\n- ما علاقة الرقمنة بالهوية الاجتماعية؟"}
    ]

# Display history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input
if prompt := st.chat_input("اكتب سؤالك هنا... (عربي أو إنجليزي)"):
    # User message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Assistant
    with st.chat_message("assistant"):
        with st.spinner("وصال تبحث في 12 مصدر..."):
            key = get_key()
            if not key:
                st.error('أضيفي OPENROUTER_API_KEY في Secrets')
                st.stop()
            pkg = build_context_package(prompt, df, bm25)
            ctx = pkg["context_text"]
            if not ctx:
                ans = "لم أجد مصادر كافية لهذا السؤال. جرب صياغة أخرى."
                st.write(ans)
            else:
                try:
                    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)
                    resp = client.chat.completions.create(
                        model=MODEL,
                        messages=[
                            {"role": "system", "content": f"أنت {BRAND_NAME_AR}، مساعد أكاديمي متخصص في السوسيولوجيا الرقمية. القواعد:\n1- استخدم السياق فقط\n2- إذا غير كاف قل: لا توجد معلومات كافية في المصادر\n3- اذكر المصدر (المؤلف، السنة) بعد كل معلومة\n4- جاوب بنفس لغة السؤال\n\nالسياق:\n{ctx}"},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.25,
                    )
                    ans = resp.choices[0].message.content
                    st.markdown(f'<div class="answer-box">{ans}</div>', unsafe_allow_html=True)
                    
                    if len(pkg["selected_df"]):
                        st.markdown("#### 📖 المصادر المستخدمة")
                        sources = pkg["selected_df"][["title","authors","publication_year"]].drop_duplicates()
                        for _, s in sources.iterrows():
                            st.markdown(f'<div class="source-card"><b>{s["title"]}</b><br>👤 {s["authors"]} ({s["publication_year"]})</div>', unsafe_allow_html=True)
                    
                    with st.expander("🔎 عرض النص المسترجع"):
                        st.text(ctx)
                        
                except Exception as e:
                    ans = f"خطأ: {e}"
                    st.error(ans)
            # Save to history
            if 'ans' in locals():
                st.session_state.messages.append({"role": "assistant", "content": ans})

st.divider()
st.markdown(f"<center style='color:#64748B; font-size:0.85rem'>© 2025 {BRAND_NAME_AR} | {BRAND_NAME_EN} - منصة السوسيولوجيا الرقمية - مبني بـ Llama 3.3 70B 🦙</center>", unsafe_allow_html=True)
