"""
WESAL v2 - ULTRA PREMIUM DESIGN
تصميم عالمي المستوى - مثل Perplexity + Linear
Llama فقط - RTL صحيح - مينيمال
"""
import os, re
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
from rank_bm25 import BM25Okapi
from openai import OpenAI

CHUNKS_PATH = Path("data/chunks.parquet")
MODEL = "meta-llama/llama-3.3-70b-instruct:free"

st.set_page_config(
    page_title="وصال — مساعد الأدبيات الرقمية",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ===== WORLD-CLASS CSS =====
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700&family=Inter:wght@400;500;600&display=swap');

:root {
    --bg: #FFFFFF;
    --bg-soft: #FAFAF9;
    --border: #E7E5E4;
    --text: #1C1917;
    --text-muted: #78716C;
    --accent: #0F2167;
    --accent-soft: #EFF6FF;
}

html, body, [class*="css"] {
    font-family: 'Tajawal', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background: var(--bg);
    color: var(--text);
}

#MainMenu, footer, header {visibility: hidden;}

/* Top Nav - Minimal like Linear */
.top-nav {
    position: sticky;
    top: 0;
    z-index: 10;
    background: rgba(255,255,255,0.85);
    backdrop-filter: blur(12px);
    border-bottom: 1px solid var(--border);
    padding: 12px 0;
    margin: -1rem -1rem 0 -1rem;
    padding-left: 1.5rem;
    padding-right: 1.5rem;
}
.nav-inner {
    max-width: 780px;
    margin: 0 auto;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.brand {
    display: flex;
    align-items: center;
    gap: 10px;
    font-weight: 700;
    font-size: 1.05rem;
    letter-spacing: -0.02em;
}
.brand-mark {
    width: 32px;
    height: 32px;
    background: var(--accent);
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-weight: 700;
    font-size: 14px;
}
.nav-meta {
    font-size: 0.75rem;
    color: var(--text-muted);
    background: var(--bg-soft);
    border: 1px solid var(--border);
    padding: 4px 10px;
    border-radius: 100px;
    font-family: 'Inter', monospace;
}

/* Centered container like Perplexity */
.main-container {
    max-width: 760px;
    margin: 0 auto;
    padding-top: 2rem;
}

/* Hero - only when empty */
.hero {
    text-align: center;
    padding: 3.5rem 1rem 2rem 1rem;
}
.hero-icon {
    width: 64px;
    height: 64px;
    margin: 0 auto 1.2rem auto;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 16px;
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
}
.hero h1 {
    font-size: 2.2rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    margin: 0 0 0.6rem 0;
}
.hero p {
    color: var(--text-muted);
    font-size: 1.05rem;
    line-height: 1.7;
    max-width: 520px;
    margin: 0 auto 2rem auto;
}
.examples {
    display: grid;
    grid-template-columns: 1fr;
    gap: 0.6rem;
    max-width: 520px;
    margin: 0 auto;
    text-align: right;
}
.example-card {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 14px 16px;
    font-size: 0.92rem;
    line-height: 1.6;
    cursor: pointer;
    transition: all 0.15s ease;
    text-align: right;
    direction: rtl;
}
.example-card:hover {
    border-color: var(--accent);
    background: var(--accent-soft);
    transform: translateY(-1px);
}

/* Chat messages */
.stChatMessage {
    background: transparent !important;
    border: none !important;
    padding: 1.2rem 0 !important;
}

.user-bubble {
    background: var(--bg-soft);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 14px 18px;
    max-width: 85%;
    margin-left: auto;
    line-height: 1.8;
    font-size: 0.96rem;
}
.assistant-content {
    line-height: 1.9;
    font-size: 1.02rem;
    color: var(--text);
}
.assistant-content.rtl {
    direction: rtl;
    text-align: right;
}
.assistant-content.ltr {
    direction: ltr;
    text-align: left;
}

/* Sources - minimal */
.sources {
    margin-top: 1.5rem;
    border-top: 1px solid var(--border);
    padding-top: 1rem;
}
.source-item {
    font-size: 0.85rem;
    color: var(--text-muted);
    padding: 6px 0;
    border-bottom: 1px dashed #F5F5F4;
    display: flex;
    gap: 8px;
}
.source-item b {
    color: var(--text);
    font-weight: 600;
}

/* Input */
.stChatInput {
    max-width: 760px;
    margin: 0 auto;
}
.stChatInput > div {
    border-radius: 24px !important;
    border: 1px solid var(--border) !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.06) !important;
    background: white !important;
}

/* Hide extra stuff */
div[data-testid="stToolbar"] {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- Helpers ---
def norm_ws(t): return re.sub(r"\s+"," ",t).strip()
AR_DIAC = re.compile(r"[\u064B-\u0652\u0670\u0640]")
def norm_ar(t):
    t = AR_DIAC.sub("",t)
    t = re.sub(r"[إأآ]","ا",t); t = re.sub(r"ى","ي",t)
    t = re.sub(r"[^\u0600-\u06FF\s0-9]"," ",t)
    return norm_ws(t)
def norm_en(t): return norm_ws(re.sub(r"[^\w\s]"," ",t.lower()))
def detect_lang(t):
    ar = re.compile(r"[\u0600-\u06FF]")
    letters = re.findall(r"[^\W\d_]", t, re.UNICODE)
    if not letters: return "en"
    return "ar" if len(ar.findall(t))/len(letters)>=0.15 else "en"
def norm_lex(t): return norm_ar(t) if detect_lang(t)=="ar" else norm_en(t)
def tok(t):
    t = norm_lex(t)
    return re.findall(r"[a-z0-9]+|[\u0600-\u06FF]+", t.lower())

@st.cache_resource(show_spinner=False)
def get_index():
    if not CHUNKS_PATH.exists():
        return None, "no_data"
    df = pd.read_parquet(CHUNKS_PATH)
    tokenized = [tok(t) for t in df["search_text"]]
    bm25 = BM25Okapi(tokenized)
    return (df, bm25), None

def build_pkg(q, df, bm25):
    sc = bm25.get_scores(tok(q))
    rank = np.argsort(sc)[::-1][:10]
    cands = df.iloc[rank].copy()
    cands["score"]=sc[rank]
    top=cands["score"].max() if len(cands) else 0
    cands=cands[cands["score"]>=top*0.25].head(4)
    lines=[]; rows=[]; used=0
    for _,r in cands.iterrows():
        if used+len(r["chunk_text"].split())>620: continue
        lines.append(f"[{r['title']} — {r['authors']} ({r['publication_year']})]\n{r['chunk_text']}")
        rows.append(r); used+=len(r["chunk_text"].split())
    return {"ctx":"\n\n".join(lines), "df":pd.DataFrame(rows)}

def get_key():
    if "OPENROUTER_API_KEY" in st.secrets: return st.secrets["OPENROUTER_API_KEY"]
    if "API" in st.secrets: return st.secrets["API"]
    return os.environ.get("OPENROUTER_API_KEY")

# --- TOP NAV ---
st.markdown("""
<div class="top-nav">
  <div class="nav-inner">
    <div class="brand">
      <img src="https://raw.githubusercontent.com/SaraMamd0uh/AI_Literature_Review_Assistant_RAG_FINAL/main/logo.png" style="width:32px; height:32px; border-radius:8px; object-fit:contain;" onerror="this.style.display='none'">
      <span style="margin-right:8px;">وصال</span>
      <span style="color:#A8A29E; font-weight:400; margin-right:8px; font-size:0.9em;">أثر الرقمنة</span>
    </div>
    <div class="nav-meta">Llama 3.3 • 12 مصدر</div>
  </div>
</div>
""", unsafe_allow_html=True)

# --- MAIN CONTAINER ---
st.markdown('<div class="main-container">', unsafe_allow_html=True)

(index_data, err) = get_index()
if err:
    st.error("فولدر data/chunks.parquet غير موجود"); st.stop()
df, bm25 = index_data

if "messages" not in st.session_state:
    st.session_state.messages = []

# HERO - only if no messages
if len(st.session_state.messages)==0:
    # Logo icon - keeping original logo as requested
    col1,col2,col3 = st.columns([1,1,1])
    with col2:
        if Path("logo.png").exists():
            st.image("logo.png", width=140)
        elif Path("wesal_icon.png").exists():
            st.image("wesal_icon.png", width=120)
        else:
            st.markdown('<div class="hero-icon">و</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="hero">
        <h1>وصال</h1>
        <p>مساعد أكاديمي يجيب من 11 ورقة بحثية وكتاب عن أثر الرقمنة على التفاعلات الاجتماعية. اسأل بالعربي أو الإنجليزي.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Example cards - clickable
    st.markdown('<div class="examples">', unsafe_allow_html=True)
    examples = [
        "كيف تؤثر الرقمنة على كبار السن وعلاقاتهم الاجتماعية؟",
        "What is the impact of digital devices on face-to-face interaction in Cairo coffeehouses?",
        "ما علاقة الرقمنة بالهوية الاجتماعية وتكوين المجموعات على الإنترنت؟"
    ]
    for ex in examples:
        if st.button(ex, key=f"ex_{ex[:10]}", use_container_width=True):
            st.session_state.messages.append({"role":"user","content":ex})
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
else:
    # Show chat
    for m in st.session_state.messages:
        lang = detect_lang(m["content"])
        cls = "rtl" if lang=="ar" else "ltr"
        with st.chat_message(m["role"]):
            if m["role"]=="user":
                st.markdown(f'<div class="user-bubble" dir="{cls}">{m["content"]}</div>', unsafe_allow_html=True)
            else:
                # check if message has sources metadata
                text = m["content"]
                # Split answer and sources if combined
                st.markdown(f'<div class="assistant-content {cls}">{text}</div>', unsafe_allow_html=True)
                if "sources" in m and len(m["sources"]):
                    with st.expander(f"المصادر · {len(m['sources'])}", expanded=False):
                        for s in m["sources"]:
                            st.markdown(f'<div class="source-item">📄 <b>{s["title"][:70]}...</b> — {s["authors"][:40]} ({s["publication_year"]})</div>', unsafe_allow_html=True)

# Input
if prompt := st.chat_input("اسأل عن أثر الرقمنة... / Ask about digitalization..."):
    lang = detect_lang(prompt)
    st.session_state.messages.append({"role":"user","content":prompt})
    with st.chat_message("user"):
        st.markdown(f'<div class="user-bubble" dir="{"rtl" if lang=="ar" else "ltr"}">{prompt}</div>', unsafe_allow_html=True)
    
    with st.chat_message("assistant"):
        with st.spinner(""):
            key = get_key()
            if not key:
                st.error("أضيفي OPENROUTER_API_KEY في Secrets")
                st.stop()
            pkg = build_pkg(prompt, df, bm25)
            if not pkg["ctx"]:
                ans="لم أجد مصادر كافية. جرب صياغة أخرى."
                st.markdown(f'<div class="assistant-content rtl">{ans}</div>', unsafe_allow_html=True)
            else:
                try:
                    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)
                    sys_prompt = f"أنت وصال، مساعد أكاديمي متخصص في السوسيولوجيا الرقمية. استخدم السياق فقط. إذا غير كاف قل: لا توجد معلومات كافية في المصادر. اذكر المصدر (المؤلف، السنة). جاوب بنفس لغة السؤال.\n\nالسياق:\n{pkg['ctx']}"
                    resp = client.chat.completions.create(
                        model=MODEL,
                        messages=[
                            {"role":"system","content":sys_prompt},
                            {"role":"user","content":prompt}
                        ],
                        temperature=0.2,
                    )
                    ans = resp.choices[0].message.content
                    cls = "rtl" if detect_lang(ans)=="ar" else "ltr"
                    st.markdown(f'<div class="assistant-content {cls}">{ans}</div>', unsafe_allow_html=True)
                    
                    src_list = pkg["df"][["title","authors","publication_year"]].drop_duplicates().to_dict("records") if len(pkg["df"]) else []
                    if src_list:
                        with st.expander(f"المصادر · {len(src_list)}", expanded=False):
                            for s in src_list:
                                st.markdown(f'<div class="source-item">📄 <b>{s["title"][:80]}</b><br><span style="margin-right:20px">{s["authors"][:50]} ({s["publication_year"]})</span></div>', unsafe_allow_html=True)
                    
                    # Save with sources
                    st.session_state.messages.append({"role":"assistant","content":ans,"sources":src_list})
                except Exception as e:
                    st.error(f"خطأ: {e}")
            st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

# Minimal footer
st.markdown("<div style='max-width:760px; margin:3rem auto 0 auto; text-align:center; color:#A8A29E; font-size:0.78rem; border-top:1px solid #F5F5F4; padding-top:1.2rem;'>وصال — مشروع أكاديمي • Llama 3.3 70B via OpenRouter • RAG على 11 ورقة وكتاب</div>", unsafe_allow_html=True)
