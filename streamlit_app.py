"""
WESAL v3 - ULTRA PREMIUM PROFESSIONAL DESIGN
تصميم عالمي احترافي يجذب الشركات
Inspired by: Perplexity AI + Linear + Notion + Vercel
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
    page_title="وصال — مساعد الأبحاث الأكاديمية الذكي",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ===== ULTRA PREMIUM CSS =====
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@300;400;500;700;800;900&family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg-primary: #09090B;
    --bg-secondary: #18181B;
    --bg-tertiary: #27272A;
    --bg-card: rgba(255,255,255,0.03);
    --bg-card-hover: rgba(255,255,255,0.06);
    --bg-glass: rgba(9,9,11,0.8);
    --border: rgba(255,255,255,0.06);
    --border-hover: rgba(255,255,255,0.12);
    --border-active: rgba(139,92,246,0.3);
    --text-primary: #FAFAFA;
    --text-secondary: #A1A1AA;
    --text-tertiary: #71717A;
    --accent: #8B5CF6;
    --accent-soft: rgba(139,92,246,0.1);
    --accent-glow: rgba(139,92,246,0.15);
    --gradient-1: linear-gradient(135deg, #8B5CF6, #06B6D4);
    --gradient-2: linear-gradient(135deg, #8B5CF6, #EC4899);
    --gradient-3: linear-gradient(135deg, #06B6D4, #8B5CF6, #EC4899);
    --shadow-sm: 0 1px 2px rgba(0,0,0,0.3);
    --shadow-md: 0 4px 12px rgba(0,0,0,0.4);
    --shadow-lg: 0 8px 32px rgba(0,0,0,0.5);
    --shadow-glow: 0 0 40px rgba(139,92,246,0.15);
    --radius-sm: 8px;
    --radius-md: 12px;
    --radius-lg: 16px;
    --radius-xl: 24px;
}

/* Reset & Base */
html, body, [class*="css"] {
    font-family: 'Tajawal', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}
.stApp {
    background: var(--bg-primary) !important;
}

/* Hide Streamlit defaults */
#MainMenu, footer, header,
div[data-testid="stToolbar"],
div[data-testid="stDecoration"],
div[data-testid="stStatusWidget"] {
    visibility: hidden !important;
    display: none !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--bg-tertiary); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-tertiary); }

/* ========== TOP NAVIGATION ========== */
.top-nav {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    z-index: 1000;
    background: var(--bg-glass);
    backdrop-filter: blur(20px) saturate(180%);
    -webkit-backdrop-filter: blur(20px) saturate(180%);
    border-bottom: 1px solid var(--border);
    padding: 0 2rem;
    height: 60px;
    display: flex;
    align-items: center;
}
.nav-inner {
    max-width: 1200px;
    margin: 0 auto;
    width: 100%;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.nav-brand {
    display: flex;
    align-items: center;
    gap: 12px;
    text-decoration: none;
}
.nav-logo {
    width: 36px;
    height: 36px;
    border-radius: 10px;
    background: var(--gradient-1);
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 800;
    font-size: 16px;
    color: white;
    box-shadow: var(--shadow-glow);
    position: relative;
    overflow: hidden;
}
.nav-logo::before {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(135deg, rgba(255,255,255,0.2), transparent);
    border-radius: inherit;
}
.nav-logo img {
    width: 36px;
    height: 36px;
    border-radius: 10px;
    object-fit: contain;
}
.nav-title {
    font-weight: 700;
    font-size: 1.15rem;
    color: var(--text-primary);
    letter-spacing: -0.02em;
}
.nav-subtitle {
    font-size: 0.72rem;
    color: var(--text-tertiary);
    font-weight: 400;
    margin-top: -2px;
}
.nav-right {
    display: flex;
    align-items: center;
    gap: 12px;
}
.nav-badge {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 0.72rem;
    color: var(--text-tertiary);
    background: var(--bg-card);
    border: 1px solid var(--border);
    padding: 5px 12px;
    border-radius: 100px;
    font-family: 'JetBrains Mono', monospace;
}
.nav-badge .dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #22C55E;
    animation: pulse-dot 2s ease-in-out infinite;
}
@keyframes pulse-dot {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}
.nav-status {
    font-size: 0.72rem;
    color: var(--accent);
    background: var(--accent-soft);
    border: 1px solid var(--border-active);
    padding: 5px 12px;
    border-radius: 100px;
    font-weight: 500;
}

/* Spacer for fixed nav */
.nav-spacer { height: 70px; }

/* ========== HERO SECTION ========== */
.hero-section {
    max-width: 720px;
    margin: 0 auto;
    padding: 4rem 1rem 2rem 1rem;
    text-align: center;
    position: relative;
}

/* Ambient glow behind hero */
.hero-glow {
    position: absolute;
    top: -80px;
    left: 50%;
    transform: translateX(-50%);
    width: 600px;
    height: 400px;
    background: radial-gradient(ellipse, rgba(139,92,246,0.08) 0%, rgba(6,182,212,0.04) 40%, transparent 70%);
    pointer-events: none;
    z-index: 0;
}

.hero-logo-wrap {
    position: relative;
    z-index: 1;
    margin-bottom: 2rem;
}
.hero-logo-outer {
    width: 88px;
    height: 88px;
    margin: 0 auto;
    border-radius: 22px;
    background: var(--gradient-1);
    padding: 2px;
    box-shadow: var(--shadow-glow), 0 0 80px rgba(139,92,246,0.1);
    animation: float 6s ease-in-out infinite;
}
@keyframes float {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-8px); }
}
.hero-logo-inner {
    width: 100%;
    height: 100%;
    border-radius: 20px;
    background: var(--bg-primary);
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
}
.hero-logo-inner img {
    width: 70px;
    height: 70px;
    object-fit: contain;
    filter: brightness(1.1);
}
.hero-logo-fallback {
    font-size: 2.2rem;
    font-weight: 900;
    background: var(--gradient-1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.hero-title {
    position: relative;
    z-index: 1;
    font-size: 3.2rem;
    font-weight: 900;
    letter-spacing: -0.04em;
    line-height: 1.1;
    margin-bottom: 0.3rem;
    background: linear-gradient(180deg, #FAFAFA 0%, #A1A1AA 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-tagline {
    position: relative;
    z-index: 1;
    font-size: 1.1rem;
    font-weight: 500;
    color: var(--accent);
    margin-bottom: 1rem;
    letter-spacing: 0.05em;
}
.hero-desc {
    position: relative;
    z-index: 1;
    font-size: 1.05rem;
    color: var(--text-secondary);
    line-height: 1.8;
    max-width: 520px;
    margin: 0 auto 2.5rem auto;
}

/* Stats row */
.stats-row {
    display: flex;
    justify-content: center;
    gap: 2rem;
    margin-bottom: 3rem;
    position: relative;
    z-index: 1;
}
.stat-item {
    text-align: center;
}
.stat-num {
    font-size: 1.6rem;
    font-weight: 800;
    background: var(--gradient-1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-family: 'JetBrains Mono', monospace;
}
.stat-label {
    font-size: 0.75rem;
    color: var(--text-tertiary);
    margin-top: 2px;
}

/* ========== EXAMPLE CARDS ========== */
.examples-grid {
    display: grid;
    grid-template-columns: 1fr;
    gap: 10px;
    max-width: 560px;
    margin: 0 auto;
    position: relative;
    z-index: 1;
}
.example-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 16px 20px;
    cursor: pointer;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
    text-align: right;
    direction: rtl;
    display: flex;
    align-items: center;
    gap: 12px;
    position: relative;
    overflow: hidden;
}
.example-card::before {
    content: '';
    position: absolute;
    inset: 0;
    background: var(--gradient-1);
    opacity: 0;
    transition: opacity 0.2s;
}
.example-card:hover {
    border-color: var(--border-active);
    background: var(--bg-card-hover);
    transform: translateY(-2px);
    box-shadow: var(--shadow-glow);
}
.example-card:hover::before {
    opacity: 0.03;
}
.example-icon {
    font-size: 1.3rem;
    flex-shrink: 0;
}
.example-text {
    font-size: 0.92rem;
    line-height: 1.7;
    color: var(--text-secondary);
    position: relative;
    z-index: 1;
}

/* ========== CHAT AREA ========== */
.chat-container {
    max-width: 760px;
    margin: 0 auto;
    padding: 1rem 0;
}

/* User message */
.user-msg {
    display: flex;
    justify-content: flex-end;
    margin-bottom: 1.5rem;
}
.user-bubble {
    background: var(--accent-soft);
    border: 1px solid var(--border-active);
    border-radius: 18px 18px 4px 18px;
    padding: 14px 20px;
    max-width: 80%;
    line-height: 1.8;
    font-size: 0.96rem;
    color: var(--text-primary);
    position: relative;
}
.user-bubble.rtl {
    direction: rtl;
    text-align: right;
    border-radius: 18px 18px 18px 4px;
}

/* Assistant message */
.assistant-msg {
    margin-bottom: 2rem;
}
.assistant-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 12px;
}
.assistant-avatar {
    width: 28px;
    height: 28px;
    border-radius: 8px;
    background: var(--gradient-1);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 12px;
    font-weight: 700;
    color: white;
    flex-shrink: 0;
}
.assistant-name {
    font-size: 0.82rem;
    font-weight: 600;
    color: var(--text-secondary);
}
.assistant-content {
    line-height: 2;
    font-size: 1rem;
    color: var(--text-primary);
    padding-right: 38px;
}
.assistant-content.rtl {
    direction: rtl;
    text-align: right;
    padding-right: 0;
    padding-left: 38px;
}
.assistant-content.ltr {
    direction: ltr;
    text-align: left;
}

/* Sources panel */
.sources-panel {
    margin-top: 1.2rem;
    margin-right: 38px;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    overflow: hidden;
}
.sources-panel.rtl {
    margin-right: 0;
    margin-left: 38px;
}
.sources-header {
    padding: 10px 16px;
    font-size: 0.78rem;
    font-weight: 600;
    color: var(--text-tertiary);
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 6px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.source-row {
    padding: 10px 16px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: flex-start;
    gap: 10px;
    font-size: 0.82rem;
    transition: background 0.15s;
}
.source-row:last-child { border-bottom: none; }
.source-row:hover { background: var(--bg-card-hover); }
.source-num {
    width: 20px;
    height: 20px;
    border-radius: 6px;
    background: var(--accent-soft);
    color: var(--accent);
    font-size: 0.68rem;
    font-weight: 700;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    font-family: 'JetBrains Mono', monospace;
}
.source-info {
    flex: 1;
}
.source-title {
    font-weight: 600;
    color: var(--text-primary);
    line-height: 1.5;
    margin-bottom: 2px;
}
.source-meta {
    color: var(--text-tertiary);
    font-size: 0.75rem;
}

/* ========== INPUT AREA ========== */
.stChatInput {
    max-width: 760px;
    margin: 0 auto;
}
.stChatInput > div {
    border-radius: 20px !important;
    border: 1px solid var(--border) !important;
    background: var(--bg-secondary) !important;
    box-shadow: var(--shadow-md), 0 0 40px rgba(139,92,246,0.05) !important;
    transition: all 0.2s !important;
}
.stChatInput > div:focus-within {
    border-color: var(--border-active) !important;
    box-shadow: var(--shadow-lg), var(--shadow-glow) !important;
}
.stChatInput textarea {
    color: var(--text-primary) !important;
    font-family: 'Tajawal', sans-serif !important;
    font-size: 0.95rem !important;
}
.stChatInput textarea::placeholder {
    color: var(--text-tertiary) !important;
}

/* Chat message overrides */
.stChatMessage {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    max-width: 760px;
    margin-left: auto;
    margin-right: auto;
}

/* Loading animation */
.loading-dots {
    display: flex;
    gap: 4px;
    padding: 8px 0;
}
.loading-dots span {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--accent);
    animation: bounce 1.4s ease-in-out infinite;
}
.loading-dots span:nth-child(2) { animation-delay: 0.2s; }
.loading-dots span:nth-child(3) { animation-delay: 0.4s; }
@keyframes bounce {
    0%, 80%, 100% { transform: scale(0.6); opacity: 0.4; }
    40% { transform: scale(1); opacity: 1; }
}

/* ========== FOOTER ========== */
.footer {
    max-width: 760px;
    margin: 4rem auto 1rem auto;
    text-align: center;
    padding: 1.5rem 0;
    border-top: 1px solid var(--border);
}
.footer-text {
    font-size: 0.72rem;
    color: var(--text-tertiary);
    line-height: 1.8;
}
.footer-brand {
    font-weight: 700;
    background: var(--gradient-1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

/* Streamlit button overrides */
.stButton > button {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-secondary) !important;
    border-radius: var(--radius-md) !important;
    padding: 14px 20px !important;
    text-align: right !important;
    direction: rtl !important;
    font-family: 'Tajawal', sans-serif !important;
    font-size: 0.92rem !important;
    line-height: 1.7 !important;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    width: 100% !important;
}
.stButton > button:hover {
    background: var(--bg-card-hover) !important;
    border-color: var(--border-active) !important;
    color: var(--text-primary) !important;
    transform: translateY(-2px) !important;
    box-shadow: var(--shadow-glow) !important;
}

/* Expander */
.streamlit-expanderHeader {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    color: var(--text-secondary) !important;
    font-size: 0.85rem !important;
}
.streamlit-expanderContent {
    background: var(--bg-secondary) !important;
    border: 1px solid var(--border) !important;
    border-top: none !important;
}

/* Responsive */
@media (max-width: 768px) {
    .hero-title { font-size: 2.2rem; }
    .stats-row { gap: 1.2rem; }
    .stat-num { font-size: 1.2rem; }
    .top-nav { padding: 0 1rem; }
    .nav-subtitle { display: none; }
}
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
def tok(t):
    t = norm_ar(t) if detect_lang(t)=="ar" else norm_en(t)
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

# ===== TOP NAVIGATION =====
logo_url = "https://raw.githubusercontent.com/SaraMamd0uh/AI_Literature_Review_Assistant_RAG_FINAL/main/logo.png"
st.markdown(f"""
<div class="top-nav">
  <div class="nav-inner">
    <div class="nav-brand">
      <div class="nav-logo">
        <img src="{logo_url}" onerror="this.style.display='none'; this.parentElement.innerHTML='و';">
      </div>
      <div>
        <div class="nav-title">وصال</div>
        <div class="nav-subtitle">WESAL Research Assistant</div>
      </div>
    </div>
    <div class="nav-right">
      <div class="nav-badge">
        <span class="dot"></span>
        <span>Llama 3.3 · 70B</span>
      </div>
      <div class="nav-status">⚡ RAG Active</div>
    </div>
  </div>
</div>
<div class="nav-spacer"></div>
""", unsafe_allow_html=True)

# ===== LOAD DATA =====
(index_data, err) = get_index()
if err:
    st.error("⚠️ data/chunks.parquet not found")
    st.stop()
df, bm25 = index_data

if "messages" not in st.session_state:
    st.session_state.messages = []

# ===== HERO (only when empty) =====
if len(st.session_state.messages) == 0:
    st.markdown(f"""
    <div class="hero-section">
      <div class="hero-glow"></div>
      <div class="hero-logo-wrap">
        <div class="hero-logo-outer">
          <div class="hero-logo-inner">
            <img src="{logo_url}" onerror="this.style.display='none'; this.parentElement.innerHTML='<div class=hero-logo-fallback>و</div>';">
          </div>
        </div>
      </div>
      <div class="hero-title">وصال</div>
      <div class="hero-tagline">WESAL · Academic Research Intelligence</div>
      <div class="hero-desc">
        مساعد بحثي ذكي مدعوم بالذكاء الاصطناعي، يحلل ويجيب من
        <strong>11 ورقة بحثية وكتاب</strong> عن أثر الرقمنة على التفاعلات الاجتماعية
      </div>
      <div class="stats-row">
        <div class="stat-item">
          <div class="stat-num">12</div>
          <div class="stat-label">مصدر أكاديمي</div>
        </div>
        <div class="stat-item">
          <div class="stat-num">RAG</div>
          <div class="stat-label">تقنية الاسترجاع</div>
        </div>
        <div class="stat-item">
          <div class="stat-num">70B</div>
          <div class="stat-label">Llama 3.3</div>
        </div>
        <div class="stat-item">
          <div class="stat-num">AR/EN</div>
          <div class="stat-label">ثنائي اللغة</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Example questions
    examples = [
        ("🏠", "كيف تؤثر الرقمنة على كبار السن وعلاقاتهم الاجتماعية؟"),
        ("☕", "What is the impact of digital devices on face-to-face interaction in Cairo coffeehouses?"),
        ("🌐", "ما علاقة الرقمنة بالهوية الاجتماعية وتكوين المجموعات على الإنترنت؟"),
    ]

    cols = st.columns(1)
    for icon, ex in examples:
        if st.button(f"{icon}  {ex}", key=f"ex_{ex[:15]}", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": ex})
            st.rerun()

else:
    # ===== CHAT MESSAGES =====
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    for m in st.session_state.messages:
        lang = detect_lang(m["content"])
        is_rtl = lang == "ar"
        
        if m["role"] == "user":
            cls = "rtl" if is_rtl else ""
            with st.chat_message("user"):
                st.markdown(
                    f'<div class="user-bubble {cls}">{m["content"]}</div>',
                    unsafe_allow_html=True
                )
        else:
            with st.chat_message("assistant"):
                ans_lang = detect_lang(m["content"])
                cls = "rtl" if ans_lang == "ar" else "ltr"
                
                st.markdown(f"""
                <div class="assistant-msg">
                  <div class="assistant-content {cls}">{m["content"]}</div>
                </div>
                """, unsafe_allow_html=True)
                
                if "sources" in m and m["sources"]:
                    src_cls = "rtl" if ans_lang == "ar" else ""
                    html = f'<div class="sources-panel {src_cls}"><div class="sources-header">📚 المصادر · SOURCES ({len(m["sources"])})</div>'
                    for i, s in enumerate(m["sources"], 1):
                        html += f"""
                        <div class="source-row">
                          <div class="source-num">{i}</div>
                          <div class="source-info">
                            <div class="source-title">{s.get('title','')[:90]}</div>
                            <div class="source-meta">{s.get('authors','')[:50]} · {s.get('publication_year','')}</div>
                          </div>
                        </div>"""
                    html += '</div>'
                    st.markdown(html, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# ===== INPUT =====
if prompt := st.chat_input("اسأل عن أثر الرقمنة... / Ask about digitalization..."):
    lang = detect_lang(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        cls = "rtl" if lang == "ar" else ""
        st.markdown(f'<div class="user-bubble {cls}">{prompt}</div>', unsafe_allow_html=True)
    
    with st.chat_message("assistant"):
        with st.spinner(""):
            key = get_key()
            if not key:
                st.error("⚠️ Add OPENROUTER_API_KEY to Secrets")
                st.stop()
            
            pkg = build_pkg(prompt, df, bm25)
            
            if not pkg["ctx"]:
                ans = "لم أجد مصادر كافية في قاعدة البيانات. جرب صياغة مختلفة للسؤال."
                st.markdown(f'<div class="assistant-content rtl">{ans}</div>', unsafe_allow_html=True)
                st.session_state.messages.append({"role": "assistant", "content": ans, "sources": []})
            else:
                try:
                    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)
                    
                    sys_prompt = (
                        "أنت وصال، مساعد أكاديمي متخصص في السوسيولوجيا الرقمية. "
                        "استخدم السياق المعطى فقط للإجابة. إذا لم تكن المعلومات كافية "
                        "قل: لا توجد معلومات كافية في المصادر المتاحة. "
                        "اذكر المصدر (المؤلف، السنة) عند الاستشهاد. "
                        "جاوب بنفس لغة السؤال. كن دقيقاً وأكاديمياً.\n\n"
                        f"السياق:\n{pkg['ctx']}"
                    )
                    
                    resp = client.chat.completions.create(
                        model=MODEL,
                        messages=[
                            {"role": "system", "content": sys_prompt},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.2,
                    )
                    
                    ans = resp.choices[0].message.content
                    ans_lang = detect_lang(ans)
                    cls = "rtl" if ans_lang == "ar" else "ltr"
                    
                    st.markdown(f'<div class="assistant-content {cls}">{ans}</div>', unsafe_allow_html=True)
                    
                    src_list = (
                        pkg["df"][["title", "authors", "publication_year"]]
                        .drop_duplicates()
                        .to_dict("records")
                        if len(pkg["df"]) else []
                    )
                    
                    if src_list:
                        src_cls = "rtl" if ans_lang == "ar" else ""
                        html = f'<div class="sources-panel {src_cls}"><div class="sources-header">📚 المصادر · SOURCES ({len(src_list)})</div>'
                        for i, s in enumerate(src_list, 1):
                            html += f"""
                            <div class="source-row">
                              <div class="source-num">{i}</div>
                              <div class="source-info">
                                <div class="source-title">{s.get('title','')[:90]}</div>
                                <div class="source-meta">{s.get('authors','')[:50]} · {s.get('publication_year','')}</div>
                              </div>
                            </div>"""
                        html += '</div>'
                        st.markdown(html, unsafe_allow_html=True)
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": ans,
                        "sources": src_list
                    })
                    
                except Exception as e:
                    st.error(f"خطأ: {e}")
        
        st.rerun()

# ===== FOOTER =====
st.markdown("""
<div class="footer">
  <div class="footer-text">
    <span class="footer-brand">وصال WESAL</span> — مشروع بحثي أكاديمي<br>
    Powered by Llama 3.3 70B via OpenRouter · RAG over 12 academic sources
  </div>
</div>
""", unsafe_allow_html=True)
