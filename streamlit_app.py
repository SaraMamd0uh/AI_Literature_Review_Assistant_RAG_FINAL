"""
WESAL v4 - LANDING PAGE STYLE
تصميم مستوحى من Book.com & Soundtrack
Light Theme · Warm Colors · Big Hero · Modern
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
    page_title="وصال — أثر الرقمنة على التفاعلات الاجتماعية",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ===== LANDING PAGE CSS =====
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@300;400;500;700;800;900&family=Inter:wght@400;500;600;700;800;900&family=Playfair+Display:wght@700;800;900&display=swap');

:root {
    --bg: #F8F5F0;
    --bg-card: #FFFFFF;
    --bg-soft: #FAF7F2;
    --bg-accent: #F0EBE3;
    --primary: #1A1A2E;
    --primary-soft: #2D2D44;
    --accent: #7BA98F;
    --accent-dark: #5C8A70;
    --accent-warm: #E8C4A0;
    --accent-purple: #A594C7;
    --accent-orange: #F4A574;
    --text: #1A1A2E;
    --text-muted: #6B6B7D;
    --text-light: #9B9BA8;
    --border: #E8E3DB;
    --shadow-sm: 0 2px 8px rgba(26,26,46,0.04);
    --shadow-md: 0 8px 24px rgba(26,26,46,0.08);
    --shadow-lg: 0 20px 60px rgba(26,26,46,0.12);
    --radius: 16px;
    --radius-lg: 28px;
}

/* Reset */
html, body, [class*="css"] {
    font-family: 'Tajawal', 'Inter', sans-serif !important;
    background: var(--bg) !important;
    color: var(--text) !important;
}
.stApp { background: var(--bg) !important; }

#MainMenu, footer, header,
div[data-testid="stToolbar"],
div[data-testid="stDecoration"] {
    visibility: hidden !important;
    display: none !important;
}

.block-container {
    padding-top: 0 !important;
    padding-bottom: 0 !important;
    max-width: 100% !important;
}

/* ========== MAIN CARD (like Book.com white card) ========== */
.main-card {
    max-width: 1240px;
    margin: 24px auto;
    background: var(--bg-card);
    border-radius: var(--radius-lg);
    box-shadow: var(--shadow-lg);
    overflow: hidden;
    position: relative;
}

/* Decorative circles */
.deco-circle-1 {
    position: fixed;
    top: 50px;
    left: -60px;
    width: 200px;
    height: 200px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(232,196,160,0.3), transparent);
    z-index: 0;
    pointer-events: none;
}
.deco-circle-2 {
    position: fixed;
    bottom: 100px;
    right: -80px;
    width: 240px;
    height: 240px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(165,148,199,0.25), transparent);
    z-index: 0;
    pointer-events: none;
}

/* ========== NAVIGATION ========== */
.top-nav {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 24px 48px;
    border-bottom: 1px solid var(--border);
}
.nav-brand {
    display: flex;
    align-items: center;
    gap: 10px;
}
.nav-logo-img {
    width: 42px;
    height: 42px;
    border-radius: 12px;
    object-fit: contain;
}
.nav-logo-fallback {
    width: 42px;
    height: 42px;
    border-radius: 12px;
    background: var(--primary);
    color: white;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 900;
    font-size: 20px;
    font-family: 'Playfair Display', serif;
}
.brand-text {
    font-weight: 900;
    font-size: 1.5rem;
    color: var(--primary);
    letter-spacing: -0.02em;
}
.brand-text .dot { color: var(--accent); }

.nav-links {
    display: flex;
    gap: 40px;
    align-items: center;
}
.nav-link {
    color: var(--text-muted);
    font-size: 0.92rem;
    font-weight: 500;
    cursor: pointer;
    transition: color 0.2s;
    text-decoration: none;
}
.nav-link:hover { color: var(--primary); }
.nav-link.active { color: var(--primary); font-weight: 600; }

.nav-right {
    display: flex;
    gap: 12px;
    align-items: center;
}
.btn-login {
    color: var(--primary);
    font-weight: 600;
    padding: 10px 20px;
    background: transparent;
    border: none;
    cursor: pointer;
    font-size: 0.92rem;
}
.btn-primary {
    background: var(--primary);
    color: white;
    padding: 12px 26px;
    border-radius: 100px;
    font-weight: 600;
    font-size: 0.9rem;
    border: none;
    cursor: pointer;
    transition: all 0.2s;
}
.btn-primary:hover {
    background: var(--accent-dark);
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(26,26,46,0.2);
}

/* ========== HERO SECTION ========== */
.hero {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 40px;
    padding: 60px 48px 40px 48px;
    align-items: center;
    position: relative;
}

.hero-left {
    position: relative;
    z-index: 2;
}
.hero-title {
    font-family: 'Tajawal', sans-serif;
    font-size: 4rem;
    font-weight: 900;
    line-height: 1.05;
    color: var(--primary);
    letter-spacing: -0.03em;
    margin-bottom: 24px;
    direction: rtl;
    text-align: right;
}
.hero-title .highlight {
    color: var(--accent-dark);
    position: relative;
    display: inline-block;
}
.hero-title .highlight::after {
    content: '';
    position: absolute;
    bottom: 4px;
    left: 0;
    right: 0;
    height: 12px;
    background: var(--accent-warm);
    opacity: 0.4;
    z-index: -1;
    border-radius: 4px;
}

.hero-subtitle {
    font-size: 1.05rem;
    color: var(--text-muted);
    line-height: 1.8;
    max-width: 480px;
    margin-bottom: 32px;
    direction: rtl;
    text-align: right;
    font-weight: 400;
}

/* Search Bar (like Book.com) */
.search-bar {
    background: white;
    border-radius: 100px;
    padding: 8px;
    display: flex;
    align-items: center;
    gap: 8px;
    box-shadow: var(--shadow-md);
    max-width: 480px;
    border: 1px solid var(--border);
}
.search-input-wrap {
    flex: 1;
    padding: 0 20px;
}

/* ========== HERO RIGHT (Image + Floating Icons) ========== */
.hero-right {
    position: relative;
    display: flex;
    justify-content: center;
    align-items: center;
    min-height: 480px;
}

.hero-image-wrap {
    position: relative;
    width: 100%;
    height: 480px;
    background: linear-gradient(135deg, #F4E8D8 0%, #E8D4B8 100%);
    border-radius: 24px;
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
}
.hero-image-inner {
    font-size: 12rem;
    filter: drop-shadow(0 20px 40px rgba(0,0,0,0.15));
    animation: float-hero 6s ease-in-out infinite;
}
@keyframes float-hero {
    0%, 100% { transform: translateY(0) rotate(-2deg); }
    50% { transform: translateY(-15px) rotate(2deg); }
}

/* Floating badges */
.float-badge {
    position: absolute;
    background: white;
    padding: 14px;
    border-radius: 16px;
    box-shadow: var(--shadow-md);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.5rem;
    animation: float-badge 4s ease-in-out infinite;
}
.float-badge-1 {
    top: 40px;
    left: 20px;
    background: #E8F4EC;
    animation-delay: 0s;
}
.float-badge-2 {
    top: 100px;
    right: 20px;
    background: #EDE8F5;
    animation-delay: 1s;
}
.float-badge-3 {
    bottom: 80px;
    left: 40px;
    background: #FDF0E4;
    animation-delay: 2s;
}
.float-badge-4 {
    bottom: 40px;
    right: 60px;
    background: #E8F0FB;
    animation-delay: 1.5s;
}
@keyframes float-badge {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-10px); }
}

/* ========== BOTTOM CARDS (like Book.com bottom row) ========== */
.bottom-section {
    padding: 40px 48px 48px 48px;
    border-top: 1px solid var(--border);
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 40px;
    direction: rtl;
}
.bottom-card {
    cursor: pointer;
    transition: transform 0.2s;
}
.bottom-card:hover { transform: translateY(-4px); }

.card-tag {
    font-size: 0.75rem;
    color: var(--accent-dark);
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 8px;
}
.card-tag.muted {
    color: var(--text-light);
}
.card-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: var(--primary);
    line-height: 1.5;
    letter-spacing: -0.01em;
}

/* ========== CHAT INTERFACE (after starting) ========== */
.chat-wrapper {
    max-width: 900px;
    margin: 0 auto;
    padding: 40px 48px;
}
.chat-header {
    display: flex;
    align-items: center;
    gap: 16px;
    margin-bottom: 32px;
    padding-bottom: 24px;
    border-bottom: 1px solid var(--border);
}
.chat-header-icon {
    width: 48px;
    height: 48px;
    border-radius: 14px;
    background: linear-gradient(135deg, var(--accent), var(--accent-dark));
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-size: 22px;
    font-weight: 900;
    font-family: 'Playfair Display', serif;
}
.chat-header-text {
    flex: 1;
}
.chat-header-title {
    font-size: 1.2rem;
    font-weight: 800;
    color: var(--primary);
}
.chat-header-sub {
    font-size: 0.82rem;
    color: var(--text-muted);
    margin-top: 2px;
}

.user-bubble {
    background: var(--primary);
    color: white;
    border-radius: 20px 20px 6px 20px;
    padding: 16px 22px;
    max-width: 75%;
    margin-left: auto;
    line-height: 1.7;
    font-size: 0.96rem;
    box-shadow: var(--shadow-sm);
}
.user-bubble.rtl {
    direction: rtl;
    text-align: right;
    border-radius: 20px 20px 20px 6px;
    margin-left: 0;
    margin-right: auto;
}

.assistant-wrap {
    display: flex;
    gap: 14px;
    max-width: 90%;
    margin-bottom: 8px;
}
.assistant-wrap.rtl {
    direction: rtl;
}
.assistant-avatar {
    width: 40px;
    height: 40px;
    border-radius: 12px;
    background: linear-gradient(135deg, var(--accent), var(--accent-dark));
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-size: 16px;
    font-weight: 800;
    font-family: 'Playfair Display', serif;
    flex-shrink: 0;
}
.assistant-body {
    flex: 1;
    background: var(--bg-soft);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 18px 22px;
    line-height: 1.9;
    font-size: 0.98rem;
    color: var(--text);
}
.assistant-body.rtl {
    direction: rtl;
    text-align: right;
}

/* Sources */
.sources-block {
    margin-top: 14px;
    padding: 16px;
    background: white;
    border: 1px solid var(--border);
    border-radius: 14px;
}
.sources-title {
    font-size: 0.72rem;
    color: var(--text-light);
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 12px;
}
.source-line {
    padding: 10px 0;
    border-top: 1px solid var(--border);
    font-size: 0.86rem;
    display: flex;
    gap: 10px;
    align-items: flex-start;
}
.source-line:first-of-type { border-top: none; padding-top: 0; }
.source-num-badge {
    width: 22px;
    height: 22px;
    background: var(--accent);
    color: white;
    border-radius: 6px;
    font-size: 0.72rem;
    font-weight: 700;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    font-family: 'Inter', sans-serif;
}
.source-content {
    flex: 1;
}
.source-name {
    color: var(--primary);
    font-weight: 600;
    line-height: 1.5;
}
.source-authors {
    color: var(--text-muted);
    font-size: 0.78rem;
    margin-top: 2px;
}

/* ========== STREAMLIT OVERRIDES ========== */

/* Chat input */
.stChatInput {
    max-width: 900px;
    margin: 0 auto;
    padding: 0 48px;
}
.stChatInput > div {
    border-radius: 100px !important;
    border: 1px solid var(--border) !important;
    background: white !important;
    box-shadow: var(--shadow-md) !important;
    padding: 6px !important;
}
.stChatInput > div:focus-within {
    border-color: var(--accent) !important;
    box-shadow: 0 8px 24px rgba(123,169,143,0.2) !important;
}
.stChatInput textarea {
    color: var(--text) !important;
    font-family: 'Tajawal', sans-serif !important;
    font-size: 0.96rem !important;
    background: transparent !important;
}
.stChatInput textarea::placeholder {
    color: var(--text-light) !important;
}

/* Buttons */
.stButton > button {
    background: white !important;
    border: 1px solid var(--border) !important;
    color: var(--primary) !important;
    border-radius: 14px !important;
    padding: 16px 22px !important;
    text-align: right !important;
    direction: rtl !important;
    font-family: 'Tajawal', sans-serif !important;
    font-size: 0.92rem !important;
    line-height: 1.7 !important;
    font-weight: 500 !important;
    transition: all 0.2s !important;
    width: 100% !important;
    box-shadow: var(--shadow-sm) !important;
}
.stButton > button:hover {
    background: var(--bg-soft) !important;
    border-color: var(--accent) !important;
    color: var(--primary) !important;
    transform: translateY(-2px) !important;
    box-shadow: var(--shadow-md) !important;
}

.stChatMessage {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 8px 0 !important;
}

/* Hide chat message avatars (we make custom) */
.stChatMessage [data-testid="chatAvatarIcon-assistant"],
.stChatMessage [data-testid="chatAvatarIcon-user"] {
    display: none !important;
}

/* Responsive */
@media (max-width: 900px) {
    .hero { grid-template-columns: 1fr; padding: 40px 24px; }
    .hero-title { font-size: 2.5rem; }
    .bottom-section { grid-template-columns: 1fr; padding: 32px 24px; gap: 20px; }
    .top-nav { padding: 20px 24px; flex-wrap: wrap; gap: 12px; }
    .nav-links { display: none; }
    .hero-image-wrap { height: 320px; }
    .hero-image-inner { font-size: 8rem; }
    .chat-wrapper { padding: 24px; }
    .stChatInput { padding: 0 24px; }
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
    cands["score"] = sc[rank]
    top = cands["score"].max() if len(cands) else 0
    cands = cands[cands["score"] >= top*0.25].head(4)
    lines, rows, used = [], [], 0
    for _, r in cands.iterrows():
        if used + len(r["chunk_text"].split()) > 620: continue
        lines.append(f"[{r['title']} — {r['authors']} ({r['publication_year']})]\n{r['chunk_text']}")
        rows.append(r); used += len(r["chunk_text"].split())
    return {"ctx": "\n\n".join(lines), "df": pd.DataFrame(rows)}

def get_key():
    if "OPENROUTER_API_KEY" in st.secrets: return st.secrets["OPENROUTER_API_KEY"]
    if "API" in st.secrets: return st.secrets["API"]
    return os.environ.get("OPENROUTER_API_KEY")

# Decorative circles
st.markdown('<div class="deco-circle-1"></div><div class="deco-circle-2"></div>', unsafe_allow_html=True)

# ===== MAIN CARD START =====
logo_url = "https://raw.githubusercontent.com/SaraMamd0uh/AI_Literature_Review_Assistant_RAG_FINAL/main/logo.png"

st.markdown(f"""
<div class="main-card">
  <div class="top-nav">
    <div class="nav-brand">
      <img src="{logo_url}" class="nav-logo-img" onerror="this.outerHTML='<div class=\\'nav-logo-fallback\\'>و</div>';">
      <div class="brand-text">وصال<span class="dot">.</span></div>
    </div>
    <div class="nav-links">
      <a class="nav-link active">الرئيسية</a>
      <a class="nav-link">المصادر</a>
      <a class="nav-link">حول المشروع</a>
      <a class="nav-link">تواصل</a>
    </div>
    <div class="nav-right">
      <button class="btn-login">تسجيل الدخول</button>
      <button class="btn-primary">ابدأ الآن</button>
    </div>
  </div>
""", unsafe_allow_html=True)

# ===== LOAD DATA =====
(index_data, err) = get_index()
if err:
    st.error("⚠️ data/chunks.parquet not found"); st.stop()
df, bm25 = index_data

if "messages" not in st.session_state:
    st.session_state.messages = []
if "started" not in st.session_state:
    st.session_state.started = False

# ===== HERO SECTION (Landing) =====
if not st.session_state.started and len(st.session_state.messages) == 0:
    st.markdown("""
    <div class="hero">
      <div class="hero-left">
        <div class="hero-title">
          اكتشف أثر <span class="highlight">الرقمنة</span><br>
          على تفاعلاتنا<br>
          الاجتماعية
        </div>
        <div class="hero-subtitle">
          مساعد بحثي أكاديمي ذكي يجيبك من 12 مصدر علمي محكّم
          عن التحولات الاجتماعية في العصر الرقمي
        </div>
      </div>
      
      <div class="hero-right">
        <div class="hero-image-wrap">
          <div class="float-badge float-badge-1">💡</div>
          <div class="float-badge float-badge-2">🏆</div>
          <div class="float-badge float-badge-3">🎯</div>
          <div class="float-badge float-badge-4">📖</div>
          <div class="hero-image-inner">📚</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Search bar as start button
    st.markdown('<div style="max-width:900px; margin:0 auto; padding:0 48px 40px 48px;">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([2, 3, 2])
    with col2:
        if st.button("🔍  ابدأ البحث الآن — Start Researching", key="start_btn", use_container_width=True):
            st.session_state.started = True
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Bottom cards (like Book.com)
    st.markdown("""
    <div class="bottom-section">
      <div class="bottom-card">
        <div class="card-tag">جديد · 2024</div>
        <div class="card-title">هل قرأت آخر الأبحاث عن الرقمنة الاجتماعية؟</div>
      </div>
      <div class="bottom-card">
        <div class="card-tag muted">Study · #05</div>
        <div class="card-title">كبار السن والتكنولوجيا: كيف تغيرت علاقاتهم؟</div>
      </div>
      <div class="bottom-card">
        <div class="card-tag muted">Study · #08</div>
        <div class="card-title">مقاهي القاهرة: التفاعل وجهاً لوجه في عصر الشاشات</div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)  # close main-card

else:
    # ===== CHAT INTERFACE =====
    st.markdown(f"""
    <div class="chat-wrapper">
      <div class="chat-header">
        <div class="chat-header-icon">و</div>
        <div class="chat-header-text">
          <div class="chat-header-title">محادثة مع وصال</div>
          <div class="chat-header-sub">مدعوم بـ Llama 3.3 · RAG على 12 مصدر أكاديمي</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Example prompts (if no messages yet)
    if len(st.session_state.messages) == 0:
        st.markdown('<div style="max-width:900px; margin:0 auto; padding:0 48px 24px 48px;">', unsafe_allow_html=True)
        st.markdown('<div style="color:var(--text-muted); font-size:0.9rem; margin-bottom:16px; text-align:right; direction:rtl;">✨ جرّب أحد هذه الأسئلة:</div>', unsafe_allow_html=True)
        
        examples = [
            ("🏠", "كيف تؤثر الرقمنة على كبار السن وعلاقاتهم الاجتماعية؟"),
            ("☕", "What is the impact of digital devices on face-to-face interaction in Cairo coffeehouses?"),
            ("🌐", "ما علاقة الرقمنة بالهوية الاجتماعية وتكوين المجموعات على الإنترنت؟"),
        ]
        for icon, ex in examples:
            if st.button(f"{icon}   {ex}", key=f"ex_{ex[:15]}", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": ex})
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Render chat
    st.markdown('<div style="max-width:900px; margin:0 auto; padding:0 48px;">', unsafe_allow_html=True)
    for m in st.session_state.messages:
        lang = detect_lang(m["content"])
        is_rtl = lang == "ar"
        cls = "rtl" if is_rtl else ""
        
        if m["role"] == "user":
            st.markdown(f"""
            <div style="display:flex; margin-bottom:16px;">
              <div class="user-bubble {cls}">{m["content"]}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="assistant-wrap {cls}" style="margin-bottom:20px;">
              <div class="assistant-avatar">و</div>
              <div class="assistant-body {cls}">{m["content"]}</div>
            </div>
            """, unsafe_allow_html=True)
            
            if "sources" in m and m["sources"]:
                srcs_html = f'<div style="max-width:90%; margin:{"0 0 20px 54px" if not is_rtl else "0 54px 20px 0"};"><div class="sources-block"><div class="sources-title">📚 المصادر · SOURCES ({len(m["sources"])})</div>'
                for i, s in enumerate(m["sources"], 1):
                    srcs_html += f"""
                    <div class="source-line">
                      <div class="source-num-badge">{i}</div>
                      <div class="source-content">
                        <div class="source-name">{s.get('title','')[:90]}</div>
                        <div class="source-authors">{s.get('authors','')[:50]} · {s.get('publication_year','')}</div>
                      </div>
                    </div>"""
                srcs_html += '</div></div>'
                st.markdown(srcs_html, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)  # close main-card

# ===== INPUT =====
if st.session_state.started or len(st.session_state.messages) > 0:
    if prompt := st.chat_input("اسأل عن أثر الرقمنة... / Ask about digitalization..."):
        st.session_state.started = True
        lang = detect_lang(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        key = get_key()
        if not key:
            st.error("⚠️ Add OPENROUTER_API_KEY to Secrets"); st.stop()
        
        pkg = build_pkg(prompt, df, bm25)
        
        if not pkg["ctx"]:
            ans = "لم أجد مصادر كافية في قاعدة البيانات. جرب صياغة مختلفة للسؤال."
            st.session_state.messages.append({"role": "assistant", "content": ans, "sources": []})
        else:
            try:
                client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)
                sys_prompt = (
                    "أنت وصال، مساعد أكاديمي متخصص في السوسيولوجيا الرقمية. "
                    "استخدم السياق فقط. إذا غير كاف قل: لا توجد معلومات كافية. "
                    "اذكر المصدر (المؤلف، السنة). جاوب بنفس لغة السؤال.\n\n"
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
                src_list = (
                    pkg["df"][["title", "authors", "publication_year"]]
                    .drop_duplicates()
                    .to_dict("records")
                    if len(pkg["df"]) else []
                )
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": ans,
                    "sources": src_list
                })
            except Exception as e:
                st.error(f"خطأ: {e}")
        
        st.rerun()
