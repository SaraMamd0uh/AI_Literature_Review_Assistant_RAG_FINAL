"""
WESAL - React Hero Inspired Version
نفس تصميم React اللي بعتيه لكن بـ Streamlit Python
يحافظ على نفس اللوجو الأصلي و Llama فقط
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

st.set_page_config(page_title="وصال - مساعد الأدبيات", page_icon="📚", layout="wide")

# ===== CSS - Converted from React Tailwind to Streamlit =====
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700;800&display=swap');
* {font-family: 'Tajawal', sans-serif;}
#MainMenu, footer, header {visibility: hidden;}

.hero-section {
  max-width: 1150px;
  margin: 0 auto;
  padding: 48px 20px 32px 20px;
}
.hero-grid {
  display: grid;
  grid-template-columns: 1.1fr 0.9fr;
  gap: 48px;
  align-items: center;
}
@media (max-width: 900px) {
  .hero-grid {grid-template-columns: 1fr; gap: 32px;}
}

.hero-title {
  font-size: 3.2rem;
  font-weight: 800;
  line-height: 1.15;
  color: #111827;
  margin: 0 0 16px 0;
  direction: rtl;
  text-align: right;
}
.hero-title span {display: block;}
.hero-title .primary {color: #0F2167;}

.hero-desc {
  margin-top: 20px;
  font-size: 16px;
  color: #6B7280;
  line-height: 1.8;
  max-width: 440px;
  direction: rtl;
  text-align: right;
}

.search-box {
  margin-top: 32px;
  display: flex;
  align-items: center;
  gap: 8px;
  background: white;
  border-radius: 999px;
  padding: 8px 8px 8px 20px;
  border: 1px solid #E5E7EB;
  box-shadow: 0 4px 20px rgba(0,0,0,0.06);
  max-width: 440px;
}
.search-box input {
  flex: 1;
  border: none;
  outline: none;
  font-size: 14px;
  background: transparent;
  text-align: right;
  direction: rtl;
}
.search-btn {
  background: #14B8A6;
  color: white;
  border: none;
  padding: 10px 24px;
  border-radius: 999px;
  font-weight: 700;
  font-size: 14px;
  cursor: pointer;
}

.meta-row {
  margin-top: 28px;
  display: flex;
  flex-wrap: wrap;
  gap: 20px;
  font-size: 12px;
  color: #6B7280;
  direction: rtl;
}

.hero-image-wrap {
  position: relative;
  width: 100%;
  max-width: 420px;
  margin: 0 auto;
}
.hero-image-wrap img {
  width: 100%;
  height: auto;
  border-radius: 24px;
}

.float-chip {
  position: absolute;
  background: white;
  border: 1px solid #E5E7EB;
  border-radius: 16px;
  width: 52px;
  height: 52px;
  display: grid;
  place-items: center;
  box-shadow: 0 8px 20px rgba(0,0,0,0.08);
  animation: float 3s ease-in-out infinite;
}
.float-chip:nth-child(2) {top: 24px; right: 12px; animation-delay: 0s; background: #F3F4F6;}
.float-chip:nth-child(3) {top: 80px; left: 12px; animation-delay: 0.5s;}
.float-chip:nth-child(4) {bottom: 64px; left: 24px; animation-delay: 1s; background: #F3F4F6;}

@keyframes float {
  0%, 100% {transform: translateY(0px);}
  50% {transform: translateY(-8px);}
}

/* Chat */
.chat-wrap {max-width: 760px; margin: 40px auto 0 auto; padding: 0 20px 100px 20px;}
.msg-user {background:#F9FAFB; border:1px solid #E5E7EB; border-radius:16px; padding:14px 18px; margin:16px 0 16px auto; max-width:88%; direction:rtl; text-align:right; line-height:1.8;}
.msg-assist {padding:18px 0; line-height:1.9; border-bottom:1px solid #F3F4F6; direction:rtl; text-align:right;}

.stChatInput {max-width: 760px; margin: 0 auto;}
.stChatInput > div {border-radius: 999px !important; border:1px solid #D1D5DB !important; box-shadow: 0 4px 20px rgba(0,0,0,0.06) !important;}
</style>
""", unsafe_allow_html=True)

# Helpers
def norm_ws(t): return re.sub(r"\s+"," ",t).strip()
AR_D = re.compile(r"[\u064B-\u0652\u0670\u0640]")
def norm_ar(t):
    t=AR_D.sub("",t); t=re.sub(r"[إأآ]","ا",t); t=re.sub(r"ى","ي",t); t=re.sub(r"[^\u0600-\u06FF\s0-9]"," ",t); return norm_ws(t)
def norm_en(t): return norm_ws(re.sub(r"[^\w\s]"," ",t.lower()))
def detect(t):
    ar=re.compile(r"[\u0600-\u06FF]"); lets=re.findall(r"[^\W\d_]",t,re.UNICODE)
    if not lets: return "en"
    return "ar" if len(ar.findall(t))/len(lets)>=0.15 else "en"
def lex(t): return norm_ar(t) if detect(t)=="ar" else norm_en(t)
def tok(t): return re.findall(r"[a-z0-9]+|[\u0600-\u06FF]+", lex(t).lower())

@st.cache_resource(show_spinner=False)
def get_index():
    if not CHUNKS_PATH.exists(): return None, "no_data"
    df=pd.read_parquet(CHUNKS_PATH)
    bm=BM25Okapi([tok(x) for x in df["search_text"]])
    return (df,bm), None

def build_pkg(q,df,bm):
    sc=bm.get_scores(tok(q)); rk=np.argsort(sc)[::-1][:10]
    cd=df.iloc[rk].copy(); cd["score"]=sc[rk]
    top=cd["score"].max() if len(cd) else 0
    cd=cd[cd["score"]>=top*0.25].head(4)
    lines=[]; rows=[]; used=0
    for _,r in cd.iterrows():
        if used+len(r["chunk_text"].split())>600: continue
        lines.append(f"[{r['title']} — {r['authors']} ({r['publication_year']})]\n{r['chunk_text']}")
        rows.append(r); used+=len(r["chunk_text"].split())
    return {"ctx":"\n\n".join(lines), "df":pd.DataFrame(rows)}

def get_key():
    if "OPENROUTER_API_KEY" in st.secrets: return st.secrets["OPENROUTER_API_KEY"]
    if "API" in st.secrets: return st.secrets["API"]
    return os.environ.get("OPENROUTER_API_KEY")

(df_bm, err) = get_index()
if err: st.stop()
df,bm=df_bm

if "msgs" not in st.session_state: st.session_state.msgs=[]

# ===== HERO SECTION - React Inspired =====
if len(st.session_state.msgs)==0:
    st.markdown('<div class="hero-section"><div class="hero-grid">', unsafe_allow_html=True)
    
    # Left - Text
    st.markdown("""
    <div>
      <h1 class="hero-title">
        اعثر على الإجابة
        <span>التي تبحث عنها في</span>
        <span class="primary">الخدمة الاجتماعية.</span>
      </h1>
      <p class="hero-desc">
        مساعد ذكي يجيبك من كتب ومراجع الخدمة الاجتماعية وعلم الاجتماع، مع ذكر المصدر ورقم الصفحة.
        <br>مبني على 11 ورقة بحثية وكتاب متخصص عن أثر الرقمنة.
      </p>
      <div class="meta-row">
        <span>📚 +400 مرجع أكاديمي</span>
        <span>✅ إجابات موثقة بالمصدر</span>
        <span>🌐 عربي وإنجليزي</span>
      </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Right - Image with floating chips
    st.markdown('<div class="hero-image-wrap">', unsafe_allow_html=True)
    if Path("hero_books.png").exists():
        st.image("hero_books.png", use_column_width=True)
    elif Path("logo.png").exists():
        st.image("logo.png", use_column_width=True)
    else:
        st.markdown('<div style="width:100%; height:320px; background:#F9FAFB; border-radius:24px; display:grid; place-items:center;">📚</div>', unsafe_allow_html=True)
    
    st.markdown("""
      <div class="float-chip">💡</div>
      <div class="float-chip">🏆</div>
      <div class="float-chip">🎯</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div></div>', unsafe_allow_html=True)

    # Example questions below hero
    st.markdown('<div style="max-width:760px; margin:0 auto; padding:0 20px;">', unsafe_allow_html=True)
    cols=st.columns(3)
    examples=["كيف تؤثر الرقمنة على كبار السن؟","What is impact of digital devices on Cairo coffeehouses?","علاقة الرقمنة بالهوية الاجتماعية؟"]
    for i,ex in enumerate(examples):
        with cols[i%3]:
            if st.button(ex, key=f"ex{i}", use_container_width=True):
                st.session_state.msgs.append({"role":"user","content":ex})
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

else:
    st.markdown('<div class="chat-wrap">', unsafe_allow_html=True)
    for m in st.session_state.msgs:
        is_ar = detect(m["content"])=="ar"
        if m["role"]=="user":
            st.markdown(f'<div class="msg-user">{"{0}".format(m["content"])}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="msg-assist">{m["content"]}</div>', unsafe_allow_html=True)
            if "sources" in m and m["sources"]:
                with st.expander(f"المصادر · {len(m['sources'])}"):
                    for s in m["sources"]:
                        st.caption(f"{s['title']} — {s['authors']} ({s['publication_year']})")
    st.markdown('</div>', unsafe_allow_html=True)

# Chat Input - always at bottom
if prompt := st.chat_input("اكتب سؤالك المهني هنا..."):
    st.session_state.msgs.append({"role":"user","content":prompt})
    key=get_key()
    if not key:
        st.session_state.msgs.append({"role":"assistant","content":"أضيفي OPENROUTER_API_KEY في Secrets"})
    else:
        pkg=build_pkg(prompt,df,bm)
        if not pkg["ctx"]:
            st.session_state.msgs.append({"role":"assistant","content":"لم أجد مصادر كافية، جرب صياغة أخرى."})
        else:
            try:
                client=OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)
                resp=client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role":"system","content":f"أنت مساعد أكاديمي متخصص في الخدمة الاجتماعية. استخدم السياق فقط. اذكر المصدر (المؤلف، السنة). جاوب بنفس لغة السؤال.\n\nالسياق:\n{pkg['ctx']}"},
                        {"role":"user","content":prompt}
                    ],
                    temperature=0.2
                )
                ans=resp.choices[0].message.content
                srcs=pkg["df"][["title","authors","publication_year"]].drop_duplicates().to_dict("records") if len(pkg["df"]) else []
                st.session_state.msgs.append({"role":"assistant","content":ans,"sources":srcs})
            except Exception as e:
                st.session_state.msgs.append({"role":"assistant","content":f"خطأ: {e}"})
    st.rerun()
