"""
WESAL - Full Landing Page PRO
مستوحى 100% من تصميم الصورة اللي بعتيها - بتصميم عالمي
Hero + Features + Chat + Sources + Steps
Llama فقط - يحافظ على logo.png الأصلي
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
    page_title="وصال - مساعد الخدمة الاجتماعية الذكي",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700;800&family=Inter:wght@400;500&display=swap');
* {font-family:'Tajawal',sans-serif;}
#MainMenu, footer, header, [data-testid="stToolbar"] {display:none !important;}
.stApp {background:#FDFCFB;}

.topbar {
  position:sticky; top:0; z-index:100;
  background:rgba(253,252,251,0.9); backdrop-filter:blur(12px);
  border-bottom:1px solid #EDE9E3;
  height:60px; display:flex; align-items:center;
  margin:-1rem -3rem 0 -3rem; padding:0 24px;
}
.topbar-inner{max-width:1150px; width:100%; margin:0 auto; display:flex; justify-content:space-between; align-items:center;}
.brand{display:flex; align-items:center; gap:10px; font-weight:700;}
.nav-links{display:flex; gap:24px; font-size:13px; color:#78716C;}
.nav-cta{background:#0F3D3E; color:white; padding:8px 18px; border-radius:100px; font-size:13px; font-weight:600;}

/* HERO - like image 2 */
.hero-sec{max-width:1150px; margin:0 auto; padding:48px 24px 40px 24px;}
.hero-grid{display:grid; grid-template-columns:0.9fr 1.1fr; gap:48px; align-items:center;}
@media(max-width:900px){.hero-grid{grid-template-columns:1fr;}}
.hero-img{position:relative; background:white; border-radius:24px; padding:16px; border:1px solid #EDE9E3;}
.hero-img img{width:100%; height:auto; border-radius:16px; object-fit:contain;}
.float-chip{position:absolute; background:white; border:1px solid #E7E5E4; border-radius:14px; width:48px; height:48px; display:grid; place-items:center; box-shadow:0 8px 20px rgba(0,0,0,0.07); font-size:20px;}
.fc1{top:16px; right:16px; animation:float 3s infinite;}
.fc2{top:90px; left:12px; animation:float 3s infinite 0.5s;}
.fc3{bottom:60px; left:32px; animation:float 3s infinite 1s;}
@keyframes float{0%,100%{transform:translateY(0)}50%{transform:translateY(-6px)}}

.hero-title{font-size:38px; font-weight:800; line-height:1.25; color:#1C1917; direction:rtl; text-align:right; margin:0;}
.hero-title .accent{color:#0F3D3E;}
.hero-desc{margin-top:16px; font-size:15px; color:#78716C; line-height:1.8; direction:rtl; text-align:right; max-width:480px;}
.hero-search{margin-top:28px; display:flex; gap:8px; background:white; border:1px solid #E7E5E4; border-radius:100px; padding:6px 6px 6px 16px; box-shadow:0 4px 20px rgba(0,0,0,0.05); max-width:440px; align-items:center;}
.hero-search input{flex:1; border:none; outline:none; font-size:14px; direction:rtl; text-align:right; background:transparent;}
.hero-search button{background:#14B8A6; color:white; border:none; padding:10px 22px; border-radius:100px; font-weight:700; font-size:13px; cursor:pointer;}
.hero-meta{margin-top:20px; display:flex; gap:16px; font-size:12px; color:#A8A29E; direction:rtl;}

.section{max-width:1150px; margin:0 auto; padding:56px 24px;}
.section-title{font-size:22px; font-weight:700; text-align:center; margin-bottom:8px; color:#1C1917; direction:rtl;}
.section-sub{font-size:14px; color:#78716C; text-align:center; margin-bottom:32px; direction:rtl;}
.features{display:grid; grid-template-columns:repeat(3,1fr); gap:16px;}
@media(max-width:800px){.features{grid-template-columns:1fr 1fr;}}
@media(max-width:500px){.features{grid-template-columns:1fr;}}
.feat-card{background:white; border:1px solid #EDE9E3; border-radius:16px; padding:20px; direction:rtl; text-align:right; transition:all 0.2s;}
.feat-card:hover{border-color:#D6D3D1; box-shadow:0 4px 12px rgba(0,0,0,0.04);}
.feat-icon{width:36px; height:36px; background:#F5F5F4; border-radius:10px; display:grid; place-items:center; margin-bottom:12px; font-size:18px;}
.feat-title{font-size:14px; font-weight:700; margin-bottom:6px;}
.feat-desc{font-size:12.5px; color:#78716C; line-height:1.6;}

.chat-demo{background:white; border:1px solid #EDE9E3; border-radius:20px; overflow:hidden; display:grid; grid-template-columns:1fr 1.2fr; box-shadow:0 8px 30px rgba(0,0,0,0.04);}
@media(max-width:900px){.chat-demo{grid-template-columns:1fr;}}
.chat-left{background:#0F3D3E; color:white; padding:24px;}
.chat-right{padding:24px; background:#FAFAF9;}

.step-grid{display:grid; grid-template-columns:repeat(3,1fr); gap:16px;}
@media(max-width:700px){.step-grid{grid-template-columns:1fr;}}
.step-card{background:white; border:1px solid #EDE9E3; border-radius:16px; padding:20px; text-align:center; direction:rtl;}
.step-num{width:32px; height:32px; background:#0F3D3E; color:white; border-radius:50%; display:grid; place-items:center; margin:0 auto 12px auto; font-weight:700; font-size:14px;}

/* Chat inside */
.chat-wrap{max-width:720px; margin:0 auto;}
.msg-user{background:#F5F5F4; border-radius:16px; padding:12px 16px; margin:12px 0 12px auto; max-width:85%; direction:rtl; text-align:right; font-size:14px; line-height:1.7;}
.msg-assist{padding:16px 0; line-height:1.9; direction:rtl; text-align:right; border-bottom:1px solid #F5F5F4; font-size:15px;}
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

# TOPBAR
st.markdown("""
<div class="topbar"><div class="topbar-inner">
  <div class="brand">📚 وصال <span style="font-weight:400; color:#78716C; margin-right:6px; font-size:13px;">أثر الرقمنة</span></div>
  <div style="display:flex; gap:16px; align-items:center;">
    <div class="nav-links"><span>المكتبة</span><span>كيف يعمل</span><span>المصادر</span></div>
    <div class="nav-cta">جرّب المساعد</div>
  </div>
</div></div>
""", unsafe_allow_html=True)

# HERO - exactly like second image
st.markdown('<div class="hero-sec"><div class="hero-grid">', unsafe_allow_html=True)

# Left image
st.markdown('<div class="hero-img">', unsafe_allow_html=True)
if Path("hero_final_books.jpg").exists():
    st.image("hero_final_books.jpg", use_column_width=True)
elif Path("hero_books.png").exists():
    st.image("hero_books.png", use_column_width=True)
elif Path("logo.png").exists():
    st.image("logo.png", use_column_width=True)
st.markdown("""
  <div class="float-chip fc1">💡</div>
  <div class="float-chip fc2">🏆</div>
  <div class="float-chip fc3">🎯</div>
</div>
""", unsafe_allow_html=True)

# Right text
st.markdown("""
<div>
  <h1 class="hero-title">
    اعثر على الإجابة<br>التي تبحث عنها في<br><span class="accent">الخدمة الاجتماعية.</span>
  </h1>
  <p class="hero-desc">
    مساعد ذكي يجيبك من كتب ومراجع الخدمة الاجتماعية وعلم الاجتماع، مع ذكر المصدر ورقم الصفحة. مبني على 11 ورقة بحثية وكتاب عن أثر الرقمنة.
  </p>
  <div class="hero-search">
    <span style="color:#A8A29E;">🔍</span>
    <input placeholder="اكتب سؤالك المهني هنا..." disabled>
    <button>ابحث</button>
  </div>
  <div class="hero-meta">
    <span>+400 مرجع أكاديمي</span>
    <span>• إجابات موثقة بالمصدر</span>
    <span>• عربي وإنجليزي</span>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown('</div></div>', unsafe_allow_html=True)

# FEATURES - 6 cards like image 2
st.markdown("""
<div class="section">
  <div class="section-title">معرفة المجال، منظمة ومتاحة في ثوان</div>
  <p class="section-sub">كل ما تحتاجه من مفاهيم، نظريات، وأدوات التدخل المهني في مكان واحد</p>
  <div class="features">
    <div class="feat-card"><div class="feat-icon">📖</div><div class="feat-title">بحث دلالي + لفظي</div><div class="feat-desc">يجمع بين فهم المعنى والتطابق الحرفي للوصول لأدق النتائج</div></div>
    <div class="feat-card"><div class="feat-icon">🔍</div><div class="feat-title">إجابات موثقة</div><div class="feat-desc">كل إجابة مع مصدرها، المؤلف، سنة النشر ورقم الصفحة</div></div>
    <div class="feat-card"><div class="feat-icon">🌐</div><div class="feat-title">ثنائي اللغة</div><div class="feat-desc">اسأل بالعربي أو الإنجليزي واحصل على إجابة بنفس اللغة</div></div>
    <div class="feat-card"><div class="feat-icon">⚡</div><div class="feat-title">سريع جداً</div><div class="feat-desc">يبحث في 12 مصدر و60 مقطع في أقل من ثانية</div></div>
    <div class="feat-card"><div class="feat-icon">🎯</div><div class="feat-title">متخصص</div><div class="feat-desc">متخصص في أثر الرقمنة على التفاعلات الاجتماعية والخدمة</div></div>
    <div class="feat-card"><div class="feat-icon">🤖</div><div class="feat-title">Llama 3.3 70B</div><div class="feat-desc">مدعوم بأحدث نماذج اللغة المجانية عبر OpenRouter</div></div>
  </div>
</div>
""", unsafe_allow_html=True)

# CHAT DEMO SECTION
st.markdown("""
<div class="section" style="background:white; border-top:1px solid #EDE9E3; border-bottom:1px solid #EDE9E3;">
  <div style="max-width:1150px; margin:0 auto;">
    <div style="display:grid; grid-template-columns:1fr 1.2fr; gap:32px; align-items:center;">
      <div style="direction:rtl; text-align:right;">
        <h3 style="font-size:20px; font-weight:700; margin-bottom:12px;">اسأل كما تسأل زميلاً خبيراً</h3>
        <p style="color:#78716C; font-size:14px; line-height:1.7;">اكتب سؤالك المهني بلغة طبيعية، وسيجيبك وصال من المكتبة الأكاديمية مع ذكر المصدر.</p>
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# REAL CHAT - Llama only
st.markdown('<div style="max-width:760px; margin:0 auto; padding:24px;">', unsafe_allow_html=True)

# Show past messages
for m in st.session_state.msgs:
    if m["role"]=="user":
        st.markdown(f'<div class="msg-user">{m["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="msg-assist">{m["content"]}</div>', unsafe_allow_html=True)
        if "sources" in m and m["sources"]:
            with st.expander(f"المصادر • {len(m['sources'])}"):
                for s in m["sources"]:
                    st.caption(f"{s['title']} — {s['authors']} ({s['publication_year']})")

# Example prompts
if len(st.session_state.msgs)==0:
    c1,c2,c3=st.columns(3)
    if c1.button("كيف تؤثر الرقمنة على كبار السن؟", use_container_width=True):
        st.session_state.msgs.append({"role":"user","content":"كيف تؤثر الرقمنة على كبار السن وعلاقاتهم الاجتماعية؟"}); st.rerun()
    if c2.button("What is impact in Cairo coffeehouses?", use_container_width=True):
        st.session_state.msgs.append({"role":"user","content":"What is the impact of digital devices on face-to-face interaction in Cairo coffeehouses?"}); st.rerun()
    if c3.button("علاقة الرقمنة بالهوية؟", use_container_width=True):
        st.session_state.msgs.append({"role":"user","content":"ما علاقة الرقمنة بالهوية الاجتماعية؟"}); st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

if prompt := st.chat_input("اكتب سؤالك المهني هنا..."):
    st.session_state.msgs.append({"role":"user","content":prompt})
    key=get_key()
    if not key:
        st.session_state.msgs.append({"role":"assistant","content":"أضيفي OPENROUTER_API_KEY في Secrets بهذا الشكل:\n\nOPENROUTER_API_KEY = \"sk-or-v1-...\""})
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
                        {"role":"system","content":f"أنت وصال، مساعد متخصص في الخدمة الاجتماعية وعلم الاجتماع. استخدم السياق فقط. اذكر المصدر. جاوب بنفس لغة السؤال.\n\nالسياق:\n{pkg['ctx']}"},
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

# Footer
st.markdown("""
<div style="max-width:1150px; margin:40px auto 0 auto; padding:24px; border-top:1px solid #EDE9E3; text-align:center; color:#A8A29E; font-size:12px;">
  © 2025 وصال • مبني بـ Llama 3.3 70B • RAG على 11 ورقة وكتاب • Section 16
</div>
""", unsafe_allow_html=True)
