# AI Literature Review Assistant — RAG (هيكل الدكتور)

نظام RAG ثنائي اللغة (عربي/إنجليزي) عن أثر الرقمنة على التفاعلات الاجتماعية، مبني على 11 ورقة بحثية وكتاب واحد،
مقسّم إلى المراحل المطلوبة بالظبط + تطبيق Streamlit.

## هيكل المشروع

```
rag_pipeline_app/
├── 01_documents.py              # قراءة PDFs + استخراج نص + تصحيح عربي + ميتاداتا
├── 02_preprocessing.py          # تنظيف النص (عربي/إنجليزي)
├── 03_chunking.py                # تقسيم النص لقطع (chunks)
├── 04_vector_representation.py  # حساب embeddings متعددة اللغات
├── 05_create_chroma_store.py    # بناء قاعدة بيانات Chroma الدائمة
├── 06_retrieve_context.py       # الاسترجاع الهجين (Chroma + BM25) + بناء الـ context
├── 07_prompting.py               # قوالب الـ prompt + استدعاء Gemini
├── streamlit_app.py              # التطبيق النهائي
├── requirements.txt
├── .gitignore
├── .streamlit/secrets.toml.example
└── data/
    ├── pdfs/papers/    ← ضعي هنا الـ 11 ورقة (لن يُرفع على GitHub)
    └── pdfs/book/      ← ضعي هنا الكتاب (لن يُرفع على GitHub)
```

## الفكرة العامة

المراحل 01 → 05 هي "خط أنابيب تحضير البيانات" (Data Pipeline): تتشغّل **مرة واحدة بس** (على جهازك أو Colab)
عشان تحوّل الـ PDFs إلى قاعدة بيانات Chroma جاهزة. بعد كده، بترفعي نتيجة المراحل دي (`data/*.parquet`, `data/*.npy`, `chroma_db/`)
على GitHub، والتطبيق (`streamlit_app.py`) بيستخدم المراحل 06 و 07 فقط وقت التشغيل — من غير ما يحتاج PDFs أو Drive تاني.

---

## الخطوة 1 — تجهيز الداتا محليًا

1. حمّلي الـ 11 ورقة PDF من Google Drive وضعيهم في `data/pdfs/papers/`
2. حمّلي الكتاب وضعيه في `data/pdfs/book/`
3. ثبّتي المكتبات:
   ```bash
   pip install -r requirements.txt
   ```
4. شغّلي المراحل بالترتيب:
   ```bash
   python 01_documents.py
   python 02_preprocessing.py
   python 03_chunking.py
   python 04_vector_representation.py
   python 05_create_chroma_store.py
   ```
   كل خطوة بتطبع تأكيد (✅) وعدد الصفوف/الملفات اللي تكوّنت. لو حصل خطأ في `01_documents.py` بخصوص
   عدد الملفات vs عدد عناصر `PAPER_METADATA_LIST`، راجعي ترتيب الملفات المطبوع وترتيب القائمة في نفس الملف.

بعد الخطوة دي، هيكون عندك:
```
data/documents.parquet
data/chunks.parquet
data/chunk_embeddings.npy
data/embedding_model_name.txt
chroma_db/   (فولدر)
```

## الخطوة 2 — تجربة محلية (اختياري لكن مفيد)

```bash
export GEMINI_API_KEY="مفتاحك"     # أو حطيه في .streamlit/secrets.toml
streamlit run streamlit_app.py
```
لو ظهر التطبيق شغال على `localhost:8501` وجاوب صحيح، يبقى جاهزة للرفع.

## الخطوة 3 — الرفع على GitHub

```bash
git init
git add .
git commit -m "RAG pipeline + Streamlit app"
git branch -M main
git remote add origin https://github.com/USERNAME/REPO_NAME.git
git push -u origin main
```

**تأكدي إن دول اترفعوا فعلاً** (مش مستثنين بالغلط):
```bash
git ls-files data/ chroma_db/
```
لازم تشوفي `data/chunks.parquet`, `data/chunk_embeddings.npy`, `data/embedding_model_name.txt`, وملفات جوه `chroma_db/`.
(الـ PDFs الأصلية في `data/pdfs/` مش محتاجة تترفع — مستثناة في `.gitignore` عمدًا لتوفير المساحة.)

## الخطوة 4 — مفتاح Gemini المجاني

1. [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey) → Create API Key (بدون بطاقة ائتمان).

## الخطوة 5 — النشر على Streamlit Community Cloud

1. [share.streamlit.io](https://share.streamlit.io) → سجّلي دخول بحساب GitHub → **New app**.
2. اختاري الـ repo، وفي **Main file path** اكتبي: `streamlit_app.py`
3. **Advanced settings → Secrets**:
   ```toml
   GEMINI_API_KEY = "مفتاحك هنا"
   ```
4. **Deploy**. هتاخد 3-5 دقايق أول مرة (بتنزيل موديل الـ embeddings).
5. هتحصلي على لينك دائم زي: `https://your-app-name.streamlit.app`

## تحديث المشروع بعد أي تعديل في الداتا

```bash
python 01_documents.py && python 02_preprocessing.py && python 03_chunking.py \
  && python 04_vector_representation.py && python 05_create_chroma_store.py
git add data/ chroma_db/
git commit -m "Update data"
git push
```
Streamlit Cloud بيعمل إعادة نشر تلقائي بعد كل push.
