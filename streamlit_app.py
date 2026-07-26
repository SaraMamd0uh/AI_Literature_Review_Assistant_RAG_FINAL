"""
streamlit_app.py
=================
Section 16 من النوت بوك الأصلي: تطبيق Streamlit اللي بياخد سؤال المستخدم،
يبني الـ context من 06_retrieve_context.py، يبني الـ prompt من
07_prompting.py، ويبعتهم لموديل مجاني على OpenRouter.

⚠️ الأمان: الـ API key **لازم** يتحط في Streamlit Secrets (أو environment
variable محليًا) — أبدًا متكتبيهوش مباشرة في الكود، لأن أي حد يشوف الريبو
على GitHub هيقدر ياخده ويستخدمه على حسابك.

إعداد الـ secret على Streamlit Cloud:
    Settings -> Secrets -> ضيفي سطر:
    OPENROUTER_API_KEY = "sk-or-v1-...."

تشغيل محلي:
    export OPENROUTER_API_KEY="sk-or-v1-...."
    streamlit run streamlit_app.py
"""

import importlib.util
import os
from pathlib import Path

import streamlit as st
from openai import OpenAI


def _load_module(filename, alias):
    path = Path(__file__).parent / filename
    spec = importlib.util.spec_from_file_location(alias, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

retrieve_context = _load_module("06_retrieve_context.py", "retrieve_context")
prompting = _load_module("07_prompting.py", "prompting")

FREE_MODELS = [
    "meta-llama/llama-3.3-70b-instruct:free",
    "qwen/qwen3-coder:free",
    "openai/gpt-oss-120b:free",
]

st.set_page_config(page_title="AI Literature Review Assistant", page_icon="📚")


@st.cache_resource(show_spinner="بتحمّل قاعدة المعرفة...")
def get_artifacts():
    return retrieve_context.load_artifacts("rag_export")


def get_api_key():
    # Prefer Streamlit Secrets (Streamlit Cloud), fall back to env var (local dev)
    return st.secrets.get("OPENROUTER_API_KEY", os.environ.get("OPENROUTER_API_KEY"))


def main():
    st.title("📚 AI Literature Review Assistant")
    st.caption("اسألي بالعربي أو بالإنجليزي عن أي حاجة في المستندات المرفوعة.")

    api_key = get_api_key()
    if not api_key:
        st.error(
            "مفيش OPENROUTER_API_KEY متضاف. ضيفيه في Streamlit Secrets "
            "(Settings → Secrets) أو كـ environment variable محليًا."
        )
        st.stop()

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    model = st.selectbox("اختاري الموديل (مجاني):", FREE_MODELS, index=0)

    artifacts = get_artifacts()

    query = st.text_input("اكتبي سؤالك:")

    if query:
        with st.spinner("بندوّر في المصادر..."):
            context_package = retrieve_context.build_context_package(query, artifacts)
            context_text = context_package["context_text"]

        if not context_text:
            st.warning("مفيش مصادر مرتبطة كفاية بالسؤال ده.")
            return

        prompt = prompting.build_strict_prompt(query, context_text)

        with st.spinner("بيكتب الإجابة..."):
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )

        st.markdown(response.choices[0].message.content)

        with st.expander(f"المصادر المستخدمة ({context_package['num_sources']} مصدر, {context_package['num_chunks']} مقطع)"):
            for _, row in context_package["selected_df"].iterrows():
                st.markdown(f"**{row['title']}** — {row['authors']} ({row['publication_year']})")
                st.caption(row["chunk_text"][:300] + "...")


if __name__ == "__main__":
    main()
