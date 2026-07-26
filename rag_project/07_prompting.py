"""
07_prompting.py
================
Section 15 من النوت بوك الأصلي: ثلاث نسخ من الـ prompt (weak / better /
strict) بتاخد السؤال + الـ context وتبني الـ prompt النهائي اللي بيتبعت
للموديل. التطبيق (streamlit_app.py) بيستخدم build_strict_prompt بشكل
افتراضي لأنها بتفرض الاستشهاد بالمصادر واللغة المناسبة.
"""


def build_weak_prompt(query, context_text):
    """Simple baseline prompt."""
    return f"""
Answer the following question using the provided context.

Question:
{query}

Context:
{context_text}

Answer:
""".strip()


def build_better_prompt(query, context_text):
    """Improved prompt with citation guidance."""
    return f"""
You are a research assistant specialized in digital sociology.

Use ONLY the provided context to answer the question.

If the context does not contain enough information,
say so clearly instead of guessing.

Whenever you state a factual claim, cite the source
using this format:

(Author, Year)

Question:
{query}

Context:
{context_text}

Answer:
""".strip()


def build_strict_prompt(query, context_text):
    """Strict prompt used in the final RAG pipeline."""
    return f"""
You are a research assistant specialized in digital sociology
and the impact of digitalization on social interaction.

Rules:

1. Use ONLY the information inside the provided context.
2. Do NOT use prior knowledge.
3. If the context is insufficient, respond exactly with:

"لا توجد معلومات كافية في المصادر للإجابة على هذا السؤال."

4. Every factual statement must include a citation in the format:

(Author, Year)

5. If different sources disagree, explicitly mention the disagreement.

6. Answer in the same language as the user's question.
   - Arabic question -> Arabic answer.
   - English question -> English answer.

Output format:

Short Answer:
(2-4 sentences)

Sources Used:
- (Author, Year) -- Title

Question:
{query}

Context:
{context_text}

Answer:
""".strip()
