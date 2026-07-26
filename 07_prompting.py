"""
07_prompting.py
=================
Stage 7 of the RAG pipeline (runtime library — imported by streamlit_app.py).

Builds the final prompt sent to the LLM from a retrieved context block,
and calls Google Gemini (free tier) to generate the grounded answer.
"""

from google import genai

DEFAULT_MODEL_NAME = "gemini-2.5-flash"


def build_weak_prompt(query, context_text):
    return f"""Answer the question using the context.

Question:
{query}

Context:
{context_text}
"""


def build_better_prompt(query, context_text):
    return f"""You are a research assistant specialized in digital sociology.

Answer using only the provided context. If the context does not contain
enough information, say so clearly instead of guessing.

Whenever you state a fact, mention its source in this format: (Author, Year).

Question:
{query}

Context:
{context_text}

Answer:"""


def build_strict_prompt(query, context_text):
    return f"""You are a research assistant specialized in digital sociology
and the impact of digitalization on social interaction.

Rules:
1. Use ONLY the information inside <context>. Do not use outside knowledge.
2. If the context is insufficient, respond exactly: "لا توجد معلومات كافية في المصادر للإجابة على هذا السؤال."
3. Every claim must be followed by a citation in the format (Author, Year).
4. If sources conflict, mention the disagreement explicitly instead of picking one silently.
5. Answer in the same language as the question (Arabic question -> Arabic answer, English question -> English answer).

Output format:
- Short direct answer (2-4 sentences)
- "Sources used:" followed by a bullet list of (Author, Year, Title)

<context>
{context_text}
</context>

Question: {query}
"""


PROMPT_BUILDERS = {
    "weak": build_weak_prompt,
    "better": build_better_prompt,
    "strict": build_strict_prompt,
}


def call_llm(prompt, api_key, model_name=DEFAULT_MODEL_NAME):
    """Call Google Gemini (free tier) to generate the final grounded answer."""
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(model=model_name, contents=prompt)
    return response.text


def generate_answer(query, context_text, api_key, prompt_style="strict", model_name=DEFAULT_MODEL_NAME):
    """Build the prompt for the given style and call the LLM in one step."""
    prompt = PROMPT_BUILDERS[prompt_style](query, context_text)
    answer = call_llm(prompt, api_key=api_key, model_name=model_name)
    return answer, prompt
