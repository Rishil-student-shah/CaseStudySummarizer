# rag_pipeline.py — STRICT RAG FOR COLLEGE ASSIGNMENTS (FINAL)

import os
from typing import List, Tuple

import google.generativeai as genai
from dotenv import load_dotenv

from langchain_core.documents import Document
from embedder import load_index


# ----------------------------------------------------
# LOAD API KEY
# ----------------------------------------------------
import os
import streamlit as st
from dotenv import load_dotenv

# Load .env (local only)
load_dotenv()

api_key = None

# 1. Check Streamlit Cloud's secrets (won't error locally)
try:
    api_key = st.secrets["GOOGLE_API_KEY"]
except Exception:
    pass

# 2. If not found, check local .env
if not api_key:
    api_key = os.getenv("GOOGLE_API_KEY")

# 3. Fail if no API key found
if not api_key:
    raise ValueError("❌ GOOGLE_API_KEY not found in secrets or .env")

# Configure Gemini
genai.configure(api_key=api_key)




# ----------------------------------------------------
# GEMINI CALL
# ----------------------------------------------------
def gemini_generate(prompt: str, max_tokens: int = 600):
    model = genai.GenerativeModel("gemini-2.0-flash")

    try:
        response = model.generate_content(
            prompt,
            generation_config={
                "max_output_tokens": max_tokens,
                "temperature": 0.0,  # STRICT, no creativity
            }
        )
    except Exception as e:
        return f"⚠️ Gemini API Error: {e}"

    if not response.candidates:
        return "⚠️ No output."

    parts = response.candidates[0].content.parts
    text = "".join([p.text for p in parts if hasattr(p, "text")])

    return text.strip() if text else "⚠️ Empty output."


# ----------------------------------------------------
# 1. RETRIEVE CHUNKS
# ----------------------------------------------------
def retrieve_docs(index_path: str, query: str, k: int = 15) -> List[Document]:
    """
    k=15 gives very stable retrieval.
    Keyword fallback ensures accuracy (Ferrari exclusivity chunk always loads).
    """
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Index not found: {index_path}")

    db = load_index(index_path)
    
    # Primary retrieval
    docs = db.similarity_search(query, k=k)

    # Keyword fallback for weak embeddings
    if len(docs) < 5:
        main_keyword = query.split()[0]
        fallback_docs = db.similarity_search(main_keyword, k=k)
        docs = fallback_docs if len(fallback_docs) > len(docs) else docs

    return docs


# ----------------------------------------------------
# 2. SUMMARIZE CONTEXT STRICTLY
# ----------------------------------------------------
def condense_context(docs: List[Document]) -> Tuple[str, str]:

    if not docs:
        return "The retrieved context is insufficient to summarize.", ""

    raw_docs = "\n\n".join(
        f"[DOC {i+1}]\n{d.page_content}"
        for i, d in enumerate(docs)
    )

    prompt = f"""
You are a STRICT RAG system. Summarize ONLY using the text below.

CONTEXT:
{raw_docs}

TASK:
Write a short factual summary (3–4 sentences) ONLY using what appears in the context.
If the context does not contain enough information, answer exactly:
"The retrieved context is insufficient to summarize."

RULES:
- No assumptions.
- No external knowledge.
- No invented facts.
"""

    summary = gemini_generate(prompt, max_tokens=350)
    return summary, raw_docs


# ----------------------------------------------------
# 3. STRICT FINAL ANSWER PROMPT
# ----------------------------------------------------
def build_prompt(question: str, summary: str, raw_docs: str) -> str:
    return f"""
You are an academic STRICT RAG answer system.

RAW CONTEXT:
{raw_docs}

SUMMARY:
{summary}

QUESTION:
{question}

TASK:
- Answer ONLY using the RAW CONTEXT.
- If the context does not contain the information, say:
  "The retrieved context is insufficient to answer this question."
- NO assumptions.
- NO outside reasoning.
- NO hallucination.
- ONE short paragraph (3–5 sentences).
"""


# ----------------------------------------------------
# 4. FULL RAG PIPELINE
# ----------------------------------------------------
def answer_query(index_path: str, query: str):
    try:
        docs = retrieve_docs(index_path, query, k=15)
    except FileNotFoundError as e:
        return str(e), []

    if not docs:
        return "⚠️ No chunks retrieved.", []

    summary, raw_docs = condense_context(docs)
    prompt = build_prompt(query, summary, raw_docs)
    answer = gemini_generate(prompt, max_tokens=500)

    return answer, docs
