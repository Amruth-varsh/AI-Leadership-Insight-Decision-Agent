"""
debug_query.py — Step-by-step RAG pipeline trace for a single question.
Run: python debug_query.py
"""

import os
import sys
import textwrap
import numpy as np

# ── tweak this question ──────────────────────────────────────────────────────
QUESTION = "What was Adobe's total revenue for FY2025?"
# ────────────────────────────────────────────────────────────────────────────

def hr(title="", ch="─"):
    width = 80
    if title:
        pad = (width - len(title) - 2) // 2
        print(f"\n{ch * pad} {title} {ch * (width - pad - len(title) - 2)}\n")
    else:
        print(ch * width)


# ── Step 0: Config ───────────────────────────────────────────────────────────
hr("STEP 0 · Configuration", "═")
from config import (
    GROQ_API_KEY, MODEL_NAME, EMBEDDING_MODEL,
    TEMPERATURE, TOP_K_RESULTS, CHUNK_SIZE, CHUNK_OVERLAP,
    FAISS_INDEX_PATH,
)
print(f"  MODEL_NAME       : {MODEL_NAME}")
print(f"  EMBEDDING_MODEL  : {EMBEDDING_MODEL}")
print(f"  TEMPERATURE      : {TEMPERATURE}")
print(f"  TOP_K_RESULTS    : {TOP_K_RESULTS}")
print(f"  CHUNK_SIZE       : {CHUNK_SIZE}")
print(f"  CHUNK_OVERLAP    : {CHUNK_OVERLAP}")
print(f"  FAISS_INDEX_PATH : {FAISS_INDEX_PATH}")
print(f"  GROQ_API_KEY     : {GROQ_API_KEY[:12]}... (truncated)")


# ── Step 1: Load Vector Store ─────────────────────────────────────────────────
hr("STEP 1 · Load FAISS Vector Store", "═")
from src.embeddings import load_vector_store, build_vector_store, _get_embedding_instance
from src.ingest import load_documents

vector_store = load_vector_store()
if vector_store is None:
    print("  ⚠  No existing index found — building from PDFs in data/ ...")
    chunks = load_documents("./data")
    vector_store = build_vector_store(chunks)
else:
    # Inspect the index
    index = vector_store.index
    print(f"  ✔  Index loaded from : {FAISS_INDEX_PATH}")
    print(f"  Total vectors stored : {index.ntotal}")
    print(f"  Vector dimensions    : {index.d}")
    print(f"  Distance strategy    : Cosine (inner product on L2-normalised vecs)")


# ── Step 2: Embed the Question ────────────────────────────────────────────────
hr("STEP 2 · Embed the Question", "═")
print(f'  Question: "{QUESTION}"\n')
embed_model = _get_embedding_instance()
q_vector = embed_model.embed_query(QUESTION)
arr = np.array(q_vector)
print(f"  Embedding model  : {EMBEDDING_MODEL}")
print(f"  Vector length    : {len(q_vector)} dimensions")
print(f"  Vector norm      : {np.linalg.norm(arr):.6f}  (should be ~1.0 for cosine)")
print(f"  Min value        : {arr.min():.6f}")
print(f"  Max value        : {arr.max():.6f}")
print(f"  Mean value       : {arr.mean():.6f}")
print(f"  First 8 values   : {[round(v, 5) for v in q_vector[:8]]}")


# ── Step 3: FAISS Similarity Search ──────────────────────────────────────────
hr("STEP 3 · FAISS Similarity Search", "═")
print(f"  Searching for top-{TOP_K_RESULTS} most similar chunks ...\n")

# Use similarity_search_with_score to get distances too
results_with_scores = vector_store.similarity_search_with_score(QUESTION, k=TOP_K_RESULTS)

print(f"  {'#':<3}  {'Score':>8}  {'Source':<45}  {'Page':>4}")
print(f"  {'─'*3}  {'─'*8}  {'─'*45}  {'─'*4}")
for i, (doc, score) in enumerate(results_with_scores, 1):
    source = os.path.basename(doc.metadata.get("source", "unknown"))
    page   = doc.metadata.get("page", "?")
    # FAISS cosine: score is L2 distance on normalised vectors; lower = more similar
    # Convert to cosine similarity: cos_sim = 1 - (score / 2)
    cos_sim = 1 - (score / 2)
    print(f"  {i:<3}  {cos_sim:>8.4f}  {source:<45}  {page:>4}")


# ── Step 4: Show Retrieved Chunks ────────────────────────────────────────────
hr("STEP 4 · Retrieved Chunk Contents", "═")
results = [doc for doc, _ in results_with_scores]

for i, (doc, score) in enumerate(results_with_scores, 1):
    source  = os.path.basename(doc.metadata.get("source", "unknown"))
    page    = doc.metadata.get("page", "?")
    cos_sim = 1 - (score / 2)
    content = doc.page_content.strip()

    hr(f"Chunk {i} of {TOP_K_RESULTS}  |  cos_sim={cos_sim:.4f}  |  {source}  p.{page}", "·")
    print(f"  Characters in chunk : {len(content)}")
    print(f"  Content preview:\n")
    # Wrap and indent content
    for line in textwrap.wrap(content, width=74):
        print(f"    {line}")


# ── Step 5: Assemble the Full Prompt ─────────────────────────────────────────
hr("STEP 5 · Full Prompt Sent to LLM", "═")
from src.agent import PROMPT_TEMPLATE, _format_context

context_text = _format_context(results)
full_prompt  = PROMPT_TEMPLATE.format(context=context_text, question=QUESTION)

print(f"  Total prompt length : {len(full_prompt)} characters  (~{len(full_prompt)//4} tokens est.)")
print(f"  Context blocks      : {len(results)}")
print()
print("  ┌─ FULL PROMPT ─────────────────────────────────────────────────────")
for line in full_prompt.split("\n"):
    print(f"  │  {line}")
print("  └───────────────────────────────────────────────────────────────────")


# ── Step 6: LLM Call ─────────────────────────────────────────────────────────
hr("STEP 6 · Groq LLM Inference", "═")
import time
from src.agent import _get_llm, _clean_response

print(f"  Model    : {MODEL_NAME}")
print(f"  Temp     : {TEMPERATURE}")
print(f"  Calling Groq API ...\n")

t0 = time.time()
llm = _get_llm()
response = llm.invoke(full_prompt)
elapsed = time.time() - t0

raw_content = response.content if hasattr(response, "content") else str(response)

print(f"  ✔  Response received in {elapsed:.2f}s")
print(f"  Raw response length : {len(raw_content)} characters")

# Check for think tokens
import re
think_blocks = re.findall(r"<think>.*?</think>", raw_content, flags=re.DOTALL)
if think_blocks:
    print(f"  ⚠  Found {len(think_blocks)} <think> block(s) — will be stripped")
    for b in think_blocks:
        print(f"     Think block ({len(b)} chars): {b[:120]}...")
else:
    print(f"  ✔  No <think> blocks detected")

print()
print("  ┌─ RAW LLM OUTPUT ──────────────────────────────────────────────────")
for line in raw_content.split("\n"):
    print(f"  │  {line}")
print("  └───────────────────────────────────────────────────────────────────")


# ── Step 7: Clean & Final Answer ─────────────────────────────────────────────
hr("STEP 7 · Final Cleaned Answer", "═")
final_answer = _clean_response(raw_content)
print(final_answer)

hr("", "═")
print(f"  Total pipeline time: {elapsed:.2f}s  (excludes index-load & embedding time)")
hr("", "═")
