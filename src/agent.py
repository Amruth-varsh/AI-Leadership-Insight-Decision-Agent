import logging
import re

from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from config import MODEL_NAME, TOP_K_RESULTS, GROQ_API_KEY, TEMPERATURE

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a senior financial analyst AI assistant for company leadership.

QUESTION: {question}

==============================
STEP 1 – CLASSIFY FIRST, THEN RESPOND
==============================
Classify the question as CASUAL or ANALYTICAL.

CASUAL = greetings, farewells, small talk, thanks, questions about who you are or how you are — including typos like "how areu", "wassup", "heyy", "wuts up", "byee".
ANALYTICAL = anything about financials, business, strategy, metrics, operations, or documents.

IF CASUAL:
- Output one friendly sentence. Nothing else.
- No "Answer:", no "Sources:", no notes, no explanations, no extra sentences.
- STOP immediately. Do not continue to MODE 2.

Casual examples (output exactly like this, nothing more):
- "hi" → Hello! How can I assist you today?
- "how are you" → I'm doing well, thank you!
- "how areu" → I'm doing well, thank you!
- "wassup" → Hello! How can I assist you today?
- "thanks" → You're welcome!
- "who are you" → I'm an AI assistant here to help with your financial questions!
- "what are you" → I'm an AI assistant here to help with your financial questions!

IF ANALYTICAL → continue below.

==============================
MODE 2 – Analyst (ANALYTICAL only)
==============================
DOCUMENT EXCERPTS:
{context}

STRICT RULES:
1. Use only the provided document excerpts; do not use outside knowledge.
2. If context is partial, summarize only what is shown.
3. Cite exact numbers, percentages, and figures if available.
4. If no relevant information exists, respond: "Information not found in documents."
5. Do not add reasoning, preamble, or step-by-step thinking.

**Answer:**
3–5 sentences answering the question directly, lead with the most important finding.

**Sources:**
- [filename], Page [X] – [brief factual description]
- [filename], Page [X] – [brief factual description]
""",
)




def _get_llm():
    """Return a Groq-hosted LLM instance."""
    return ChatGroq(
        model=MODEL_NAME,
        api_key=GROQ_API_KEY,
        temperature=TEMPERATURE,
    )


def _clean_response(text):
    """Strip any reasoning/thinking blocks from model output."""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned.strip()


def _format_context(results):
    """Format retrieved documents into a context string for the prompt."""
    parts = []
    for doc in results:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        parts.append(f"[Source: {source}, Page {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def answer_question(question, vector_store):
    """Retrieve relevant chunks and generate an answer using Groq LLM."""
    if vector_store is None:
        return "Vector store not available. Please add documents to the data/ folder and restart."

    try:
        results = vector_store.similarity_search(question, k=TOP_K_RESULTS)
    except Exception as e:
        return f"Error during retrieval: {e}"

    if not results:
        return "Information not found in documents."

    logger.debug(f"Retrieved {len(results)} chunks for: {question}")

    context_text = _format_context(results)
    full_prompt = PROMPT_TEMPLATE.format(context=context_text, question=question)

    try:
        llm = _get_llm()
        response = llm.invoke(full_prompt)
        answer_text = response.content if hasattr(response, "content") else str(response)
        return _clean_response(answer_text) or "The model returned an empty response."
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return f"LLM error: {e}"
