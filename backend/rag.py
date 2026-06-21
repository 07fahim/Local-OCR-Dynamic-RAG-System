"""
Answer generation (the "G" in RAG).

FULLY LOCAL: every mode here runs on-machine — no external/commercial API is
ever called. The model only ever receives the user's query text + the
already-retrieved chunk text; the original file, image, and full raw OCR output
never leave the pipeline.

Modes:
  local -> synthesise a grounded answer via a local Ollama model (default).
           If Ollama is unreachable, fall back to `none` behaviour with a note.
  none  -> no LLM. Return the retrieved, cited chunks verbatim, clearly
           labelled as unsynthesised. Zero dependencies, zero network.
"""
from __future__ import annotations

import re

from .config import settings

# The sample invoices abbreviate Bangla Taka as "(ট)" in table column headers, so
# the model sometimes copies that abbreviation next to a value (e.g. "১৭০ (ট)").
# This regex catches the parenthesised abbreviation so we can expand it to the
# full currency word deterministically (see _normalize_currency).
_TAKA_PAREN_RE = re.compile(r"\(\s*ট\s*\)")
_BANGLA_CHAR_RE = re.compile(r"[ঀ-৿]")

_SYSTEM_PROMPT_BASE = (
    "You are a strict document-grounded question-answering system. You answer "
    "ONLY from the CONTEXT passages given to you. The context may be in Bangla, "
    "English, or both.\n"
    "\n"
    "ABSOLUTE RULES (follow every one):\n"
    "1. GROUNDING: Use ONLY information explicitly written in the context. Never "
    "use outside knowledge, training data, assumptions, or guesses. If the "
    "context does not contain enough information to answer, reply with EXACTLY "
    "this sentence and nothing else: "
    "\"I don't know based on the provided documents.\"\n"
    "2. NO INFERENCE: Do not infer, estimate, calculate, sum, convert, or combine "
    "values. Only state a number or fact if it appears verbatim in the context. "
    "If a total/result is not literally written, say you don't know — do not "
    "compute it yourself.\n"
    "3. VERBATIM NUMBERS: Copy all numeric values, dates, codes, and proper nouns "
    "EXACTLY as they appear, digit for digit. Do not translate, round, or reformat "
    "the digits (e.g. keep \"৫,৩১০\" and \"১৭০\" exactly as written).\n"
    "4. CURRENCY WORDING: The money unit in these documents is the Bangla Taka. "
    "The marks \"৳\", \"(ট)\", \"টা\", \"Tk\", and \"BDT\" all mean Taka. Keep the "
    "number exact, but ALWAYS write the currency as a full word — use \"টাকা\" "
    "when answering in Bangla and \"BDT\" when answering in English. NEVER output "
    "the bare abbreviation \"(ট)\" or a lone \"ট\". For example, a price shown as "
    "\"১৭০\" under a \"(ট)\" column must be written \"১৭০ টাকা\" (Bangla) or "
    "\"170 BDT\" (English).\n"
    "5. {lang_instruction}\n"
    "6. DIRECT & CONCISE: Give only the answer. No preamble, no explanation of "
    "your reasoning, no phrases like \"according to the context\" or \"document "
    "[1]\". If one short sentence or value answers it, give just that.\n"
    "7. NO FABRICATION: If you are not fully certain the answer is supported by "
    "the context, choose the \"I don't know\" response rather than risk being "
    "wrong. A correct \"I don't know\" is better than a confident guess."
)

_LANG_INSTRUCTIONS = {
    "bn": "You MUST answer entirely in Bangla (বাংলা), even if the question is in English.",
    "en": "You MUST answer entirely in English, even if the context is in Bangla.",
    "mixed": "Answer using both Bangla and English as appropriate.",
}
_LANG_DEFAULT = (
    "Answer in the SAME language as the QUESTION (English question -> "
    "English answer; Bangla question -> Bangla answer)."
)


def _build_system_prompt(language_filter: str | None) -> str:
    lang_instruction = _LANG_INSTRUCTIONS.get(language_filter or "", _LANG_DEFAULT)
    return _SYSTEM_PROMPT_BASE.format(lang_instruction=lang_instruction)


def _normalize_currency(text: str, language_filter: str | None) -> str:
    """
    Deterministically expand the Taka abbreviation "(ট)" / "৳" that the OCR table
    headers carry into the full currency word, so an answer never shows a bare
    "(ট)". The numeric value is never touched — only the currency token. This is a
    guaranteed safety net on top of the prompt rule (a small LLM does not always
    obey "never output (ট)").
    """
    if not text:
        return text
    if language_filter == "en":
        word = "BDT"
    elif language_filter in ("bn", "mixed"):
        word = "টাকা"
    else:
        # "any"/default: choose by the script the answer itself is written in.
        word = "টাকা" if _BANGLA_CHAR_RE.search(text) else "BDT"
    text = _TAKA_PAREN_RE.sub(word, text)
    text = text.replace("৳", word)
    # The model can emit the abbreviation more than once (e.g. "১৭০ (ট) (ট)"),
    # which would expand to a repeated currency word — collapse those.
    text = re.sub(r"(টাকা)(\s+টাকা)+", r"\1", text)
    text = re.sub(r"\bBDT(\s+BDT)+\b", "BDT", text)
    # Collapse any double spaces the substitution may have produced.
    return re.sub(r"[ \t]{2,}", " ", text).strip()


def _format_context(context_chunks: list[str]) -> str:
    return "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(context_chunks))


def _build_user_prompt(query: str, context_chunks: list[str]) -> str:
    return (
        "CONTEXT (the only information you may use):\n"
        f"{_format_context(context_chunks)}\n\n"
        f"QUESTION: {query}\n\n"
        "Answer the question using ONLY the context above. If the context does "
        "not explicitly contain the answer, reply exactly: \"I don't know based "
        "on the provided documents.\" Do not add anything that is not in the "
        "context."
    )


def _answer_none(context_chunks: list[str]) -> str:
    if not context_chunks:
        return "No matching content was found for this query and filter combination."
    body = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(context_chunks))
    return (
        "(LLM synthesis disabled — LLM_MODE=none. Showing the most relevant "
        "retrieved passages verbatim.)\n\n" + body
    )


def _answer_local(query: str, context_chunks: list[str], language_filter: str | None = None) -> str:
    """Synthesise via a local Ollama server. Falls back to `none` if unreachable."""
    import json
    import urllib.error
    import urllib.request

    url = f"{settings.ollama_base_url.rstrip('/')}/api/chat"
    payload = {
        "model": settings.ollama_model,
        "stream": False,
        # Keep the model resident in RAM for 30 min after each call. Ollama unloads
        # idle models after ~5 min by default; on CPU a cold reload of a 7B model
        # can exceed the request timeout. keep_alive avoids reloading between
        # queries (important during a live demo).
        "keep_alive": "30m",
        "messages": [
            {"role": "system", "content": _build_system_prompt(language_filter)},
            {"role": "user", "content": _build_user_prompt(query, context_chunks)},
        ],
        # Greedy, fully deterministic decoding to minimise hallucination:
        #   temperature 0  -> always pick the highest-probability token (no sampling)
        #   top_p / top_k  -> collapse the candidate set to the single best token
        #   seed           -> reproducible output for the same context+question
        #   repeat_penalty -> mild, to avoid degenerate loops without forcing drift
        #   num_ctx 8192   -> fit several retrieved chunks without truncation
        "options": {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 1,
            "seed": 42,
            "repeat_penalty": 1.1,
            "num_ctx": 8192,
        },
    }
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        # Generous timeout: a cold load of a 7B model on CPU (weights into RAM +
        # first-token latency) can take a few minutes. keep_alive keeps subsequent
        # calls fast, but the first one after an idle period must be allowed to load.
        with urllib.request.urlopen(req, timeout=600) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        answer = data["message"]["content"].strip()
        return _normalize_currency(answer, language_filter)
    except (urllib.error.URLError, KeyError, TimeoutError) as exc:
        return (
            f"[LLM_MODE=local requested but Ollama was unreachable ({exc}). "
            "Falling back to local passages.]\n\n" + _answer_none(context_chunks)
        )


def generate_answer(
    query: str,
    context_chunks: list[str],
    mode: str | None = None,
    language_filter: str | None = None,
) -> str:
    """Dispatch to the configured (fully-local) answer-generation mode."""
    mode = (mode or settings.llm_mode or "local").lower()
    if mode == "local":
        return _answer_local(query, context_chunks, language_filter)
    return _answer_none(context_chunks)
