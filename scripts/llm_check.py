"""
Answer-generation smoke test — exercises generate_answer() with fixed sample
chunks so you can verify the configured LLM_MODE (local / none) works
WITHOUT installing the heavy OCR + embedding stack.

Usage:
    # uses LLM_MODE from .env
    python -m scripts.llm_check
    # or force a mode:
    LLM_MODE=local python -m scripts.llm_check
    LLM_MODE=none  python -m scripts.llm_check

For LLM_MODE=local: install Ollama (https://ollama.com/download) and run
`ollama pull qwen2.5:7b` first.
"""
from __future__ import annotations

from backend.config import settings
from backend.rag import generate_answer

# Simulated retrieved chunks (as if returned from a Bangla invoice + an English report).
SAMPLE_CHUNKS = [
    "মা-বাবার ফল ভান্ডার। আম (ব্যাংগানাপল্লী) ১ কেজি ১৪০ টাকা। কলা (সাগর) ১ ডজন ৮০ টাকা।",
    "The total invoice amount is 5000 BDT, payable within 30 days of issue.",
]

QUERIES = [
    ("আমের দাম কত?", "Bangla query"),
    ("What is the invoice total and the payment term?", "English query"),
    ("What is the capital of France?", "Out-of-context (expect 'I don't know')"),
]


def main() -> None:
    print(f"LLM_MODE = {settings.llm_mode}")
    if settings.llm_mode == "local":
        print(f"Ollama   = {settings.ollama_base_url}  model={settings.ollama_model}")
    print("=" * 60)
    for query, label in QUERIES:
        print(f"\n--- {label} ---\nQ: {query}")
        print("A:", generate_answer(query, SAMPLE_CHUNKS, settings.llm_mode))


if __name__ == "__main__":
    main()
