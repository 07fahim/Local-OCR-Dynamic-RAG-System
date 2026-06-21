"""
Cross-lingual embedding sanity check for docs/explain.md Section 2.

Embeds a Bangla sentence and its English translation with the configured model
and prints their cosine similarity. A working multilingual model (BGE-M3) should
score these well above ~0.5, and clearly higher than an unrelated pair.

Usage:
    python -m scripts.embed_check
"""
from __future__ import annotations

from backend.embeddings import get_embedder


def cosine(a: list[float], b: list[float]) -> float:
    # vectors are already L2-normalised by the embedder -> dot product == cosine
    return sum(x * y for x, y in zip(a, b))


def main() -> None:
    pairs = [
        ("বাংলাদেশের রাজধানী ঢাকা।", "The capital of Bangladesh is Dhaka."),
        ("আমি ভাত খাই।", "I eat rice."),
    ]
    unrelated = ("বাংলাদেশের রাজধানী ঢাকা।", "Quarterly revenue grew by twelve percent.")

    emb = get_embedder()
    print("Cross-lingual related pairs (expect HIGH):")
    for bn, en in pairs:
        v = emb.embed([bn, en])
        print(f"  cos = {cosine(v[0], v[1]):.4f}  | {bn}  <->  {en}")

    v = emb.embed([unrelated[0], unrelated[1]])
    print("\nUnrelated pair (expect LOW):")
    print(f"  cos = {cosine(v[0], v[1]):.4f}  | {unrelated[0]}  <->  {unrelated[1]}")


if __name__ == "__main__":
    main()
