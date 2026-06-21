"""
Local dense-embedding wrapper.

WHY BGE-M3 (feeds docs/explain.md Section 2):
* Multilingual model that projects Bangla AND English into ONE shared vector
  space. A monolingual English model gives geometrically unrelated vectors for
  Bangla text, so a Bangla query vs an English chunk (or vice versa) scores
  near-zero even when the meaning matches. BGE-M3 makes cross-lingual retrieval
  work — a Bangla document can be retrieved by an English query.
* BGE-M3 also offers sparse and multi-vector (ColBERT) outputs. For this build
  we use ONLY the dense embedding, for simplicity — a deliberate scope choice,
  not an oversight. Dense + ChromaDB cosine search is sufficient for the
  hybrid-filter requirement.
* No external API: the model is downloaded once and runs on-device (CPU or GPU).
"""
from __future__ import annotations

from typing import Any

from .config import settings


class LocalEmbedder:
    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.embedding_model
        self._model: Any = None  # lazy-loaded on first use

    def _ensure_model(self) -> None:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            print(f"[embed] Loading embedding model '{self.model_name}' locally...")
            self._model = SentenceTransformer(self.model_name)
            print("[embed] Embedding model ready.")

    def embed(self, texts: list[str], batch_size: int = 8) -> list[list[float]]:
        """
        Embed a list of texts into normalised dense vectors.

        Vectors are L2-normalised so that a downstream cosine/IP search behaves
        identically. batch_size is conservative to stay within CPU RAM.
        """
        if not texts:
            return []
        self._ensure_model()
        vectors = self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vectors.tolist()

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


# Process-wide singleton so the model loads exactly once.
_embedder: LocalEmbedder | None = None


def get_embedder() -> LocalEmbedder:
    global _embedder
    if _embedder is None:
        _embedder = LocalEmbedder()
    return _embedder
