"""
ChromaDB persistent vector store + the hybrid-search mechanism.

THE HYBRID MECHANISM (feeds docs/explain.md Section 3):
Manual metadata filters are applied as a Boolean `where` PRE-FILTER. ChromaDB
restricts the candidate pool to chunks whose metadata satisfies the filter
BEFORE computing vector similarity, so the embedding ranking only ever operates
over chunks that already pass the hard constraints (filter-then-search). This is
"hybrid": a hard Boolean filter (metadata) + soft semantic ranking (vectors).

Filter construction notes:
* Only filters that are actually present are included — we never send an empty
  `where={}` (Chroma rejects it).
* Current ChromaDB requires an explicit `$and` when combining 2+ conditions; a
  single condition is passed as a flat one-key dict.
* Date range filters use `$gte` / `$lte` on the numeric `document_date_int`
  (YYYYMMDD, e.g. 20250314). ChromaDB 0.5.x rejects string operands for range
  operators, so the date is compared as a zero-padded integer (which preserves
  chronological order). Documents without a date omit `document_date_int`
  entirely and are therefore excluded from any date-range filter.
"""
from __future__ import annotations

from typing import Any

import chromadb

from .chunking import date_to_int
from .config import settings

COLLECTION_NAME = "documents"


def build_where(filters: dict | None) -> dict | None:
    """
    Translate request filters into a ChromaDB `where` clause.

    Accepted keys: document_type, language, date_from, date_to.
    Returns None when no usable filter is present (caller must omit `where`).
    """
    if not filters:
        return None

    conditions: list[dict] = []

    doc_type = filters.get("document_type")
    if doc_type:
        conditions.append({"document_type": {"$eq": doc_type}})

    language = filters.get("language")
    if language:
        if language == "mixed":
            conditions.append({"language": {"$eq": "mixed"}})
        else:
            # "bn" also includes "mixed" (bilingual chunks contain Bangla);
            # "en" also includes "mixed" (bilingual chunks contain English).
            conditions.append({"language": {"$in": [language, "mixed"]}})

    # Range filters compare the numeric document_date_int (YYYYMMDD); ChromaDB
    # 0.5.x requires int/float operands for $gte/$lte.
    date_from = date_to_int(filters.get("date_from"))
    date_to = date_to_int(filters.get("date_to"))
    if date_from is not None:
        conditions.append({"document_date_int": {"$gte": date_from}})
    if date_to is not None:
        conditions.append({"document_date_int": {"$lte": date_to}})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


class VectorStore:
    def __init__(self, persist_directory: str | None = None) -> None:
        self.persist_directory = persist_directory or settings.chroma_dir
        self._client = chromadb.PersistentClient(path=self.persist_directory)
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    # ---- ingestion ---- #
    def ingest_chunks(self, chunks: list[dict], embeddings: list[list[float]]) -> int:
        if not chunks:
            return 0
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings length mismatch")

        self._collection.upsert(
            ids=[c["id"] for c in chunks],
            documents=[c["document"] for c in chunks],
            embeddings=embeddings,
            metadatas=[c["metadata"] for c in chunks],
        )
        return len(chunks)

    # ---- query ---- #
    def query(
        self, query_embedding: list[float], filters: dict | None, top_k: int = 5
    ) -> list[dict]:
        """
        Filter-then-search. Returns up to top_k results, each:
            { "id", "chunk_text", "metadata", "distance" }
        Fewer than top_k may be returned if the filter is narrow — the caller
        surfaces that (see /search warning).
        """
        where = build_where(filters)
        kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where is not None:
            kwargs["where"] = where  # never pass an empty dict

        res = self._collection.query(**kwargs)

        ids = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]

        out: list[dict] = []
        for i in range(len(ids)):
            out.append(
                {
                    "id": ids[i],
                    "chunk_text": docs[i] if i < len(docs) else "",
                    "metadata": metas[i] if i < len(metas) else {},
                    "distance": dists[i] if i < len(dists) else None,
                }
            )
        return out

    def count(self) -> int:
        return self._collection.count()

    def delete_document(self, document_id: str) -> None:
        self._collection.delete(where={"document_id": {"$eq": document_id}})


_store: VectorStore | None = None


def get_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore()
    return _store
