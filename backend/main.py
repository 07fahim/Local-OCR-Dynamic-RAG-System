"""
FastAPI application — endpoints from PRD Section 8.

  POST /upload     multipart file + optional document_type/document_date
                   -> OCR -> chunk -> embed -> ChromaDB -> SQLite log
  POST /search     query + optional filters -> hybrid retrieval -> answer
  GET  /documents  list everything in the ingestion log
  GET  /           minimal HTML upload/search UI
  GET  /health     liveness + active config

Every pipeline step logs to stdout with timestamps so the demo video shows real
local processing.
"""
from __future__ import annotations

import os
import time
import uuid
from datetime import datetime, timezone

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from . import database
from .chunking import chunk_document
from .config import settings
from .embeddings import get_embedder
from .models import (
    DocumentInfo,
    DocumentListResponse,
    SearchRequest,
    SearchResponse,
    Source,
    UploadResponse,
)
from .ocr import extract_text_from_file
from .rag import generate_answer
from .vectorstore import build_where, get_store

app = FastAPI(title="Local OCR & Dynamic RAG System", version="1.0.0")

_TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")
templates = Jinja2Templates(directory=_TEMPLATES_DIR)


def _log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


@app.on_event("startup")
def _startup() -> None:
    os.makedirs(settings.upload_dir, exist_ok=True)
    database.init_db()
    _log(
        f"Startup ready. OCR=surya EMBEDDING={settings.embedding_model} "
        f"LLM_MODE={settings.llm_mode}"
    )


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "ocr_engine": "surya",
        "embedding_model": settings.embedding_model,
        "llm_mode": settings.llm_mode,
        "chunks_indexed": get_store().count(),
    }


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("upload_search.html", {"request": request})


@app.post("/upload", response_model=UploadResponse)
async def upload(
    file: UploadFile = File(...),
    document_type: str = Form("other"),
    document_date: str | None = Form(None),
) -> UploadResponse:
    document_id = str(uuid.uuid4())
    upload_ts = datetime.now(timezone.utc).isoformat()
    safe_name = os.path.basename(file.filename or "upload")
    saved_path = os.path.join(settings.upload_dir, f"{document_id}__{safe_name}")

    _log(f"UPLOAD start: {safe_name} (doc_id={document_id}, type={document_type})")
    contents = await file.read()
    with open(saved_path, "wb") as fh:
        fh.write(contents)

    database.log_ingestion(
        document_id, safe_name, upload_ts, 0, 0, [], document_date, document_type, "processing"
    )

    try:
        t0 = time.time()
        _log("OCR: extracting text locally...")
        pages = extract_text_from_file(saved_path)
        _log(f"OCR: done, {len(pages)} page(s) in {time.time() - t0:.1f}s")

        chunks = chunk_document(pages, document_id, safe_name, document_date, document_type)
        _log(f"CHUNK: produced {len(chunks)} chunk(s)")
        if not chunks:
            database.set_status(document_id, "error")
            raise HTTPException(status_code=422, detail="No text could be extracted from the document.")

        langs_detected = sorted({c["metadata"]["language"] for c in chunks})

        _log("EMBED: encoding chunks locally...")
        embedder = get_embedder()
        vectors = embedder.embed([c["document"] for c in chunks])
        _log(f"EMBED: {len(vectors)} vector(s) of dim {len(vectors[0]) if vectors else 0}")

        added = get_store().ingest_chunks(chunks, vectors)
        _log(f"VECTORSTORE: upserted {added} chunk(s) into ChromaDB")

        database.log_ingestion(
            document_id, safe_name, upload_ts, len(pages), len(chunks),
            langs_detected, document_date, document_type, "done",
        )
        _log(f"UPLOAD done: doc_id={document_id}")

        return UploadResponse(
            document_id=document_id,
            status="done",
            filename=safe_name,
            page_count=len(pages),
            chunk_count=len(chunks),
            languages_detected=langs_detected,
        )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        database.set_status(document_id, "error")
        _log(f"UPLOAD error: {exc}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest) -> SearchResponse:
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")

    filters = req.filters.model_dump(exclude_none=True)
    top_k = req.top_k or settings.default_top_k
    _log(f"SEARCH: q='{req.query[:60]}' filters={filters} top_k={top_k}")

    embedder = get_embedder()
    q_vec = embedder.embed_one(req.query)

    # Over-fetch, then drop exact-duplicate chunk texts (e.g. the same document
    # ingested more than once) so we return top_k UNIQUE passages. Feeding the LLM
    # the same text several times wastes context and badly slows CPU inference.
    raw_results = get_store().query(q_vec, filters, top_k=top_k * 3)
    seen: set[str] = set()
    results: list[dict] = []
    for r in raw_results:
        key = (r["chunk_text"] or "").strip()
        if key in seen:
            continue
        seen.add(key)
        results.append(r)
        if len(results) >= top_k:
            break
    _log(
        f"SEARCH: {len(raw_results)} retrieved -> {len(results)} unique chunk(s) "
        f"(where={build_where(filters)})"
    )

    warning: str | None = None
    if len(results) == 0:
        warning = "No chunks matched the query under the active filters. Try relaxing the filters."
    elif len(results) < top_k:
        warning = f"Only {len(results)} chunk(s) matched the filters (requested top_k={top_k})."

    context_chunks = [r["chunk_text"] for r in results]
    language_filter = filters.get("language")
    answer = generate_answer(req.query, context_chunks, settings.llm_mode, language_filter)

    sources = [
        Source(
            filename=r["metadata"].get("filename", ""),
            page_number=int(r["metadata"].get("page_number", 0)),
            document_date=(r["metadata"].get("document_date") or None),
            document_type=r["metadata"].get("document_type"),
            language=r["metadata"].get("language"),
            chunk_text=r["chunk_text"],
            distance=r.get("distance"),
        )
        for r in results
    ]

    return SearchResponse(
        answer=answer,
        answer_mode=settings.llm_mode,
        sources=sources,
        filters_applied=filters,
        chunks_retrieved=len(results),
        warning=warning,
    )


@app.get("/documents", response_model=DocumentListResponse)
def documents() -> DocumentListResponse:
    rows = database.get_all_documents()
    docs = [
        DocumentInfo(
            document_id=r["document_id"],
            filename=r["filename"],
            upload_timestamp=r["upload_timestamp"],
            page_count=r.get("page_count"),
            chunk_count=r.get("chunk_count"),
            languages_detected=r.get("languages_detected", []),
            document_date=r.get("document_date"),
            document_type=r.get("document_type"),
            status=r["status"],
        )
        for r in rows
    ]
    return DocumentListResponse(documents=docs, total=len(docs))
