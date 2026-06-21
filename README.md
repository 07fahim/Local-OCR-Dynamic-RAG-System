# Local OCR & Dynamic RAG System

A secure, locally-run document pipeline + hybrid RAG search for **Bangla,
English, or mixed** scanned documents and PDFs.

- **OCR (local):** Surya OCR (`bn,en`)
- **Embeddings (local):** BAAI/BGE-M3 multilingual dense vectors
- **Vector store:** ChromaDB (persistent, native metadata filtering)
- **Hybrid search:** manual metadata filters (`document_type`, `language`, date
  range) applied as a hard pre-filter, then semantic similarity ranking
- **Answer generation:** fully local via Ollama (or chunk-only mode, no LLM)
- **Backend:** FastAPI + a minimal HTML UI

## Locality statement (read this)

This system is **fully local — no external or commercial API is ever called.**
Every stage (OCR, embedding, retrieval, and answer generation) runs on-machine.
Answer generation is controlled by `LLM_MODE`:

| `LLM_MODE` | Answer behaviour | Network |
|---|---|---|
| `local` (default, recommended) | Synthesises a natural-language answer via a local Ollama model | **None** (localhost only) |
| `none` | Returns the retrieved, cited chunks verbatim (no LLM) | **None** |

Both modes satisfy the brief's "fully localized … without sending data to
external commercial APIs" requirement with zero external calls. See
`docs/explain.md` Section 3.

---

## Prerequisites

- **Python 3.10+**
- For **Surya OCR**: no system package needed (PDFs render via `pypdfium2`);
  first run downloads ~2–3 GB of model weights. A GPU is optional but much
  faster than CPU.
- For **fully-local answers** (`LLM_MODE=local`, recommended): install
  **[Ollama](https://ollama.com/download)** and pull a model:
  ```bash
  ollama pull qwen2.5:7b      # ~4.7 GB, recommended; qwen2.5:3b (~2 GB) is a lighter fallback
  ```
  Ollama runs a local server at `http://localhost:11434` automatically.
- Optional: **Docker** + Compose (an alternative to the manual run; the compose
  file includes an Ollama service so the whole stack is self-contained).

### Verify answer generation without the heavy OCR/embedding install

```bash
python -m scripts.llm_check        # uses LLM_MODE from .env (local / none)
```

---

## Setup

```bash
# 1. clone, then from the project root:
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate

pip install -r requirements.txt        # Surya pulls in torch (CPU build is fine)

# 2. configure
cp .env.example .env                    # defaults to LLM_MODE=local (Ollama)
#    - LLM_MODE=local : local Ollama answers (run `ollama pull qwen2.5:7b` first)
#    - LLM_MODE=none  : no LLM, returns cited chunks (zero extra setup)
```

> `.env` is **gitignored** — never commit local config.

### Run (manual — recommended)

```bash
uvicorn backend.main:app --reload --port 8000
# open http://localhost:8000
```

### Run (Docker — optional, includes Ollama)

```bash
cd docker
docker compose up --build
docker compose exec ollama ollama pull qwen2.5:7b   # one-time model pull
# open http://localhost:8000
```

### Run without Ollama

Set `LLM_MODE=none` in `.env`. Search returns cited chunks instead
of a synthesised answer.

---

## Sample documents

Sample Bangla/English files are in `data/sample_docs/` (invoice, newspaper,
article, agreement, etc.). Drop your own scans there too.

---

## API reference

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | Minimal HTML upload/search UI |
| `GET` | `/health` | Status + active config |
| `POST` | `/upload` | multipart `file` + optional `document_type`, `document_date` → OCR→chunk→embed→index |
| `POST` | `/search` | `{query, filters{document_type,language,date_from,date_to}, top_k}` → hybrid retrieval + answer |
| `GET` | `/documents` | List ingested documents from the SQLite log |

Example search:

```bash
curl -s http://localhost:8000/search -H "Content-Type: application/json" -d '{
  "query": "আমের দাম কত?",
  "filters": {"language": "bn", "document_type": "invoice"},
  "top_k": 5
}'
```

---

## Measure the OCR / embedding baselines

```bash
python -m scripts.ocr_baseline "data/sample_docs/bangla newspaper.jpg"
python -m scripts.embed_check
```

These commands produced the measured numbers already recorded in
`docs/explain.md` (OCR accuracy/throughput and cross-lingual cosine).

---

## Project layout

```
backend/        FastAPI app, OCR, chunking, embeddings, vector store, RAG
backend/templates/upload_search.html   minimal UI
scripts/        ocr_baseline.py, embed_check.py (measurement harnesses)
docker/         Dockerfile, docker-compose.yml (optional)
data/sample_docs/   sample Bangla/English documents
docs/           explain.md (Must-Explain answers), schema.md
.env.example    config template (copy to .env)
requirements.txt
```

See `docs/explain.md` for the OCR / chunking / hybrid-search rationale and
`docs/schema.md` for the data model.

---

## Known limitations

- Surya on CPU is slow (seconds/page); a GPU is significantly faster.
- OCR quality on degraded/low-resolution scans is bounded by the source image.
- BGE-M3 first load is ~2.2 GB of weights and uses a few GB of RAM while resident.
- `LLM_MODE=local` needs Ollama running + the model pulled; otherwise use
  `LLM_MODE=none` (returns cited chunks, no LLM).
