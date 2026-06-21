"""
Bangla-aware text chunking + per-chunk language detection.

WHY THESE CHOICES (this comment block feeds docs/explain.md Section 2):

* Chunk size ~300-400 tokens with ~50-token overlap.
  This target is about RETRIEVAL GRANULARITY, not a model token ceiling:
  chunks that are too large dilute similarity scores and return imprecise
  context; chunks that are too small lose the surrounding context needed to
  answer a question. BGE-M3 supports sequences up to 8192 tokens, so there is
  NO truncation risk being managed here — purely a precision/context trade-off.
  We approximate "tokens" by whitespace+script word count, which is good enough
  for sizing (we are not enforcing a hard model limit).

* Separators include the Bangla danda "।" (U+0964).
  A recursive splitter that only knows Latin ".?!" runs straight through Bangla
  sentence boundaries, producing chunks that cut mid-sentence and mangle
  context. We split on "।" and "॥" (U+0965) in addition to ".?!".

* Language is detected PER CHUNK, not per document, because a single document
  (e.g. a bilingual form or article) mixes Bangla and English across sections.
  The per-chunk `language` tag is what makes the `language` metadata filter
  meaningful at query time.

* Chunks never merge across page boundaries (page number is preserved for
  citation).
"""
from __future__ import annotations

import re
from datetime import datetime, timezone

# Bangla Unicode block: U+0980 - U+09FF
_BANGLA_RE = re.compile(r"[ঀ-৿]")
# A "letter" for the purpose of the language ratio = Bangla char or Latin a-zA-Z.
_LETTER_RE = re.compile(r"[ঀ-৿ A-Za-z]")
_LATIN_RE = re.compile(r"[A-Za-z]")

# Sentence-boundary separators, longest/strongest first. Includes Bangla danda.
_SENTENCE_SEPARATORS = ["।", "॥", "\n\n", "\n", ". ", "? ", "! ", "; "]

# Rough word-count target per chunk and overlap (see module docstring).
TARGET_WORDS = 320
OVERLAP_WORDS = 50


def date_to_int(value: str | None) -> int | None:
    """
    Convert a ``YYYY-MM-DD`` date into a zero-padded integer ``YYYYMMDD``
    (e.g. ``2025-03-14`` -> ``20250314``).

    ChromaDB 0.5.x requires int/float operands for the range operators
    ``$gte``/``$lte``; a date string is rejected. Storing the date as an integer
    keeps chronological order intact for numeric range filtering. Returns
    ``None`` for a missing/invalid date — callers then omit the key, so undated
    documents are naturally excluded from date-range filters.
    """
    if not value:
        return None
    try:
        return int(datetime.strptime(value.strip(), "%Y-%m-%d").strftime("%Y%m%d"))
    except (ValueError, TypeError):
        return None


def detect_language(text: str) -> str:
    """
    Classify a chunk as 'bn', 'en', or 'mixed' using a Unicode-range heuristic.

    Ratio is computed over *letters only* (Bangla + Latin), ignoring digits,
    punctuation and whitespace so that numerals/symbols don't skew the result.
    """
    bangla = len(_BANGLA_RE.findall(text))
    latin = len(_LATIN_RE.findall(text))
    total = bangla + latin
    if total == 0:
        return "en"  # no script letters (e.g. pure numbers) -> default to en
    bn_ratio = bangla / total
    if bn_ratio >= 0.85:
        return "bn"
    if bn_ratio <= 0.15:
        return "en"
    return "mixed"


def _split_into_sentences(text: str) -> list[str]:
    """Recursively split text on the strongest available separator."""
    units = [text]
    for sep in _SENTENCE_SEPARATORS:
        nxt: list[str] = []
        for unit in units:
            if sep in unit:
                parts = unit.split(sep)
                # Re-attach the separator (except the trailing empty split).
                for i, p in enumerate(parts):
                    piece = p + (sep if i < len(parts) - 1 else "")
                    if piece.strip():
                        nxt.append(piece)
            elif unit.strip():
                nxt.append(unit)
        units = nxt
    return units


def _word_count(text: str) -> int:
    return len(text.split())


def _pack_sentences(sentences: list[str]) -> list[str]:
    """
    Greedily pack sentences into ~TARGET_WORDS chunks with ~OVERLAP_WORDS overlap
    carried from the tail of the previous chunk.
    """
    chunks: list[str] = []
    current: list[str] = []
    current_words = 0

    for sent in sentences:
        sw = _word_count(sent)
        if current and current_words + sw > TARGET_WORDS:
            chunks.append(" ".join(current).strip())
            # Build overlap from the tail words of the chunk just emitted.
            tail_words = " ".join(current).split()[-OVERLAP_WORDS:]
            current = [" ".join(tail_words)] if tail_words else []
            current_words = _word_count(" ".join(current))
        current.append(sent.strip())
        current_words += sw

    if current and "".join(current).strip():
        chunks.append(" ".join(current).strip())
    return [c for c in chunks if c.strip()]


def chunk_page(
    page_dict: dict,
    document_id: str,
    filename: str,
    document_date: str | None,
    document_type: str,
    upload_timestamp: str | None = None,
) -> list[dict]:
    """
    Split one OCR'd page into retrieval chunks with full metadata.

    Returns a list of:
        {
          "id": "<document_id>_p<page_num>_c<chunk_idx>",
          "document": "<chunk text>",
          "metadata": { document_id, filename, page_number, chunk_index,
                        language, document_date, document_type, upload_timestamp }
        }
    """
    if upload_timestamp is None:
        upload_timestamp = datetime.now(timezone.utc).isoformat()

    page_num = page_dict["page_num"]
    text = (page_dict.get("text") or "").strip()
    if not text:
        return []

    sentences = _split_into_sentences(text)
    chunk_texts = _pack_sentences(sentences)

    date_int = date_to_int(document_date)  # numeric form for range filtering

    chunks: list[dict] = []
    for idx, chunk_text in enumerate(chunk_texts):
        metadata = {
            "document_id": document_id,
            "filename": filename,
            "page_number": int(page_num),
            "chunk_index": idx,
            "language": detect_language(chunk_text),
            # Chroma metadata cannot hold None -> use "" for absent date.
            "document_date": document_date or "",
            "document_type": document_type or "other",
            "upload_timestamp": upload_timestamp,
        }
        # Only present when the date is valid; absent => excluded from date filters.
        if date_int is not None:
            metadata["document_date_int"] = date_int
        chunks.append(
            {
                "id": f"{document_id}_p{page_num}_c{idx}",
                "document": chunk_text,
                "metadata": metadata,
            }
        )
    return chunks


def chunk_document(
    pages: list[dict],
    document_id: str,
    filename: str,
    document_date: str | None,
    document_type: str,
) -> list[dict]:
    """Chunk every page of a document, never merging across page boundaries."""
    upload_timestamp = datetime.now(timezone.utc).isoformat()
    all_chunks: list[dict] = []
    for page in pages:
        all_chunks.extend(
            chunk_page(page, document_id, filename, document_date, document_type, upload_timestamp)
        )
    return all_chunks
