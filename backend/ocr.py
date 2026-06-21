"""
Local OCR extraction with Surya OCR.

Surya is used for its stronger Bangla/conjunct recognition and layout handling
on dense, multi-column scans.

Hard rule: NO function in this module ever performs an external/network call.
All recognition runs on-machine. PDFs are rasterised with pypdfium2 (no system
Poppler dependency).

Public API:
    extract_text_from_file(file_path) -> list[dict]
        [{ "page_num": 0, "text": "...", "char_count": int }, ...]
"""
from __future__ import annotations

import os
import time
from typing import Any

from PIL import Image

from .config import settings

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}
PDF_EXTS = {".pdf"}

# Lazily-initialised heavy objects (models). Loaded once, reused across calls.
_surya_models: dict[str, Any] | None = None


# --------------------------------------------------------------------------- #
#  PDF -> PIL images (local, no Poppler)
# --------------------------------------------------------------------------- #
def _pdf_to_images(file_path: str, dpi: int = 200) -> list[Image.Image]:
    """Rasterise each PDF page to a PIL image using pypdfium2 (pure-python/C, local)."""
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(file_path)
    images: list[Image.Image] = []
    scale = dpi / 72.0  # PDF points are 1/72 inch
    try:
        for i in range(len(pdf)):
            page = pdf[i]
            bitmap = page.render(scale=scale)
            images.append(bitmap.to_pil().convert("RGB"))
    finally:
        pdf.close()
    return images


def _load_images(file_path: str) -> list[Image.Image]:
    ext = os.path.splitext(file_path)[1].lower()
    if ext in PDF_EXTS:
        return _pdf_to_images(file_path)
    if ext in IMAGE_EXTS:
        return [Image.open(file_path).convert("RGB")]
    raise ValueError(f"Unsupported file type '{ext}'. Supported: PDF and {sorted(IMAGE_EXTS)}.")


# --------------------------------------------------------------------------- #
#  Surya engine (0.6.x function-based API)
# --------------------------------------------------------------------------- #
def _get_surya_models() -> dict[str, Any]:
    """
    Load Surya detection + recognition models once (downloaded on first run).

    Surya 0.6.x uses a function-based API:
      surya.model.detection.model.load_model / load_processor
      surya.model.recognition.model.load_model
      surya.model.recognition.processor.load_processor
      surya.ocr.run_ocr(images, langs, det_model, det_processor, rec_model, rec_processor)
    """
    global _surya_models
    if _surya_models is None:
        from surya.model.detection import model as det_module
        from surya.model.recognition import model as rec_module
        from surya.model.recognition import processor as proc_module

        print("[ocr] Loading Surya OCR models (first run downloads weights locally)...")
        det_model = det_module.load_model()
        det_processor = det_module.load_processor()
        rec_model = rec_module.load_model()
        rec_processor = proc_module.load_processor()
        _surya_models = {
            "det_model": det_model,
            "det_processor": det_processor,
            "rec_model": rec_model,
            "rec_processor": rec_processor,
        }
        print("[ocr] Surya models ready.")
    return _surya_models


def _surya_page_text(result: Any) -> str:
    """
    Flatten a Surya OCRResult for one page into human reading-order text.

    Surya returns detected text lines in model/detection order, which scrambles
    multi-column and TABULAR layouts: an invoice row's cells (item, qty, unit
    price, total) come out interleaved and detached from their column headers.
    For the bilingual invoice that produced output like
    `চাল (মিনিকেট) / Rice  →  ২৫ কেজি  →  ০১  →  ২,০০০  →  ৮০` in jumbled order.

    We reconstruct reading order from each line's bbox: group lines into rows by
    vertical overlap, then sort each row left-to-right. This keeps a table row's
    cells adjacent, so when the (short) invoice lands in a single chunk the
    item/qty/price/total association is preserved — materially better context
    for table Q&A. For ordinary single-column prose every line falls in its own
    row, so output is identical to the naive flatten (no regression).
    """
    items: list[tuple[list[float] | None, str]] = []
    for line in getattr(result, "text_lines", []):
        text = (getattr(line, "text", "") or "").strip()
        if not text:
            continue
        bbox = getattr(line, "bbox", None)  # [x1, y1, x2, y2]
        items.append((list(bbox) if bbox and len(bbox) >= 4 else None, text))

    if not items:
        return ""
    # If any line lacks a usable bbox, fall back to detection order (can't sort).
    if any(b is None for b, _ in items):
        return "\n".join(t for _, t in items)

    # Row-grouping tolerance from the median line height.
    heights = sorted((b[3] - b[1]) for b, _ in items)
    med_h = heights[len(heights) // 2] or 1.0
    tol = med_h * 0.6

    items.sort(key=lambda it: (it[0][1] + it[0][3]) / 2.0)  # by vertical centre
    rows: list[tuple[float, list[tuple[float, str]]]] = []
    for bbox, text in items:
        y_c = (bbox[1] + bbox[3]) / 2.0
        if rows and abs(y_c - rows[-1][0]) <= tol:
            rows[-1][1].append((bbox[0], text))  # same visual row
        else:
            rows.append((y_c, [(bbox[0], text)]))  # new row

    out: list[str] = []
    for _, cells in rows:
        cells.sort(key=lambda c: c[0])  # left-to-right within the row
        out.append("  ".join(t for _, t in cells))
    return "\n".join(out)


def _extract_surya(images: list[Image.Image]) -> list[dict]:
    from surya.ocr import run_ocr

    models = _get_surya_models()
    langs = settings.ocr_langs  # e.g. ["bn", "en"]
    # run_ocr expects langs as a list-per-image of lists
    langs_per_image = [langs] * len(images)

    pages: list[dict] = []
    total = len(images)
    for idx, img in enumerate(images):
        print(f"[ocr] Surya processing page {idx + 1}/{total}...")
        results = run_ocr(
            [img],
            [langs_per_image[idx]],
            models["det_model"],
            models["det_processor"],
            models["rec_model"],
            models["rec_processor"],
        )
        page_text = _surya_page_text(results[0]) if results else ""
        pages.append({"page_num": idx, "text": page_text, "char_count": len(page_text)})
    return pages


# --------------------------------------------------------------------------- #
#  Public entry point
# --------------------------------------------------------------------------- #
def extract_text_from_file(file_path: str) -> list[dict]:
    """
    OCR a PDF or image file fully locally with Surya.

    Returns one dict per page:
        { "page_num": int (0-based), "text": str, "char_count": int }
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)

    print(f"[ocr] Engine=surya  file={os.path.basename(file_path)}")
    images = _load_images(file_path)
    print(f"[ocr] {len(images)} page(s) to process.")

    started = time.time()
    pages = _extract_surya(images)

    elapsed = time.time() - started
    per_page = elapsed / max(len(images), 1)
    print(f"[ocr] Done: {len(images)} page(s) in {elapsed:.1f}s ({per_page:.1f}s/page).")
    return pages
