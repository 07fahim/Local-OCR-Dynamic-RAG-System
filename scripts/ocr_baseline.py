"""
OCR baseline harness — run Surya on one file and print the raw extracted text +
timing, so you can hand-check Bangla accuracy for docs/explain.md Section 1.

Usage:
    python -m scripts.ocr_baseline "data/sample_docs/bilingual_sample.pdf"
"""
from __future__ import annotations

import sys
import time

from backend.config import settings
from backend.ocr import extract_text_from_file


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: python -m scripts.ocr_baseline <file>")
        raise SystemExit(1)
    path = sys.argv[1]
    print("Engine : surya")
    print(f"Langs  : {settings.ocr_langs}")
    print(f"File   : {path}\n" + "=" * 60)

    t0 = time.time()
    pages = extract_text_from_file(path)
    elapsed = time.time() - t0

    total_chars = 0
    for p in pages:
        total_chars += p["char_count"]
        print(f"\n----- page {p['page_num']} ({p['char_count']} chars) -----")
        print(p["text"])

    print("\n" + "=" * 60)
    ppm = len(pages) / (elapsed / 60) if elapsed else 0
    print(f"Pages: {len(pages)}  chars: {total_chars}  time: {elapsed:.1f}s  "
          f"throughput: {ppm:.1f} pages/min")


if __name__ == "__main__":
    main()
