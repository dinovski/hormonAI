#!/usr/bin/env python3
"""
pdf_fidelity.py — extraction quality check for the PDF sources in a manifest.

PDF text extraction is the weakest link in the ingestion pipeline: multi-column
reflow, running headers/footers, hyphenated line-wraps, and (worst case) scanned
image-only pages that yield no text at all. For a patient-facing medical tool a
garbled number or a dropped "not" is a safety issue, so this script lets the
extraction be eyeballed BEFORE committing to an ingest.

For every PDF article in the manifest it reports: backend used, pages, raw vs
cleaned character counts, chars/page, resulting chunk count, and a flag
(scanned / thin / ok). With --preview it also prints the head and tail of the
cleaned text and the first chunk, so wording, numbers, and negations can be
spot-checked.

Exits non-zero if any PDF looks scanned/empty, so it can gate an ingest in CI.

Usage:
    python tests/pdf_fidelity.py                     # uses manifest.json
    python tests/pdf_fidelity.py --manifest m.json --preview
    python tests/pdf_fidelity.py --preview --only NCI-HT.pdf
"""
from __future__ import annotations

import os
import sys
import json
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import ingest as ig  # noqa: E402

SCANNED_CPP = 200   # chars/page below this => likely scanned / no text layer
THIN_CPP = 500      # chars/page below this => sparse, worth a look


def main() -> int:
    ap = argparse.ArgumentParser(description="Check PDF extraction fidelity for a manifest.")
    ap.add_argument("--manifest", default="manifest.json")
    ap.add_argument("--preview", action="store_true", help="Print cleaned head/tail + first chunk.")
    ap.add_argument("--only", default=None, help="Only check files whose path contains this substring.")
    ap.add_argument("--chunk-size", type=int, default=250)
    ap.add_argument("--chunk-overlap", type=int, default=40)
    args = ap.parse_args()

    with open(args.manifest, "r", encoding="utf-8") as f:
        entries = json.load(f)

    pdfs = [e for e in entries
            if e.get("type") == "article"
            and os.path.splitext(e["path"])[1].lower() == ".pdf"
            and (args.only is None or args.only in e["path"])]

    if not pdfs:
        print("No PDF articles found in manifest.")
        return 0

    print(f"{'file':46s} {'lang':>4s} {'bkd':>7s} {'pg':>3s} {'raw':>7s} {'clean':>7s} {'c/pg':>5s} {'chunks':>6s}  flag")
    print("-" * 100)

    scanned, thin = [], []
    for e in pdfs:
        path, lang = e["path"], e.get("lang", "?")
        name = os.path.basename(path)
        if not os.path.exists(path):
            print(f"{name:46s} {lang:>4s}  MISSING FILE")
            scanned.append(name)
            continue

        pages, backend = ig.extract_pdf_pages(path)
        raw_chars = sum(len(p or "") for p in pages)
        cleaned = ig.clean_pdf_pages(pages)
        clean_chars = len(cleaned)
        npg = max(1, len(pages))
        cpp = clean_chars // npg
        items = ig.parse_article(path, lang, args.chunk_size, args.chunk_overlap)

        if cpp < SCANNED_CPP:
            flag = "SCANNED? -> OCR needed"
            scanned.append(name)
        elif cpp < THIN_CPP:
            flag = "thin -> review"
            thin.append(name)
        else:
            flag = "ok"

        print(f"{name:46s} {lang:>4s} {backend:>7s} {len(pages):>3d} "
              f"{raw_chars:>7d} {clean_chars:>7d} {cpp:>5d} {len(items):>6d}  {flag}")

        if args.preview:
            head = cleaned[:400].replace("\n", " ")
            tail = cleaned[-300:].replace("\n", " ")
            print(f"    head: {head}")
            print(f"    tail: {tail}")
            if items:
                print(f"    chunk[0] ({ig.word_count(items[0]['answer'])} words): "
                      f"{items[0]['answer'][:400].replace(chr(10),' ')}")
            print()

    print("-" * 100)
    print(f"Total PDFs: {len(pdfs)} | ok: {len(pdfs)-len(scanned)-len(thin)} | "
          f"thin: {len(thin)} | scanned/empty: {len(scanned)}")
    if thin:
        print("  review (thin):", ", ".join(thin))
    if scanned:
        print("  SCANNED/EMPTY (need OCR or a better source):", ", ".join(scanned))

    return 1 if scanned else 0


if __name__ == "__main__":
    raise SystemExit(main())
