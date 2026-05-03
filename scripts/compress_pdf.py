#!/usr/bin/env python3
"""
Remove header and logo images from csirt_bulletins_merged.pdf to reduce file size.

Usage:
    python compress_pdf.py [--input csirt_bulletins_merged.pdf] [--output csirt_bulletins_compressed.pdf]
"""

import argparse
from pathlib import Path

import fitz
from tqdm import tqdm


def is_header(rect, page_width: float) -> bool:
    return rect.y0 < 30 and rect.width > page_width * 0.45


def is_logo(rect, page_width: float) -> bool:
    return rect.x0 > page_width * 0.6 and rect.y0 < 220


def compress(input_path: Path, output_path: Path) -> None:
    doc = fitz.open(str(input_path))
    print(f"Opened: {input_path} — {doc.page_count} pages")

    total_redacted = 0

    for page_num in tqdm(range(doc.page_count), desc="Removing headers/logos"):
        page = doc[page_num]
        page_width = page.rect.width
        img_list = page.get_images(full=True)
        page_redacted = 0

        for img in img_list:
            xref = img[0]
            try:
                rects = page.get_image_rects(xref)
            except Exception:
                continue
            for rect in rects:
                if is_header(rect, page_width) or is_logo(rect, page_width):
                    page.add_redact_annot(rect, fill=(1, 1, 1))
                    page_redacted += 1

        if page_redacted:
            # PDF_REDACT_IMAGE_REMOVE = 1: fully removes image data under redaction
            page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_REMOVE)
            total_redacted += page_redacted

    print(f"\nRedacted {total_redacted} header/logo images across {doc.page_count} pages.")
    print("Saving compressed PDF (this may take a minute)...")

    doc.save(
        str(output_path),
        garbage=4,        # remove unused objects and deduplicate streams
        deflate=True,     # compress all streams
        deflate_images=True,
        deflate_fonts=True,
        clean=True,
    )
    doc.close()

    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"Saved → {output_path} ({size_mb:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="csirt_bulletins_merged.pdf")
    parser.add_argument("--output", default="csirt_bulletins_compressed.pdf")
    args = parser.parse_args()

    repo_root = Path(__file__).parent
    compress(repo_root / args.input, repo_root / args.output)


if __name__ == "__main__":
    main()
