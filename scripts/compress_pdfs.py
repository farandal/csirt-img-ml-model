#!/usr/bin/env python3
"""Compress PDFs in csirt_pdf/ using pypdf."""

import os
from pathlib import Path
from pypdf import PdfReader, PdfWriter

input_dir = Path("csirt_pdf")
output_dir = Path("csirt_pdf_compressed")
output_dir.mkdir(exist_ok=True)

files = sorted(input_dir.glob("*.pdf"))
total = len(files)
total_before = 0
total_after = 0

for i, pdf_path in enumerate(files, 1):
    out_path = output_dir / pdf_path.name
    size_before = pdf_path.stat().st_size

    try:
        reader = PdfReader(str(pdf_path))
        writer = PdfWriter()

        for page in reader.pages:
            writer.add_page(page)

        for page in writer.pages:
            page.compress_content_streams()

        # Copy metadata
        if reader.metadata:
            writer.add_metadata(reader.metadata)

        with open(out_path, "wb") as f:
            writer.write(f)

        size_after = out_path.stat().st_size
        reduction = (1 - size_after / size_before) * 100
        total_before += size_before
        total_after += size_after
        print(f"[{i}/{total}] {pdf_path.name}: {size_before//1024}KB -> {size_after//1024}KB ({reduction:.1f}% reduction)")

    except Exception as e:
        print(f"[{i}/{total}] SKIP {pdf_path.name}: {e}")
        # Copy original if compression fails
        import shutil
        shutil.copy2(pdf_path, out_path)
        total_before += size_before
        total_after += size_before

overall = (1 - total_after / total_before) * 100 if total_before else 0
print(f"\nDone: {total_before//1024//1024}MB -> {total_after//1024//1024}MB (overall {overall:.1f}% reduction)")
print(f"Compressed files saved to: {output_dir}/")
