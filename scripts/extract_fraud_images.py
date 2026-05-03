#!/usr/bin/env python3
"""
Extract fraudulent email/site screenshots from CSIRT PDFs.
For each image, creates a matching .txt file with the incident table data
from the right column of the same page. Skips header banners and CSIRT logos.
"""

import shutil
from pathlib import Path
import fitz  # PyMuPDF

INPUT_DIR = Path("csirt_pdf")
OUTPUT_DIR = Path("csirt_extracted")

SKIP_TEXT_PATTERNS = [
    "Ministerio del Interior",
    "Página ",
    "https://www.csirt.gob.cl",
    "Teléfonos: 1510",
    "@CSIRTGOB",
    "@csirtgob",
    "https://www.linkedin.com",
    "CONTACTO Y REDES SOCIALES",
    "Equipo de Respuesta ante Incidentes",
    "Coordinación Nacional de Ciberseguridad",
    "Gobierno de Chile",
    "+ (562)",
]


def is_header(rect, page_width: float) -> bool:
    """Full-width banner spanning the top of the page."""
    return rect.y0 < 30 and rect.width > page_width * 0.45


def is_logo(rect, page_width: float) -> bool:
    """CSIRT logo in the top-right corner (appears in different positions across layouts)."""
    return rect.x0 > page_width * 0.6 and rect.y0 < 220


def is_tiny(rect) -> bool:
    """Invisible filler image."""
    return rect.width < 40 or rect.height < 30


def find_incident_sections(page):
    """
    Return a list of (y_start, text) for each incident table found
    in the right column of the page (x0 > 200).
    """
    blocks = page.get_text("blocks")

    # Collect right-column text blocks (table is always in the right half)
    right_blocks = []
    for b in blocks:
        x0, y0, x1, y1, text, _, block_type = b
        if block_type != 0 or x0 < 200 or not text.strip():
            continue
        if any(p in text for p in SKIP_TEXT_PATTERNS):
            continue
        right_blocks.append((y0, y1, text))

    right_blocks.sort(key=lambda b: b[0])

    # Detect incident boundaries using the primary marker first.
    # "Alerta de seguridad" appears in the title block, grouping the full table under it.
    # Fall back to "Clase de alerta" only for older PDFs that lack the title block.
    boundary_ys = [y0 for y0, y1, text in right_blocks if "Alerta de seguridad" in text]
    if not boundary_ys:
        boundary_ys = [y0 for y0, y1, text in right_blocks if "Clase de alerta" in text]

    if not boundary_ys:
        return []

    # Build one section per detected incident
    sections = []
    for i, start_y in enumerate(boundary_ys):
        end_y = boundary_ys[i + 1] if i + 1 < len(boundary_ys) else float("inf")
        section_text = "\n\n".join(
            t.strip()
            for y0, y1, t in right_blocks
            if start_y - 5 <= y0 < end_y
        )
        if section_text.strip():
            sections.append((start_y, section_text))

    return sections


def match_section(image_rect, sections):
    """Return the incident text whose y_start is closest to the image's top."""
    if not sections:
        return ""
    img_y = image_rect.y0
    best = min(sections, key=lambda s: abs(s[0] - img_y))
    return best[1]


def process_pdf(pdf_path: Path, pdf_output_dir: Path) -> int:
    try:
        doc = fitz.open(str(pdf_path))
    except Exception as e:
        print(f"  ERROR opening {pdf_path.name}: {e}")
        return 0

    image_count = 0

    for page_num in range(len(doc)):
        page = doc[page_num]
        page_width = page.rect.width

        img_list = page.get_images(full=True)

        # Collect content images for this page (deduplicated by xref)
        seen_xrefs = set()
        content_images = []
        for img in img_list:
            xref = img[0]
            if xref in seen_xrefs:
                continue
            rects = page.get_image_rects(xref)
            if not rects:
                continue
            rect = rects[0]
            if is_header(rect, page_width) or is_logo(rect, page_width) or is_tiny(rect):
                continue
            seen_xrefs.add(xref)
            content_images.append((rect.y0, xref, rect))

        if not content_images:
            continue

        # Build incident sections from the right column first.
        # Skip pages with no incident table (e.g. summary/index pages).
        sections = find_incident_sections(page)
        if not sections:
            continue

        # Sort images top-to-bottom so numbering is consistent
        content_images.sort(key=lambda x: x[0])

        for idx, (_, xref, rect) in enumerate(content_images, 1):
            try:
                base_image = doc.extract_image(xref)
            except Exception:
                continue

            ext = base_image.get("ext", "png")
            img_data = base_image["image"]
            w, h = base_image.get("width", 0), base_image.get("height", 0)

            base_name = f"{pdf_path.stem}_p{page_num + 1:02d}_{idx:02d}"
            img_path = pdf_output_dir / f"{base_name}.{ext}"
            txt_path = pdf_output_dir / f"{base_name}.txt"

            img_path.write_bytes(img_data)

            incident_text = match_section(rect, sections)
            header_line = (
                f"Source: {pdf_path.name} | Page: {page_num + 1} | "
                f"Image: {base_name} ({w}x{h}px)\n"
                + "-" * 60 + "\n"
            )
            txt_path.write_text(header_line + incident_text, encoding="utf-8")
            image_count += 1

    doc.close()
    return image_count


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    pdf_files = sorted(INPUT_DIR.glob("*.pdf"))
    total_pdfs = len(pdf_files)
    total_images = 0

    print(f"Processing {total_pdfs} PDFs from {INPUT_DIR}/\n")

    for i, pdf_path in enumerate(pdf_files, 1):
        pdf_output_dir = OUTPUT_DIR / pdf_path.stem
        pdf_output_dir.mkdir(exist_ok=True)

        count = process_pdf(pdf_path, pdf_output_dir)
        if count:
            total_images += count
            print(f"[{i}/{total_pdfs}] {pdf_path.name}: {count} image(s)")
        else:
            shutil.rmtree(pdf_output_dir, ignore_errors=True)
            print(f"[{i}/{total_pdfs}] {pdf_path.name}: no fraud images")

    print(f"\nDone. {total_images} images extracted to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
