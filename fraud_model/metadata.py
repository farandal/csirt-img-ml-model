"""Parser for CSIRT incident .txt files produced by extract_fraud_images.py."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class IncidentMetadata:
    source_pdf: str
    page: int
    image_name: str
    image_path: str          # absolute path to the paired image file
    txt_path: str            # absolute path to this txt file
    title: str
    alert_code: str
    clase_de_alerta: str
    tipo_de_incidente: str
    nivel_de_riesgo: str
    tlp: str
    fecha_lanzamiento: str
    ultima_revision: str
    indicadores: str         # raw IoC block (URLs, IPs, hashes, …)

    def to_dict(self) -> dict:
        return asdict(self)


# Image extensions to try when resolving the paired image file
_IMAGE_EXTS = (".jpeg", ".jpg", ".png", ".webp", ".gif")


def _find_image(txt_path: Path) -> str:
    for ext in _IMAGE_EXTS:
        candidate = txt_path.with_suffix(ext)
        if candidate.exists():
            return str(candidate)
    return ""


def _extract(body: str, label: str) -> str:
    """
    Pull the value that follows a label line, stopping at the next blank line
    or end-of-string.
    """
    pattern = rf"^{re.escape(label)}\s*\n(.*?)(?:\n\n|\Z)"
    m = re.search(pattern, body, re.MULTILINE | re.DOTALL)
    return m.group(1).strip() if m else ""


def parse_txt(txt_path: str | Path) -> IncidentMetadata:
    """Parse a CSIRT incident txt file into structured metadata."""
    txt_path = Path(txt_path)
    raw = txt_path.read_text(encoding="utf-8", errors="replace")

    # ── Header line ──────────────────────────────────────────────────────────
    header_m = re.match(
        r"Source:\s*(.+?)\s*\|\s*Page:\s*(\d+)\s*\|\s*Image:\s*(.+?)(?:\s*\(.+?\))?\s*\n",
        raw,
    )
    source_pdf = header_m.group(1).strip() if header_m else ""
    page = int(header_m.group(2)) if header_m else 0
    image_name = header_m.group(3).strip() if header_m else txt_path.stem

    # ── Body (everything after the dashed separator) ──────────────────────────
    # The separator is 60 dashes; split on any run of 20+ dashes to be robust.
    body = re.split(r"-{20,}", raw, maxsplit=1)[-1].strip()

    # Incident title = first non-empty, non-dash, non-table-header line
    _SKIP_TITLE = {"Alerta de seguridad", "Clase de alerta", "Tipo de incidente"}
    title = ""
    for line in body.splitlines():
        line = line.strip()
        if not line or set(line) == {"-"}:
            continue
        if any(kw in line for kw in _SKIP_TITLE):
            break
        title = line
        break

    # Alert code lives on the line right after "Alerta de seguridad …"
    alert_code = ""
    alert_m = re.search(r"Alerta de seguridad[^\n]*\n(.+?)(?:\n|\Z)", body)
    if alert_m:
        alert_code = alert_m.group(1).strip()

    clase = _extract(body, "Clase de alerta")
    tipo = _extract(body, "Tipo de incidente")
    nivel = _extract(body, "Nivel de riesgo")
    tlp = _extract(body, "TLP")
    fecha = _extract(body, "Fecha de lanzamiento original")
    revision = _extract(body, "Última revisión")

    # IoC block — everything from "Indicadores de compromiso" to end
    ioc_m = re.search(r"Indicadores de compromiso\s*\n(.+)", body, re.DOTALL)
    indicadores = ioc_m.group(1).strip() if ioc_m else ""

    return IncidentMetadata(
        source_pdf=source_pdf,
        page=page,
        image_name=image_name,
        image_path=_find_image(txt_path),
        txt_path=str(txt_path),
        title=title,
        alert_code=alert_code,
        clase_de_alerta=clase,
        tipo_de_incidente=tipo,
        nivel_de_riesgo=nivel,
        tlp=tlp,
        fecha_lanzamiento=fecha,
        ultima_revision=revision,
        indicadores=indicadores,
    )
