#!/usr/bin/env python3
"""
Score an image for fraud risk using the pre-built FAISS index.

Usage:
    python predict.py path/to/image.jpg
    python predict.py path/to/image.jpg --top-k 3 --json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from fraud_model.scorer import FraudDetector

# Colour helpers for terminal output
_COLOURS = {
    "CRITICAL": "\033[1;31m",  # bold red
    "HIGH":     "\033[0;31m",  # red
    "MEDIUM":   "\033[0;33m",  # yellow
    "LOW":      "\033[0;32m",  # green
    "MINIMAL":  "\033[0;37m",  # grey
    "RESET":    "\033[0m",
}


def _colour(text: str, level: str) -> str:
    return _COLOURS.get(level, "") + text + _COLOURS["RESET"]


def pretty_print(result) -> None:
    score_pct = f"{result.fraud_score * 100:.1f}%"
    print()
    print("┌─── CSIRT Fraud Risk Analysis ─────────────────────────────────┐")
    print(f"│  Fraud Score : {score_pct:<10}                                   │")
    print(f"│  Risk Level  : {_colour(result.risk_level, result.risk_level):<10}                                   │")
    print("└────────────────────────────────────────────────────────────────┘")

    for match in result.top_matches:
        inc = match.incident
        sim_pct = f"{match.similarity * 100:.1f}%"
        print()
        print(f"  ── Match #{match.rank}  (similarity: {sim_pct}) ────────────────────────")
        if inc.get("title"):
            print(f"     Title      : {inc['title']}")
        if inc.get("alert_code"):
            print(f"     Alert code : {inc['alert_code']}")
        if inc.get("clase_de_alerta"):
            print(f"     Category   : {inc['clase_de_alerta']}")
        if inc.get("tipo_de_incidente"):
            print(f"     Type       : {inc['tipo_de_incidente']}")
        if inc.get("nivel_de_riesgo"):
            print(f"     Risk level : {inc['nivel_de_riesgo']}")
        if inc.get("fecha_lanzamiento"):
            print(f"     Date       : {inc['fecha_lanzamiento']}")
        if inc.get("indicadores"):
            lines = inc["indicadores"].splitlines()
            preview = "\n               ".join(lines[:6])
            print(f"     IoC data   : {preview}")
            if len(lines) > 6:
                print(f"               … (+{len(lines) - 6} more lines)")
        if inc.get("image_path"):
            print(f"     Ref image  : {inc['image_path']}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score an image for fraud risk against the CSIRT dataset."
    )
    parser.add_argument("image", help="Path to the query image")
    parser.add_argument(
        "--top-k", type=int, default=5,
        help="Number of nearest-neighbour matches to return (default: 5)",
    )
    parser.add_argument(
        "--artifacts", default=None,
        help="Path to artifacts directory (default: fraud_model/artifacts/)",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output raw JSON instead of formatted text",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: image not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    artifacts = Path(args.artifacts) if args.artifacts else None

    try:
        detector = FraudDetector(artifacts_dir=artifacts, top_k=args.top_k)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    result = detector.score(image_path)

    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    else:
        pretty_print(result)


if __name__ == "__main__":
    main()
