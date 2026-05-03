"""
FraudDetector — main inference class.

Loads the pre-built FAISS index + metadata and scores a query image
against all known fraud images in the dataset.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

# torch must be imported before faiss on macOS to avoid BLAS/AVX conflict (SIGSEGV)
import torch  # noqa: F401  (side-effect import — keeps BLAS context)
import faiss
import numpy as np
from PIL import Image

from .embedder import FraudImageEmbedder
from .metadata import IncidentMetadata

# ── Risk thresholds (cosine similarity, CLIP ViT-B/32) ──────────────────────
# Calibrated against CLIP's typical similarity distribution:
#   identical images  → ~0.99
#   same campaign     → ~0.80-0.95
#   related fraud     → ~0.60-0.79
#   different content → <0.55
_THRESHOLDS = [
    (0.80, "CRITICAL"),
    (0.65, "HIGH"),
    (0.50, "MEDIUM"),
    (0.35, "LOW"),
]


def _risk_level(score: float) -> str:
    for threshold, label in _THRESHOLDS:
        if score >= threshold:
            return label
    return "MINIMAL"


@dataclass
class FraudMatch:
    """A single nearest-neighbour match from the fraud index."""
    rank: int
    similarity: float          # cosine similarity 0-1
    image_path: str
    incident: dict             # parsed IncidentMetadata as dict


@dataclass
class FraudResult:
    """Full result returned by FraudDetector.score()."""
    fraud_score: float         # mean similarity of top-k matches, 0-1
    risk_level: str            # MINIMAL / LOW / MEDIUM / HIGH / CRITICAL
    top_matches: list[FraudMatch]

    def to_dict(self) -> dict:
        return {
            "fraud_score": round(self.fraud_score, 4),
            "risk_level": self.risk_level,
            "top_matches": [
                {
                    "rank": m.rank,
                    "similarity": round(m.similarity, 4),
                    "image_path": m.image_path,
                    "incident": m.incident,
                }
                for m in self.top_matches
            ],
        }


class FraudDetector:
    """
    Similarity-based fraud scorer.

    Usage
    -----
    detector = FraudDetector()                   # loads from default artifacts/
    result   = detector.score("query.jpg")       # path, or PIL.Image
    print(result.fraud_score, result.risk_level)
    for m in result.top_matches:
        print(m.similarity, m.incident["title"])
    """

    DEFAULT_ARTIFACTS = Path(__file__).parent / "artifacts"

    def __init__(
        self,
        artifacts_dir: str | Path | None = None,
        top_k: int = 5,
        device: str | None = None,
    ) -> None:
        self.top_k = top_k
        artifacts = Path(artifacts_dir) if artifacts_dir else self.DEFAULT_ARTIFACTS

        index_path = artifacts / "index.faiss"
        meta_path = artifacts / "metadata.json"

        if not index_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"Artifacts not found in {artifacts}. Run train.py first."
            )

        self._index: faiss.IndexFlatIP = faiss.read_index(str(index_path))
        with open(meta_path, encoding="utf-8") as f:
            self._metadata: list[dict] = json.load(f)

        self._embedder = FraudImageEmbedder(device=device)

    def score(
        self,
        image: str | Path | Image.Image,
        top_k: int | None = None,
    ) -> FraudResult:
        """
        Score a query image against the fraud index.

        Parameters
        ----------
        image   : file path, or a PIL.Image already loaded
        top_k   : override the instance-level default

        Returns
        -------
        FraudResult with fraud_score, risk_level, and top_matches
        """
        k = top_k or self.top_k
        embedding = self._embedder.embed(image).reshape(1, -1)

        # FAISS inner-product = cosine similarity (embeddings are L2-normalised)
        similarities, indices = self._index.search(embedding, k)
        sims = similarities[0].tolist()
        idxs = indices[0].tolist()

        matches: list[FraudMatch] = []
        for rank, (sim, idx) in enumerate(zip(sims, idxs), start=1):
            if idx < 0:   # FAISS returns -1 when fewer than k results exist
                continue
            meta = self._metadata[idx]
            matches.append(
                FraudMatch(
                    rank=rank,
                    similarity=float(sim),
                    image_path=meta.get("image_path", ""),
                    incident=meta,
                )
            )

        fraud_score = float(np.mean([m.similarity for m in matches])) if matches else 0.0
        fraud_score = max(0.0, min(1.0, fraud_score))

        return FraudResult(
            fraud_score=fraud_score,
            risk_level=_risk_level(fraud_score),
            top_matches=matches,
        )
