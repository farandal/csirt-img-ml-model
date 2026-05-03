"""CLIP ViT-B/32 image embedder using open_clip_torch.

open_clip avoids the torch.load security restriction in newer transformers,
works with torch>=2.0, and is the canonical CLIP inference library.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import open_clip
from PIL import Image

# quickgelu variant matches the OpenAI pretrained weights exactly (no mismatch warning)
_MODEL_NAME = "ViT-B-32-quickgelu"
_PRETRAINED = "openai"


class FraudImageEmbedder:
    """Wraps CLIP (open_clip) to produce L2-normalised 512-dim embeddings."""

    def __init__(self, device: str | None = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            _MODEL_NAME, pretrained=_PRETRAINED
        )
        self.model = self.model.to(self.device)
        self.model.eval()

    def _load(self, source: str | Path | Image.Image) -> Image.Image:
        if isinstance(source, Image.Image):
            return source.convert("RGB")
        return Image.open(source).convert("RGB")

    def _infer(self, tensors: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            features = self.model.encode_image(tensors)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().astype(np.float32)

    def embed(self, source: str | Path | Image.Image) -> np.ndarray:
        """Return a normalised (512,) embedding for a single image."""
        tensor = self.preprocess(self._load(source)).unsqueeze(0).to(self.device)
        return self._infer(tensor)[0]

    def embed_batch(
        self,
        sources: list[str | Path | Image.Image],
        batch_size: int = 32,
    ) -> np.ndarray:
        """Return a (N, 512) array of normalised embeddings."""
        all_embeddings: list[np.ndarray] = []

        for start in range(0, len(sources), batch_size):
            batch = sources[start : start + batch_size]
            tensors = torch.stack(
                [self.preprocess(self._load(s)) for s in batch]
            ).to(self.device)
            all_embeddings.append(self._infer(tensors))

        return np.vstack(all_embeddings)
