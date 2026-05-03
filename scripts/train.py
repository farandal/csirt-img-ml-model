#!/usr/bin/env python3
"""
Build the FAISS fraud index from the csirt_extracted dataset.

Usage:
    python train.py [--dataset dataset/csirt_extracted] [--artifacts fraud_model/artifacts]
"""

import argparse
import json
import sys
from pathlib import Path

# torch must be imported before faiss on macOS (BLAS/AVX SIGSEGV prevention)
import torch  # noqa: F401
import faiss
import numpy as np
from tqdm import tqdm

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent))

from fraud_model.embedder import FraudImageEmbedder
from fraud_model.metadata import parse_txt

IMAGE_EXTS = {".jpeg", ".jpg", ".png", ".webp", ".gif"}


def find_pairs(dataset_dir: Path) -> list[tuple[Path, Path]]:
    """Return (image_path, txt_path) pairs for every image in the dataset."""
    pairs: list[tuple[Path, Path]] = []
    for img_path in sorted(dataset_dir.rglob("*")):
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue
        txt_path = img_path.with_suffix(".txt")
        if not txt_path.exists():
            continue
        pairs.append((img_path, txt_path))
    return pairs


def build_index(
    dataset_dir: Path,
    artifacts_dir: Path,
    batch_size: int = 32,
) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning dataset: {dataset_dir}")
    pairs = find_pairs(dataset_dir)
    if not pairs:
        print("No image/txt pairs found. Check --dataset path.")
        sys.exit(1)

    print(f"Found {len(pairs)} image/txt pairs.")

    embedder = FraudImageEmbedder()
    print(f"Using device: {embedder.device}")

    all_embeddings: list[np.ndarray] = []
    metadata_records: list[dict] = []
    skipped = 0

    # Process in batches for efficiency
    image_paths = [p for p, _ in pairs]
    txt_paths = [t for _, t in pairs]

    for start in tqdm(range(0, len(pairs), batch_size), desc="Embedding"):
        batch_imgs = image_paths[start : start + batch_size]
        batch_txts = txt_paths[start : start + batch_size]

        # Embed images (skip any that can't be loaded)
        valid_imgs: list[Path] = []
        valid_txts: list[Path] = []
        for img, txt in zip(batch_imgs, batch_txts):
            try:
                from PIL import Image
                Image.open(img).verify()
                valid_imgs.append(img)
                valid_txts.append(txt)
            except Exception:
                skipped += 1
                continue

        if not valid_imgs:
            continue

        try:
            embeddings = embedder.embed_batch(valid_imgs, batch_size=len(valid_imgs))
        except Exception as e:
            print(f"\nBatch embedding failed ({e}), falling back to one-by-one")
            embeddings_list = []
            kept_txts = []
            for img, txt in zip(valid_imgs, valid_txts):
                try:
                    embeddings_list.append(embedder.embed(img))
                    kept_txts.append(txt)
                except Exception:
                    skipped += 1
            if not embeddings_list:
                continue
            embeddings = np.stack(embeddings_list)
            valid_txts = kept_txts

        all_embeddings.append(embeddings)

        # Parse metadata for each successfully embedded image
        for img_path, txt_path in zip(valid_imgs, valid_txts):
            try:
                meta = parse_txt(txt_path)
                record = meta.to_dict()
            except Exception:
                record = {
                    "image_path": str(img_path),
                    "txt_path": str(txt_path),
                    "title": "",
                    "alert_code": "",
                    "clase_de_alerta": "",
                    "tipo_de_incidente": "",
                    "nivel_de_riesgo": "",
                    "indicadores": "",
                }
            metadata_records.append(record)

    if not all_embeddings:
        print("No embeddings produced. Aborting.")
        sys.exit(1)

    embeddings_matrix = np.vstack(all_embeddings).astype(np.float32)
    dim = embeddings_matrix.shape[1]

    print(f"\nBuilding FAISS index: {len(metadata_records)} vectors, dim={dim}")

    # IndexFlatIP = exact cosine similarity (vectors are already L2-normalised)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings_matrix)

    # Save artifacts
    index_path = artifacts_dir / "index.faiss"
    meta_path = artifacts_dir / "metadata.json"
    embeddings_path = artifacts_dir / "embeddings.npy"

    faiss.write_index(index, str(index_path))
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata_records, f, ensure_ascii=False, indent=2)
    np.save(str(embeddings_path), embeddings_matrix)

    print(f"Saved index      → {index_path}")
    print(f"Saved metadata   → {meta_path}")
    print(f"Saved embeddings → {embeddings_path}")
    if skipped:
        print(f"Skipped {skipped} unreadable images.")
    print(f"\nDone. Index contains {index.ntotal} fraud image embeddings.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the CSIRT fraud FAISS index.")
    parser.add_argument(
        "--dataset",
        default="dataset/csirt_extracted",
        help="Root of the csirt_extracted folder (default: dataset/csirt_extracted)",
    )
    parser.add_argument(
        "--artifacts",
        default="fraud_model/artifacts",
        help="Where to save index.faiss + metadata.json (default: fraud_model/artifacts)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Images per CLIP batch (default: 32)",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).parent
    build_index(
        dataset_dir=repo_root / args.dataset,
        artifacts_dir=repo_root / args.artifacts,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
