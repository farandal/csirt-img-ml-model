# CSIRT Fraud Image ML Model

Similarity-based fraud-risk scorer for images of phishing emails, fraudulent websites, and malicious SMS messages published by the Chilean Government CSIRT (Computer Security Incident Response Team).

Given an image, the model returns a **fraud risk score (0–1)** and the **full incident metadata** from the closest matching known-fraud cases in the CSIRT dataset.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Dataset](#2-dataset)
3. [Technology Stack](#3-technology-stack)
4. [ML Architecture](#4-ml-architecture)
5. [Step-by-Step: Dataset Preparation](#5-step-by-step-dataset-preparation)
6. [Step-by-Step: Training (Index Building)](#6-step-by-step-training-index-building)
7. [Step-by-Step: Prediction](#7-step-by-step-prediction)
8. [Artifacts](#8-artifacts)
9. [Risk Score Interpretation](#9-risk-score-interpretation)
10. [Quick Start](#10-quick-start)
11. [Programmatic API](#11-programmatic-api)
12. [Project Structure](#12-project-structure)
13. [Design Decisions and Limitations](#13-design-decisions-and-limitations)

---

## 1. Project Overview

This model was built to support cybersecurity analysts in identifying fraudulent images. Rather than training a binary classifier from scratch (which would require labelled negative examples), the system uses **retrieval-based detection**: every incoming image is compared against a reference library of 3,764 confirmed fraud screenshots. The more visually similar an image is to known fraud, the higher its score.

This approach has several practical advantages:

- **No negative labels needed** — the entire dataset consists of confirmed fraud cases from official CSIRT bulletins.
- **Explainable by design** — every score is backed by specific matching incidents with alert codes, IoC indicators, and source PDFs.
- **Updatable without retraining a neural network** — adding new fraud cases means embedding the new images and inserting them into the index.
- **Zero-shot generalisation** — CLIP's pre-trained semantic understanding lets the model detect new fraud patterns that visually resemble known campaigns, even if the exact content is different.

---

## 2. Dataset

### Source

256 weekly cybersecurity bulletins (PDFs) published by the **CSIRT del Gobierno de Chile** between 2019 and 2024, downloaded from `csirt.gob.cl`.

### Extraction Process

The script `extract_fraud_images.py` (using **PyMuPDF / fitz**) processed each PDF page-by-page:

1. **Header detection** — pages with a full-width banner in the top 30pt were identified as cover pages and skipped.
2. **Logo detection** — the CSIRT logo (always in the upper-right quadrant, x > 60% of page width, y < 220pt) was excluded.
3. **Incident table detection** — only pages whose right column (x > 200pt) contained the string `"Alerta de seguridad"` were treated as fraud incident pages. This filtered out summary pages, IoC-only pages, and recommendation sections.
4. **Image extraction** — for each qualifying page, embedded images were extracted via `fitz.Document.extract_image()`.
5. **Text association** — the incident table text to the right of each image was extracted using positional block analysis and matched to the image by y-position proximity.

### Dataset Statistics

| Metric | Value |
|---|---|
| Source PDFs | 256 |
| PDFs with embedded fraud images | 212 |
| Total image/text pairs | **3,764** |
| JPEG images | 2,935 |
| PNG images | 829 |
| Incident class — Fraude | 2,760 |
| Incident class — Vulnerabilidad | 982 |
| Incident class — Intentos de Intrusión | 19 |
| Incident type — Falsificación de Identidad | 1,286 |
| Incident type — Phishing | 997 |
| Incident type — Sistema/Software Abierto | 982 |
| Incident type — Malware | 347 |
| Incident type — Fraude directo | 105 |

### Paired Text Files

Each image `X.jpeg` has a companion `X.txt` containing structured incident metadata extracted from the table adjacent to the image in the PDF:

```
Source: 13BCS22-000178-01.pdf | Page: 3 | Image: 13BCS22-000178-01_p03_01 (736x376px)
------------------------------------------------------------
CSIRT alerta campaña de phishing que suplanta al SII
Alerta de seguridad cibernética
2CMV22-00349-01

Clase de alerta
Fraude

Tipo de incidente
Malware

Nivel de riesgo
Alto

TLP
Blanco

Fecha de lanzamiento original
23 de septiembre de 2022

Última revisión
23 de septiembre de 2022

Indicadores de compromiso
Asunto
FACTURA – ERICA AMALIA , Notificacion Giro Folio ...
Correo de salida
dukcapiltapinkab@server.tapinkab.go.id
SHA256
cc83ecc8da9069f2e3be95cee116d722163a3d10...
```

---

## 3. Technology Stack

| Component | Library | Version | Role |
|---|---|---|---|
| Image encoder | **open-clip-torch** | 3.3.0 | CLIP ViT-B/32 — converts images to 512-dim semantic vectors |
| Similarity index | **FAISS** (faiss-cpu) | 1.7.4 | Exact cosine-similarity nearest-neighbour search |
| Deep learning runtime | **PyTorch** | 2.2.2 | Tensor operations, model inference |
| Image loading | **Pillow** | 10.x | Opens JPEG/PNG fraud screenshots |
| Numerical computing | **NumPy** | 1.26.x | Embedding matrix operations |
| PDF parsing | **PyMuPDF (fitz)** | 1.22.x | Dataset extraction from PDFs |
| Progress display | **tqdm** | 4.x | Training progress bar |
| Python | **CPython** | 3.9 | Runtime |

### Why CLIP?

CLIP (Contrastive Language-Image Pretraining, OpenAI 2021) was trained on 400 million image-text pairs scraped from the internet. Its visual encoder learns **semantic** image representations rather than low-level pixel statistics. This matters for fraud detection because:

- Two phishing emails impersonating the same bank look visually similar even if the pixel content differs (different fonts, button colours, screenshot crops).
- CLIP understands that a fake SII (tax agency) login page is semantically related to other fake login pages, even from different campaigns.
- It generalises to unseen fraud patterns that share visual structure with known cases.

### Why FAISS IndexFlatIP?

`IndexFlatIP` performs **exact inner-product search**. When the embedding vectors are L2-normalised (unit norm), inner product equals cosine similarity. This gives a score in [0, 1] that is directly interpretable as semantic similarity. No approximation is used — every query compares against all 3,764 vectors for maximum accuracy. At this dataset size (3,764 × 512-dim float32 = 7.7 MB), exact search is instantaneous.

---

## 4. ML Architecture

```
Query Image
    │
    ▼
┌───────────────────────────────────┐
│  CLIP Visual Encoder              │
│  (ViT-B/32-quickgelu, OpenAI)     │
│                                   │
│  1. Resize → 224×224 px           │
│  2. Normalise pixels              │
│  3. Patch tokenisation (32×32)    │
│  4. 12-layer Vision Transformer   │
│  5. Project → 512-dim vector      │
│  6. L2-normalise                  │
└───────────────┬───────────────────┘
                │  query embedding  (512-dim, unit norm)
                ▼
┌───────────────────────────────────┐
│  FAISS IndexFlatIP                │
│  (3,764 reference embeddings)     │
│                                   │
│  cosine_sim(query, ref_i)         │
│    = query · ref_i                │
│    (both unit-norm vectors)       │
│                                   │
│  Returns top-k (default 5)        │
│  most similar reference images    │
└───────────────┬───────────────────┘
                │  [(sim_1, idx_1), …, (sim_k, idx_k)]
                ▼
┌───────────────────────────────────┐
│  Fraud Scorer                     │
│                                   │
│  fraud_score = mean(sim_1…sim_k)  │
│  risk_level  = threshold(score)   │
│                                   │
│  For each match idx_i:            │
│    metadata[idx_i] → incident     │
│    data from the paired .txt file │
└───────────────────────────────────┘
                │
                ▼
          FraudResult
    fraud_score  : 0.0 – 1.0
    risk_level   : MINIMAL / LOW / MEDIUM / HIGH / CRITICAL
    top_matches  : [{similarity, incident_title, alert_code,
                    clase, tipo, ioc_data, ref_image_path}]
```

---

## 5. Step-by-Step: Dataset Preparation

> Script: `extract_fraud_images.py` — run once, output in `dataset/csirt_extracted/`

### Step 1 — Open each PDF

PyMuPDF opens every PDF in `dataset/csirt_pdf/` (256 files). Each page is inspected independently.

### Step 2 — Qualify the page

The page is processed only if the right column (text blocks with `x0 > 200pt`) contains the string `"Alerta de seguridad"`. This string appears exclusively in the CSIRT incident table header. Pages without it (summary dashboards, IoC-only lists, recommendation sections) are skipped entirely.

```python
right_column_blocks = [b for b in page.get_text("blocks") if b.x0 > 200]
boundary_ys = [b.y0 for b in right_column_blocks if "Alerta de seguridad" in b.text]
if not boundary_ys:
    skip page
```

### Step 3 — Extract images

For each image object on a qualified page, three spatial filters are applied:

| Filter | Condition | Reason |
|---|---|---|
| Header | `rect.y0 < 30pt AND rect.width > 45% of page` | Full-width banner at top |
| Logo | `rect.x0 > 60% of page width AND rect.y0 < 220pt` | CSIRT logo in upper right |
| Tiny | `rect.width < 40pt OR rect.height < 30pt` | Invisible filler / decorative pixel |

Images passing all three filters are extracted as raw bytes via `fitz.Document.extract_image()`.

### Step 4 — Associate incident text

Each qualified page may contain one or two incidents, each with a screenshot on the left and a metadata table on the right. The text for each image is found by:

1. Collecting all right-column text blocks sorted by `y0` (top-to-bottom).
2. Identifying incident boundaries: blocks containing `"Alerta de seguridad"` define where each incident starts.
3. Collecting all right-column text from that boundary until the next incident begins.

### Step 5 — Write outputs

For each extracted image a pair of files is written to `dataset/csirt_extracted/{pdf_stem}/`:

- `{pdf_stem}_p{page:02d}_{idx:02d}.jpeg` — the fraud screenshot
- `{pdf_stem}_p{page:02d}_{idx:02d}.txt` — the associated incident metadata

---

## 6. Step-by-Step: Training (Index Building)

> Script: `scripts/train.py` — run once, writes to `fraud_model/artifacts/`
> Runtime: ~9 minutes on CPU (single-core)

Training in this context means **building the reference embedding index**. There are no learnable parameters being optimised. CLIP's weights are frozen pre-trained weights; what changes is the index of fraud-image representations stored in FAISS.

### Step 1 — Discover image/text pairs

```python
pairs = [(img, img.with_suffix(".txt"))
         for img in dataset_dir.rglob("*")
         if img.suffix in IMAGE_EXTS and img.with_suffix(".txt").exists()]
# Result: 3,764 pairs
```

### Step 2 — Load the CLIP model

```python
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32-quickgelu", pretrained="openai"
)
model.eval()
```

`ViT-B-32-quickgelu` uses the QuickGELU activation that matches OpenAI's original weights exactly. The standard `ViT-B-32` tag has an activation mismatch that causes numerical instability on macOS. The model is never put in training mode (`model.train()` is never called) and no gradients are computed.

### Step 3 — Embed images in batches of 32

```python
for batch in chunks(image_paths, 32):
    # 1. Load each image as PIL RGB
    images = [Image.open(p).convert("RGB") for p in batch]

    # 2. Apply CLIP preprocessing:
    #    - Resize shortest side to 224px (bicubic interpolation)
    #    - Centre-crop to 224×224
    #    - Normalise channels:
    #        mean = (0.48145466, 0.4578275, 0.40821073)
    #        std  = (0.26862954, 0.26130258, 0.27577711)
    tensors = torch.stack([preprocess(img) for img in images])  # (B, 3, 224, 224)

    # 3. Forward pass through Vision Transformer — no gradient tracking
    with torch.no_grad():
        features = model.encode_image(tensors)   # (B, 512)

    # 4. L2-normalise so that inner product == cosine similarity
    features = features / features.norm(dim=-1, keepdim=True)

    embeddings.append(features.numpy())          # (B, 512) float32
```

**Vision Transformer internals (ViT-B/32):**

| Property | Value |
|---|---|
| Input resolution | 224 × 224 RGB |
| Patch size | 32 × 32 px |
| Sequence length | 49 patches + 1 CLS token = 50 tokens |
| Embedding dimension | 768 |
| Transformer depth | 12 layers |
| Attention heads | 12 |
| Output projection | 768 → 512 dims |
| Total parameters | ~87 million |

### Step 4 — Build FAISS index

```python
dim   = 512
index = faiss.IndexFlatIP(dim)    # exact inner-product (= cosine for unit vectors)
index.add(embeddings_matrix)      # shape: (3764, 512), dtype: float32
```

`IndexFlatIP` stores every vector verbatim and performs brute-force inner-product search at query time. With 3,764 vectors at 512 dimensions each, a single query takes under 1 ms on CPU.

### Step 5 — Parse and store metadata

Each `.txt` file is parsed into an `IncidentMetadata` dataclass:

```python
@dataclass
class IncidentMetadata:
    source_pdf: str        # "13BCS22-000178-01.pdf"
    page: int              # page number within the source PDF
    image_path: str        # absolute path to the .jpeg / .png file
    txt_path: str          # absolute path to this .txt file
    title: str             # incident headline from the bulletin
    alert_code: str        # "2CMV22-00349-01"
    clase_de_alerta: str   # "Fraude" / "Vulnerabilidad" / …
    tipo_de_incidente: str # "Phishing" / "Malware" / "Falsificación de Identidad" / …
    nivel_de_riesgo: str   # "Alto" / "Medio" / "Bajo"
    tlp: str               # "Blanco" / "Verde" / …
    fecha_lanzamiento: str # original publication date
    ultima_revision: str   # last revision date
    indicadores: str       # full IoC block: URLs, IPs, SHA-256 hashes, email metadata
```

The 3,764 metadata records are serialised as a JSON array ordered identically to the FAISS index — `metadata[i]` always corresponds to the embedding at row `i`.

### Step 6 — Save artifacts

```
fraud_model/artifacts/
├── index.faiss       7.7 MB   FAISS index (3764 × 512 float32 vectors)
├── metadata.json     3.0 MB   Parsed incident metadata (JSON array, 3764 objects)
└── embeddings.npy    7.7 MB   Raw NumPy array backup (shape: 3764 × 512)
```

Total artifact footprint: **18.4 MB**

---

## 7. Step-by-Step: Prediction

> Script: `scripts/predict.py` · Class: `fraud_model.scorer.FraudDetector`

### Step 1 — Load artifacts

On first instantiation, `FraudDetector` loads the FAISS index, metadata JSON, and initialises the CLIP embedder. Subsequent calls reuse the already-loaded objects.

```python
detector = FraudDetector()          # default: fraud_model/artifacts/
detector = FraudDetector(artifacts_dir="/path/to/artifacts", top_k=3)
```

### Step 2 — Embed the query image

The query image undergoes the **identical preprocessing pipeline** used during training (resize → centre-crop 224×224 → channel normalise → ViT-B/32 forward pass → L2-normalise), producing a 512-dim unit vector.

Accepting the same preprocessing for training and inference is essential: the FAISS index was built from embeddings computed with this exact transform, so any deviation would produce wrong similarity scores.

### Step 3 — Nearest-neighbour search

```python
# query_embedding: shape (1, 512), unit norm
similarities, indices = index.search(query_embedding, k=5)
# similarities: shape (1, 5) — cosine similarity values, highest first
# indices:      shape (1, 5) — row indices into the metadata array
```

Because both query and reference vectors are L2-normalised, the inner product equals cosine similarity. A value of 1.0 means the query is identical (in embedding space) to a known fraud image; values above ~0.65 indicate strong visual similarity to a fraud campaign.

### Step 4 — Compute fraud score

```python
fraud_score = mean(similarities[0])   # average of top-5 cosine similarities
fraud_score = clamp(fraud_score, 0.0, 1.0)
```

Taking the **mean** of the top-5 rather than the maximum makes the score more robust — a single coincidental high similarity cannot alone inflate the score to a dangerous level.

### Step 5 — Assign risk level

| Score threshold | Risk level | Interpretation |
|---|---|---|
| ≥ 0.80 | **CRITICAL** | Near-identical to a known fraud image |
| ≥ 0.65 | **HIGH** | Strong visual similarity to a fraud campaign |
| ≥ 0.50 | **MEDIUM** | Noticeable resemblance to fraud patterns |
| ≥ 0.35 | **LOW** | Weak resemblance; warrants manual review |
| < 0.35 | **MINIMAL** | Visually dissimilar to all known fraud |

### Step 6 — Return enriched result

```python
FraudResult(
    fraud_score = 0.909,
    risk_level  = "CRITICAL",
    top_matches = [
        FraudMatch(
            rank       = 1,
            similarity = 1.000,
            image_path = "dataset/csirt_extracted/13BCS22-000178-01/…jpeg",
            incident   = {
                "title":             "CSIRT alerta campaña de phishing que suplanta al SII",
                "alert_code":        "2CMV22-00349-01",
                "clase_de_alerta":   "Fraude",
                "tipo_de_incidente": "Malware",
                "nivel_de_riesgo":   "Alto",
                "fecha_lanzamiento": "23 de septiembre de 2022",
                "indicadores":       "Asunto\n✅ FACTURA – ERICA AMALIA …\nSHA256\ncc83ecc8…",
                …
            }
        ),
        …
    ]
)
```

---

## 8. Artifacts

After running `scripts/train.py`, three files are written to `fraud_model/artifacts/`:

**`index.faiss` (7.7 MB)**
Binary FAISS index. Contains all 3,764 L2-normalised 512-dim float32 embeddings. Loaded with `faiss.read_index()`. Supports sub-millisecond exact nearest-neighbour queries.

**`metadata.json` (3.0 MB)**
JSON array of 3,764 objects. Element `i` corresponds to the embedding at row `i` in the FAISS index. Contains all fields of `IncidentMetadata` including the full IoC block.

**`embeddings.npy` (7.7 MB)**
NumPy array of shape `(3764, 512)`, dtype `float32`. Backup of the embedding matrix; useful for offline analysis, clustering, or rebuilding the index with different FAISS settings without re-running CLIP.

---

## 9. Risk Score Interpretation

The fraud score is a **cosine similarity in CLIP embedding space**, not a calibrated probability.

- **1.00** — Query image is identical (or near-identical) to a fraud image in the index. Happens when querying an image that is itself part of the training set.
- **0.80–0.95** — Same fraud campaign: same bank being impersonated, same SMS template, same phishing kit with minor visual changes.
- **0.65–0.79** — Related campaign: different target brand or variant template that shares visual structure with a known campaign.
- **0.50–0.64** — Structural resemblance: the layout, colour scheme, or UI elements are similar to fraud patterns but the content diverges.
- **< 0.50** — Likely new fraud type or legitimate image. Scores below 0.35 against a purely fraud-image corpus are uncommon for actual phishing content.

---

## 10. Quick Start

### Setup

```bash
cd csirt-img-ml-model
python3.9 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Build the dataset (if not already done)

```bash
# Requires PyMuPDF: pip install pymupdf
python scripts/extract_fraud_images.py
# Input:  dataset/csirt_pdf/    (256 PDFs)
# Output: dataset/csirt_extracted/  (3,764 image+txt pairs)
```

### Train — build the FAISS index

```bash
python scripts/train.py
# Optional:
#   --dataset   dataset/csirt_extracted   (default)
#   --artifacts fraud_model/artifacts     (default)
#   --batch-size 32                       (images per CLIP batch)
#
# Runtime: ~9 min on CPU. CLIP model (~340 MB) is auto-downloaded on first run.
```

### Predict

```bash
# Formatted terminal output
python scripts/predict.py path/to/suspicious_image.jpg

# Limit to top 3 matches
python scripts/predict.py path/to/image.jpg --top-k 3

# Machine-readable JSON
python scripts/predict.py path/to/image.jpg --json
```

---

## 11. Programmatic API

```python
from fraud_model import FraudDetector

# Load once, reuse for many queries
detector = FraudDetector()
# Or with custom paths:
# detector = FraudDetector(artifacts_dir="fraud_model/artifacts", top_k=3)

# Score an image — accepts file path (str/Path) or PIL.Image
result = detector.score("suspicious.jpg")

# Top-level result
print(result.fraud_score)   # float 0.0–1.0,  e.g. 0.871
print(result.risk_level)    # str,             e.g. "HIGH"

# Nearest-neighbour matches
for match in result.top_matches:
    print(match.rank)                           # 1, 2, 3, …
    print(match.similarity)                     # e.g. 0.924
    print(match.incident["title"])              # "CSIRT alerta campaña de phishing…"
    print(match.incident["alert_code"])         # "2CMV22-00349-01"
    print(match.incident["clase_de_alerta"])    # "Fraude"
    print(match.incident["tipo_de_incidente"])  # "Phishing"
    print(match.incident["nivel_de_riesgo"])    # "Alto"
    print(match.incident["fecha_lanzamiento"])  # "23 de septiembre de 2022"
    print(match.incident["indicadores"])        # full IoC block (URLs, IPs, hashes)
    print(match.incident["image_path"])         # path to the reference image

# Serialise to dict / JSON
import json
print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
```

---

## 12. Project Structure

```
csirt-img-ml-model/
│
├── dataset/
│   ├── csirt_extracted/           # 3,764 image+txt pairs in 212 subdirectories
│   │   ├── 13BCS20-00054-01/
│   │   │   ├── 13BCS20-00054-01_p03_01.jpeg
│   │   │   ├── 13BCS20-00054-01_p03_01.txt
│   │   │   └── …
│   │   └── …
│   ├── csirt_pdf/                 # 256 original CSIRT bulletin PDFs
│   └── csirt_pdf_compressed/      # 256 compressed PDF copies
│
├── fraud_model/                   # installable Python package
│   ├── __init__.py                # public API: FraudDetector, FraudResult, parse_txt
│   ├── embedder.py                # FraudImageEmbedder — CLIP ViT-B/32 wrapper
│   ├── metadata.py                # parse_txt() — .txt parser → IncidentMetadata
│   ├── scorer.py                  # FraudDetector — loads index, scores images
│   └── artifacts/                 # generated by train.py
│       ├── index.faiss            # FAISS similarity index       (7.7 MB)
│       ├── metadata.json          # incident metadata JSON array (3.0 MB)
│       └── embeddings.npy         # raw embedding matrix backup  (7.7 MB)
│
├── scripts/
│   ├── compress_pdf.py            # Single-PDF compression utility
│   ├── compress_pdfs.py           # Batch PDF compression utility
│   ├── extract_fraud_images.py    # Step 1: extract images+txt from PDFs
│   ├── train.py                   # Step 2: build FAISS index from extracted images
│   └── predict.py                 # Step 3: CLI to score a query image
├── requirements.txt               # Python dependencies
└── README.md                      # this file
```

---

## 13. Design Decisions and Limitations

**No negative examples.** The model has no concept of "definitely not fraud" from its training data. Scores are relative to the fraud corpus, not absolute probabilities. A legitimate image that visually resembles a CSIRT document page will score higher than an unrelated image. For production use, calibrate thresholds against a held-out set that includes legitimate images.

**Language-agnostic.** CLIP processes visual features, not OCR text. Fraud content in any language is assessed equally, as long as the visual structure resembles known patterns.

**Frozen encoder.** The CLIP weights are never fine-tuned on the CSIRT data. Fine-tuning on 3,764 images without any negative examples would overfit. CLIP's zero-shot generalisation is preferred.

**Exact search.** `IndexFlatIP` is exact (no approximation). For this corpus size this is optimal — query time is under 1 ms. If the corpus grows beyond ~1 million images, consider switching to `IndexIVFFlat` or `IndexHNSWFlat` for sub-linear query time at the cost of minor accuracy reduction.

**Adding new fraud cases without full retraining.** New confirmed fraud images can be incrementally added to the index:

```python
from fraud_model.embedder import FraudImageEmbedder
from fraud_model.metadata import parse_txt
import faiss, json, numpy as np

embedder = FraudImageEmbedder()
index    = faiss.read_index("fraud_model/artifacts/index.faiss")
metadata = json.load(open("fraud_model/artifacts/metadata.json"))

# Embed and add new images
new_embeddings = embedder.embed_batch(new_image_paths)   # (N, 512)
index.add(new_embeddings)
faiss.write_index(index, "fraud_model/artifacts/index.faiss")

# Append metadata
for txt_path in new_txt_paths:
    metadata.append(parse_txt(txt_path).to_dict())
json.dump(metadata, open("fraud_model/artifacts/metadata.json", "w"),
          ensure_ascii=False, indent=2)
```

**macOS BLAS conflict.** On macOS (Intel), PyTorch and FAISS both link against AVX2 BLAS routines. If FAISS is imported before PyTorch, the competing initialisation causes a SIGSEGV. All modules in this package import `torch` before `faiss` to prevent this.
