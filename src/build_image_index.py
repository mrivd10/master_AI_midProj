"""
build_image_index.py - Build the CLIP reference index from product photos.

One-time setup script. Reads images from data/raw/product_images/<product>/,
encodes each one with CLIP via sentence-transformers, and saves:
  - data/processed/image_index.faiss   FAISS index of L2-normalized embeddings
  - data/processed/image_labels.json   vector_idx -> {product, image_path}

At query time, identify_product.py loads both files and looks up the nearest
reference image(s) to a new photo from the user.

Run from the project root:
    python src/build_image_index.py
"""

import json
from pathlib import Path

import faiss
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer

# --- Config ---
MODEL_NAME  = "clip-ViT-B-32"
IMAGES_DIR  = Path("data/raw/product_images")
INDEX_OUT   = Path("data/processed/image_index.faiss")
LABELS_OUT  = Path("data/processed/image_labels.json")
IMAGE_EXTS  = {".jpg", ".jpeg", ".png", ".webp"}


def collect_image_paths():
    """Walk IMAGES_DIR/<product>/<image> and return (product_name, path) pairs."""
    if not IMAGES_DIR.exists():
        raise FileNotFoundError(
            f"Images directory not found: {IMAGES_DIR}\n"
            f"Create it with one subfolder per product, each holding 3-5 photos."
        )

    pairs = []
    for product_dir in sorted(IMAGES_DIR.iterdir()):
        if not product_dir.is_dir():
            continue
        images = [p for p in sorted(product_dir.iterdir())
                  if p.suffix.lower() in IMAGE_EXTS]
        if not images:
            print(f"  WARN: no images in {product_dir.name}")
            continue
        for img_path in images:
            pairs.append((product_dir.name, img_path))
    return pairs


def encode_images(pairs, model):
    """Encode each image into a normalized CLIP embedding (float32)."""
    images      = []
    valid_pairs = []
    for product_name, img_path in pairs:
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"  ERROR loading {img_path}: {e}")
            continue
        images.append(img)
        valid_pairs.append((product_name, img_path))

    # sentence-transformers handles batching, normalization, and progress bar
    embeddings = model.encode(
        images,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    return embeddings.astype("float32"), valid_pairs


def main():
    print(f"Loading CLIP model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    print(f"Collecting images from {IMAGES_DIR}")
    pairs      = collect_image_paths()
    n_products = len(set(p[0] for p in pairs))
    print(f"  found {len(pairs)} images across {n_products} products")

    print("Encoding images with CLIP...")
    embeddings, valid_pairs = encode_images(pairs, model)
    print(f"  encoded {embeddings.shape[0]} images -> {embeddings.shape[1]}-d vectors")

    print("Building FAISS index (inner product on normalized vectors = cosine)")
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    INDEX_OUT.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_OUT))

    labels = [
        {"vector_idx": i, "product": product, "image_path": str(path)}
        for i, (product, path) in enumerate(valid_pairs)
    ]
    with LABELS_OUT.open("w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)

    print("\n=== DONE ===")
    print(f"  Index : {INDEX_OUT}  ({index.ntotal} vectors)")
    print(f"  Labels: {LABELS_OUT}")


if __name__ == "__main__":
    main()
