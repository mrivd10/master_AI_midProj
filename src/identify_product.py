"""
identify_product.py - Identify a Forever Living product from a photo using CLIP.

Loads the reference image index built by build_image_index.py and matches
new photos against it via nearest-neighbor search in CLIP embedding space.

Usage as a module:
    from src.identify_product import ProductIdentifier
    identifier = ProductIdentifier()
    result = identifier.identify("path/to/photo.jpg")
    print(result["product"], result["confidence"])

Usage from CLI (for testing):
    python src/identify_product.py path/to/photo.jpg
"""

import json
import sys
from pathlib import Path
from typing import Dict, Union

import faiss
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer

# --- Config ---
MODEL_NAME             = "clip-ViT-B-32"
INDEX_PATH             = Path("data/processed/image_index.faiss")
LABELS_PATH            = Path("data/processed/image_labels.json")
DEFAULT_TOP_K          = 5
DEFAULT_MIN_CONFIDENCE = 0.70   # cosine similarity threshold for a confident match


class ProductIdentifier:
    """Identify products from images using CLIP-based nearest-neighbor search."""

    def __init__(
        self,
        index_path: Path = INDEX_PATH,
        labels_path: Path = LABELS_PATH,
        model_name: str = MODEL_NAME,
    ):
        if not index_path.exists():
            raise FileNotFoundError(
                f"Index not found: {index_path}\n"
                f"Run build_image_index.py first to create it."
            )
        if not labels_path.exists():
            raise FileNotFoundError(
                f"Labels not found: {labels_path}\n"
                f"Run build_image_index.py first to create them."
            )

        print(f"Loading CLIP model: {model_name}")
        self.model = SentenceTransformer(model_name)

        print(f"Loading FAISS index: {index_path}")
        self.index = faiss.read_index(str(index_path))

        print(f"Loading labels: {labels_path}")
        with labels_path.open("r", encoding="utf-8") as f:
            self.labels = json.load(f)

        n_products = len(set(L["product"] for L in self.labels))
        print(f"  {self.index.ntotal} reference images across {n_products} products")

    def identify(
        self,
        image: Union[str, Path, Image.Image],
        top_k: int = DEFAULT_TOP_K,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    ) -> Dict:
        """
        Identify the product in the given image.

        Args:
            image: file path or PIL Image
            top_k: how many nearest reference images to retrieve
            min_confidence: cosine similarity below which we declare 'uncertain'

        Returns dict with:
            product:       best matching product name (None if uncertain)
            confidence:    cosine similarity of the best match in [0, 1]
            is_confident:  True if confidence >= min_confidence
            top_matches:   list of {product, score, image_path}, best per product
        """
        # Load and encode the query image
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            image = image.convert("RGB")
        else:
            raise TypeError(f"image must be path or PIL.Image, got {type(image)}")

        query_emb = self.model.encode(
            [image],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")

        # FAISS inner-product search (normalized vectors => cosine similarity)
        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_emb, k)
        scores, indices = scores[0], indices[0]

        # Top-1 reference image decides the identification
        top1_score   = float(scores[0])
        top1_product = self.labels[int(indices[0])]["product"]

        # Build per-product summary: best score per unique product in top-K
        seen = {}
        for score, idx in zip(scores, indices):
            entry = self.labels[int(idx)]
            product = entry["product"]
            if product not in seen:
                seen[product] = {
                    "product":    product,
                    "score":      round(float(score), 4),
                    "image_path": entry["image_path"],
                }
        top_matches = list(seen.values())

        return {
            "product":      top1_product if top1_score >= min_confidence else None,
            "confidence":   round(top1_score, 4),
            "is_confident": top1_score >= min_confidence,
            "top_matches":  top_matches,
        }


def main():
    """CLI: identify a single image and print the result."""
    if len(sys.argv) != 2:
        print("Usage: python src/identify_product.py <image_path>")
        sys.exit(1)

    image_path = Path(sys.argv[1])
    if not image_path.exists():
        print(f"Image not found: {image_path}")
        sys.exit(1)

    identifier = ProductIdentifier()
    result = identifier.identify(image_path)

    print()
    print("=" * 70)
    print(f"  Query image: {image_path}")
    print("=" * 70)
    if result["is_confident"]:
        print(f"  Identified : {result['product']}")
        print(f"  Confidence : {result['confidence']:.3f}  (above {DEFAULT_MIN_CONFIDENCE} threshold)")
    else:
        print(f"  Identified : UNCERTAIN -- best guess below confidence threshold")
        print(f"  Best guess : {result['top_matches'][0]['product']}")
        print(f"  Confidence : {result['confidence']:.3f}  (below {DEFAULT_MIN_CONFIDENCE} threshold)")
        print(f"  Action     : Streamlit UI should offer manual product selection")
    print()
    print("  Top matches (best score per product in top-K):")
    for i, m in enumerate(result["top_matches"], 1):
        marker = ">" if i == 1 else " "
        print(f"  {marker} {i}. {m['product']:55s} score={m['score']:.3f}")
    print()


if __name__ == "__main__":
    main()
