"""
streamlit_app.py - Image-based product identification + RAG Q&A.

A mobile-friendly Streamlit interface that ties together three pieces:
  1. CLIP-based product identification from a user-uploaded photo
  2. Existing FAISS retrieval over the product knowledge base
  3. Gemini-based grounded answer generation

For a confirmed product, retrieval uses a filter-first strategy: chunks are
filtered to the selected product by metadata first, then ranked by similarity
within that subset. This guarantees that aggregative questions ("which
ingredients in X act as moisturizers?") see all of the product's ingredient
chunks, not just whichever ones won a broad top-k against other products.

Run from the project root:
    streamlit run streamlit_app.py

For phone access on same network:
    streamlit run streamlit_app.py --server.address 0.0.0.0
Then open http://<your-PC-LAN-ip>:8501 on the phone.
"""

import sys
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

# Make src/ importable so we can reuse the existing RAG modules
sys.path.insert(0, str(Path(__file__).parent / "src"))

from identify_product import ProductIdentifier   # noqa: E402
from retrieve import Retriever                   # noqa: E402
from generation import generate_answer           # noqa: E402


# ============================================================
# Cached resource loaders (run once per Streamlit session)
# ============================================================
@st.cache_resource
def load_identifier():
    return ProductIdentifier()


@st.cache_resource
def load_retriever():
    return Retriever()


# ============================================================
# Helpers
# ============================================================
def list_known_products(identifier):
    """Sorted list of all product folder names present in the image index."""
    return sorted(set(L["product"] for L in identifier.labels))


def chunk_matches_product(chunk, target_product):
    """
    Decide if a retrieved chunk belongs to the user-selected product.

    The folder-name format (e.g. 'forever-living-aloe-lips') may not match the
    chunks' metadata['product_name'] (e.g. 'Forever Living Aloe Lips') exactly,
    so we try several strategies.
    """
    pname    = chunk["metadata"].get("product_name", "")
    chunk_id = chunk.get("chunk_id", "")

    # 1) Exact match
    if pname == target_product:
        return True

    # 2) Normalized match (lowercase, drop all separators)
    def norm(s):
        return s.lower().replace(" ", "").replace("-", "").replace("_", "")

    if norm(pname) == norm(target_product):
        return True
    if norm(target_product) in norm(pname) or norm(pname) in norm(target_product):
        return True

    # 3) Fallback: chunk_id contains the product slug
    if target_product in chunk_id:
        return True

    return False


def display_name(folder_name):
    """'forever-living-aloe-ever-shield' -> 'Aloe Ever Shield' for the UI."""
    name = folder_name
    for prefix in ("forever-living-products-", "forever-living-"):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    return name.replace("-", " ").title()


def retrieve_within_product(retriever, query, target_product, top_k=5):
    """
    Filter-first retrieval: rank only the chunks of the selected product.

    Unlike retriever.retrieve() (which searches the full index then post-filters
    by product), this filters by metadata first, then ranks the survivors by
    similarity to the query. This guarantees that all of the product's chunks
    are considered, which matters for aggregative questions like 'which
    ingredients in X act as moisturizers?' where several chunks may qualify.

    Returns the same shape as retriever.retrieve(): a list of
    {"rank", "distance", "chunk"} dicts sorted by ascending L2 distance.
    """
    # 1. Find positions of chunks belonging to the selected product
    product_positions = [
        i for i, chunk in enumerate(retriever.chunks)
        if chunk_matches_product(chunk, target_product)
    ]
    if not product_positions:
        return []

    # 2. Reconstruct their embeddings from the FAISS index (no re-encoding)
    product_embeddings = np.array(
        [retriever.index.reconstruct(pos) for pos in product_positions],
        dtype=np.float32,
    )

    # 3. Encode the query with the same prefix the index was built with
    query_vec = retriever.model.encode(
        [f"query: {query}"],
        convert_to_numpy=True,
    ).astype(np.float32)[0]

    # 4. L2 distance between query and each product chunk (same metric as FAISS)
    distances = np.linalg.norm(product_embeddings - query_vec, axis=1)

    # 5. Sort ascending and take top-k
    order = np.argsort(distances)[: min(top_k, len(distances))]
    results = []
    for rank, local_idx in enumerate(order, start=1):
        pos = product_positions[int(local_idx)]
        results.append({
            "rank":     rank,
            "distance": float(distances[int(local_idx)]),
            "chunk":    retriever.chunks[pos],
        })
    return results


# ============================================================
# Page setup
# ============================================================
st.set_page_config(
    page_title="Forever Living Q&A",
    page_icon="🌿",
    layout="centered",
)

st.title("🌿 Forever Living Product Q&A")
st.write("Upload a photo of a product, then ask any question about its ingredients.")

with st.spinner("Loading models (first time only, may take 30-60s)..."):
    identifier = load_identifier()
    retriever  = load_retriever()


# ============================================================
# Step 1 - Image upload + identification
# ============================================================
st.subheader("Step 1 — Photo")
uploaded = st.file_uploader(
    "Take a photo or upload an image",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=False,
)

identified_product = None
identification_result = None

if uploaded is not None:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Your photo", use_container_width=True)

    with st.spinner("Identifying product..."):
        identification_result = identifier.identify(img)

    if identification_result["is_confident"]:
        identified_product = identification_result["product"]
        st.success(
            f"Identified: **{display_name(identified_product)}** "
            f"(confidence: {identification_result['confidence']:.0%})"
        )
    else:
        st.warning(
            f"Not sure what this is (best guess: "
            f"**{display_name(identification_result['top_matches'][0]['product'])}** "
            f"at {identification_result['confidence']:.0%} confidence). "
            f"Please select manually below."
        )

    with st.expander("See top candidates"):
        for i, m in enumerate(identification_result["top_matches"], 1):
            st.write(f"{i}. **{display_name(m['product'])}** — score: {m['score']:.3f}")


# ============================================================
# Step 2 - Confirm or override product
# ============================================================
st.subheader("Step 2 — Confirm product")
all_products = list_known_products(identifier)
default_idx  = (
    all_products.index(identified_product)
    if identified_product in all_products else 0
)
selected_product = st.selectbox(
    "Which product is this?",
    options=all_products,
    index=default_idx,
    format_func=display_name,
)


# ============================================================
# Step 3 - Ask a question
# ============================================================
st.subheader("Step 3 — Ask a question")
question = st.text_area(
    f"What would you like to know about {display_name(selected_product)}?",
    placeholder=f"e.g., What ingredients are in {display_name(selected_product)}? "
                f"What does the third ingredient do?",
    height=100,
)

go = st.button("Get answer", type="primary", disabled=(not question.strip()))


# ============================================================
# Step 4 - Filter-first retrieve, then generate
# ============================================================
if go and question.strip():
    st.subheader("Step 4 — Answer")

    with st.spinner("Retrieving relevant chunks (filter-first by product)..."):
        final_chunks = retrieve_within_product(
            retriever, question, selected_product, top_k=10
        )

    if not final_chunks:
        st.warning(
            "No chunks matched the selected product. Falling back to broad search. "
            "(If you see this often, the product-name format in chunks.jsonl may need a small adjustment.)"
        )
        final_chunks = retriever.retrieve(question, top_k=5)

    with st.spinner("Generating answer..."):
        answer = generate_answer(question, final_chunks)

    st.markdown(answer)

    with st.expander("Sources used"):
        for r in final_chunks:
            chunk = r["chunk"]
            st.markdown(
                f"**{chunk['metadata'].get('product_name', 'Unknown')}** "
                f"(chunk_id: `{chunk['chunk_id']}`, distance: {r['distance']:.3f})"
            )
            preview = chunk["text"][:300] + ("..." if len(chunk["text"]) > 300 else "")
            st.text(preview)
            st.markdown("---")
