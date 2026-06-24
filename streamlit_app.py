"""
streamlit_app.py - Image-based product identification + RAG Q&A.

Two modes:
  1. Simple mode  - the original 4-step flow (image -> identify -> question -> answer)
                    with filter-first retrieval scoped to the selected product.
  2. Agent mode   - the LLM (Gemini 2.5-flash) decides which of six tools to call,
                    in what order, until it has enough info to answer.
                    The trajectory is streamed live to the UI.

Run from the project root:
    streamlit run streamlit_app.py

For phone access on same network:
    streamlit run streamlit_app.py --server.address 0.0.0.0
Then open http://<your-PC-LAN-ip>:8501 on the phone.
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

# Make src/ importable so we can reuse the existing RAG modules
sys.path.insert(0, str(Path(__file__).parent / "src"))

from identify_product import ProductIdentifier   # noqa: E402
from retrieve import Retriever                   # noqa: E402
from generation import generate_answer           # noqa: E402
from agent import run_agent                      # noqa: E402


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
# Shared helpers
# ============================================================
def list_known_products(identifier):
    return sorted(set(L["product"] for L in identifier.labels))


def chunk_matches_product(chunk, target_product):
    pname    = chunk["metadata"].get("product_name", "")
    chunk_id = chunk.get("chunk_id", "")
    if pname == target_product:
        return True

    def norm(s):
        return s.lower().replace(" ", "").replace("-", "").replace("_", "")

    if norm(pname) == norm(target_product):
        return True
    if norm(target_product) in norm(pname) or norm(pname) in norm(target_product):
        return True
    if target_product in chunk_id:
        return True
    return False


def display_name(folder_name):
    name = folder_name
    for prefix in ("forever-living-products-", "forever-living-"):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    return name.replace("-", " ").title()


def retrieve_within_product(retriever, query, target_product, top_k=10):
    """Filter-first retrieval: rank only chunks of the selected product."""
    product_positions = [
        i for i, chunk in enumerate(retriever.chunks)
        if chunk_matches_product(chunk, target_product)
    ]
    if not product_positions:
        return []
    product_embeddings = np.array(
        [retriever.index.reconstruct(pos) for pos in product_positions],
        dtype=np.float32,
    )
    query_vec = retriever.model.encode(
        [f"query: {query}"], convert_to_numpy=True,
    ).astype(np.float32)[0]
    distances = np.linalg.norm(product_embeddings - query_vec, axis=1)
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


def save_uploaded_to_temp(uploaded_file):
    """Persist a Streamlit-uploaded image to a temp file and return its path."""
    suffix = Path(uploaded_file.name).suffix or ".jpg"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.write(uploaded_file.getvalue())
    tmp.close()
    return tmp.name


# ============================================================
# Page setup
# ============================================================
st.set_page_config(
    page_title="Forever Living Q&A",
    page_icon="🌿",
    layout="centered",
)

st.title("🌿 Forever Living Product Q&A")

with st.spinner("Loading models (first time only, may take 30-60s)..."):
    identifier = load_identifier()
    retriever  = load_retriever()

mode = st.radio(
    "Mode",
    ["Simple mode (direct RAG)", "Agent mode (autonomous tool use)"],
    horizontal=True,
)
st.markdown("---")


# ============================================================
# SIMPLE MODE — the original 4-step flow
# ============================================================
if mode == "Simple mode (direct RAG)":
    st.write("Upload a photo of a product, then ask any question about its ingredients.")

    # Step 1 - Image
    st.subheader("Step 1 — Photo")
    uploaded = st.file_uploader(
        "Take a photo or upload an image",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=False,
        key="simple_uploader",
    )

    identified_product   = None
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

    # Step 2 - Confirm
    st.subheader("Step 2 — Confirm product")
    all_products = list_known_products(identifier)
    default_idx = (
        all_products.index(identified_product)
        if identified_product in all_products else 0
    )
    selected_product = st.selectbox(
        "Which product is this?",
        options=all_products,
        index=default_idx,
        format_func=display_name,
    )

    # Step 3 - Question
    st.subheader("Step 3 — Ask a question")
    question = st.text_area(
        f"What would you like to know about {display_name(selected_product)}?",
        placeholder=f"e.g., What ingredients are in {display_name(selected_product)}?",
        height=100,
        key="simple_question",
    )
    go = st.button("Get answer", type="primary",
                   disabled=(not question.strip()), key="simple_go")

    # Step 4 - Answer
    if go and question.strip():
        st.subheader("Step 4 — Answer")
        with st.spinner("Retrieving relevant chunks (filter-first by product)..."):
            final_chunks = retrieve_within_product(
                retriever, question, selected_product, top_k=10
            )
        if not final_chunks:
            st.warning(
                "No chunks matched the selected product. "
                "Falling back to broad search."
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


# ============================================================
# AGENT MODE — autonomous tool use with streamed trajectory
# ============================================================
else:
    st.write(
        "Ask any question — the agent decides which of six tools to call "
        "(identify from image, retrieve product info, search PubMed, compare products, "
        "get product description, find a similar Forever product) "
        "and shows its reasoning step by step."
    )

    # Optional image
    st.subheader("Optional — upload a photo")
    agent_image = st.file_uploader(
        "If your question is about a product you can show in a photo, upload it here.",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=False,
        key="agent_uploader",
    )
    if agent_image is not None:
        st.image(Image.open(agent_image), caption="Your photo",
                 use_container_width=True)

    # The question
    st.subheader("Your question")
    agent_question = st.text_area(
        "Ask anything about Forever Living products, ingredients, or research.",
        placeholder=(
            "e.g.,\n"
            "  - What does Octocrylene do?\n"
            "  - Compare Aloe Sunscreen and Aloe Lips for moisturizing ingredients.\n"
            "  - Is there scientific research on aloe vera for wound healing?\n"
            "  - [with photo uploaded] What can you tell me about this product?"
        ),
        height=120,
        key="agent_question",
    )
    run = st.button("Run agent", type="primary",
                    disabled=(not agent_question.strip()), key="agent_go")

    # Run the agent and stream the trajectory
    if run and agent_question.strip():
        # Persist the image to a temp file so the agent's tools can read it
        image_path = save_uploaded_to_temp(agent_image) if agent_image else None

        st.subheader("Trajectory")
        traj_placeholder = st.empty()
        traj_lines = []

        final_answer  = None
        final_error   = None

        for event in run_agent(
            agent_question,
            image_path=image_path,
            # Note: we deliberately do NOT pass identifier here. run_agent picks
            # the right backend based on the USE_VISION flag in agent.py
            # (Vision for the agent; Simple mode above keeps using CLIP).
            retriever=retriever,
        ):
            etype = event["type"]

            if etype == "tool_call":
                args_str = ", ".join(f"{k}={v!r}" for k, v in event["args"].items())
                traj_lines.append(
                    f"🔧 **Turn {event['turn']}:** `{event['name']}({args_str})`"
                )
                traj_placeholder.markdown("\n\n".join(traj_lines))

            elif etype == "tool_result":
                preview = json.dumps(event["result"], default=str)
                if len(preview) > 250:
                    preview = preview[:250] + "..."
                traj_lines.append(f"&nbsp;&nbsp;&nbsp;↩ *Result:* `{preview}`")
                traj_placeholder.markdown("\n\n".join(traj_lines), unsafe_allow_html=True)

            elif etype == "final":
                final_answer = event["answer"]

            elif etype == "error":
                final_error = event["message"]

        # Final answer / error section
        st.subheader("Answer")
        if final_answer is not None:
            st.markdown(final_answer)
        if final_error is not None:
            st.error(final_error)
