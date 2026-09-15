"""
GTA Multimodal RAG Search - Streamlit Web Application.

File:
    app.py

Purpose:
    Interactive web application providing natural language scene search across the
    HuggingFace Grand Theft Auto image captioning dataset. Combines CLIP multimodal
    embeddings with dual-stream Reciprocal Rank Fusion (RRF) vector search and Google
    Gemini for pre-indexed or on-demand multimodal visual reasoning.

Usage / CLI Invocation:
    streamlit run app.py
    streamlit run app.py --server.port 8501

Prerequisites & Dependencies:
    - Python 3.10+
    - streamlit >= 1.38.0
    - sentence-transformers >= 3.0.0
    - chromadb >= 0.5.5
    - google-generativeai >= 0.8.0
    - python-dotenv >= 1.0.1
    - HuggingFace dataset cache in `data/` or internet access for initial download.
    - GOOGLE_API_KEY set in `.env` (copy from `.env.example`).

Inputs & Outputs:
    - Inputs: Natural language text query entered by the user in the Streamlit UI,
      optional custom multimodal questions for deep-dive scene analysis.
    - Outputs: Interactive web interface rendering matching GTA scene images,
      original dataset captions, rich pre-indexed or live Gemini scene descriptions,
      search latency metrics, and index provenance badges.
"""

from __future__ import annotations

import os
import time
from typing import Any, Tuple

from dotenv import load_dotenv
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import streamlit as st

from utils.utils import (
    get_image_description,
    get_index_metadata,
    get_logger,
    get_search_result,
    initialization,
)

# ---------------------------------------------------------------------------
# Global Setup & Environment Configuration
# ---------------------------------------------------------------------------
# Load environment variables from .env (overrides system defaults to ensure fresh keys)
load_dotenv(override=True)

# Configure Streamlit page layout and title
# 'wide' mode allows side-by-side display of retrieved images and LLM descriptions
st.set_page_config(
    page_title="GTA Multimodal RAG Search",
    page_icon="🎮",
    layout="wide"
)


# ---------------------------------------------------------------------------
# Cached Resource Initialization
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Initializing GTA vector database and CLIP model...")
def get_resources() -> Tuple[Any, Any, SentenceTransformer, Any]:
    """
    Initialize and cache expensive application resources across user interactions.

    Streamlit executes scripts top-to-bottom on every user event (button click, text change).
    Using `@st.cache_resource` ensures that heavy objects—such as the SentenceTransformer CLIP
    neural weights (~600MB), the local ChromaDB persistent index, and dataset pickles—are
    instantiated exactly once in memory and reused across all subsequent script reruns.

    Returns:
        Tuple[Any, Any, SentenceTransformer, Any]:
            - collection: ChromaDB collection instance (prioritizing v2 hybrid collection).
            - data_set: Hugging Face dataset dictionary containing images and captions.
            - model: Loaded SentenceTransformer CLIP model instance for vector encoding.
            - logger: Fully configured Python Logger instance with dual console/file handlers.
    """
    # Retrieve the root application logger
    logger = get_logger()

    # Orchestrate folder setup, dataset loading, model weights download, and DB connection
    collection, data_set, model, logger = initialization(logger)

    return collection, data_set, model, logger


# ---------------------------------------------------------------------------
# Provenance & Status UI Components
# ---------------------------------------------------------------------------
def render_index_badge(collection: Any) -> None:
    """
    Display a provenance badge showing which models built the current active index.

    Inspects collection-level metadata stored in ChromaDB to inform the user whether
    they are operating on the state-of-the-art dual-vector hybrid index (v2.0.0 with
    pre-computed Gemini descriptions) or the legacy image-only vector index (v1.0.0).

    Args:
        collection (Any): Active ChromaDB collection object.
    """
    # Extract metadata attributes stored during the offline indexing pipeline
    meta = get_index_metadata(collection)

    # Branch display according to collection version
    if meta.get("is_hybrid"):
        # Hybrid Index v2: show embedding model, LLM generator model, document count, and creation date
        st.caption(
            f"🏷️ **Index v{meta['index_version']}** · "
            f"Embedding: `{meta['embedding_model']}` · "
            f"Descriptions: `{meta['llm_model']}` · "
            f"Docs: {meta['document_count']} · "
            f"Created: {meta['created_at'][:10] if meta['created_at'] != 'unknown' else 'unknown'}"
        )
    else:
        # Legacy Index v1: inform the user and provide the command to run offline indexing
        st.caption(
            f"⚠️ **Legacy Index v1** · Image-only vectors · "
            f"{meta['document_count']} docs · "
            "Run `python index_dataset.py` to upgrade to hybrid search"
        )


# ---------------------------------------------------------------------------
# Main Application Lifecycle
# ---------------------------------------------------------------------------
def main() -> None:
    """
    Main Streamlit application lifecycle and interface layout.

    Coordinates UI rendering, user input capture, session state management,
    search dispatching with latency benchmarking, and multimodal deep-dive analysis.
    """
    logger = get_logger()

    # Header section: Title and introductory explanation of multimodal RAG
    st.title("🎮 Search for a scene in the world of GTA!")
    st.markdown(
        "Describe any Grand Theft Auto scene in natural language. "
        "The system uses **CLIP multimodal embeddings** and **hybrid vector search** "
        "to find the closest matching scene instantly."
    )

    # 1. Load shared heavy resources (cached via get_resources)
    try:
        collection, data_set, model, logger = get_resources()
    except Exception as e:
        logger.exception(f"Failed to initialize application resources: {e}")
        st.error(f"Failed to initialize application resources: {e}")
        return

    # 2. Render database provenance badge directly below the header
    render_index_badge(collection)

    # 3. Initialize Streamlit session state keys to maintain state across reruns
    # In Streamlit, regular variables reset on each rerun. Storing results in
    # `st.session_state` preserves the active search output when interacting with other widgets.
    if "search_result" not in st.session_state:
        st.session_state["search_result"] = None
    if "search_query" not in st.session_state:
        st.session_state["search_query"] = ""
    if "live_analysis_result" not in st.session_state:
        st.session_state["live_analysis_result"] = None

    # 4. Search Input Bar
    # Pre-fills with the last executed query if available in session state
    query: str = st.text_input(
        "Describe the scene that you are looking for:",
        value=st.session_state.get("search_query", ""),
        placeholder="e.g. A sports car driving down a busy city boulevard at sunset",
        key="scene_query_input"
    )

    # 5. Primary Action Buttons (Search vs Terminate/Reset)
    # Uses a 3:1 column ratio for visual hierarchy
    btn_col1, btn_col2 = st.columns([3, 1], gap="medium")
    with btn_col1:
        search_clicked: bool = st.button("🔍 Search Scene", type="primary", width="stretch")
    with btn_col2:
        terminate_clicked: bool = st.button("⏹️ Terminate / Reset", type="secondary", width="stretch")

    # 6. Handle Search Termination / Reset
    # If the user clicks Terminate / Reset, clear all search state and force an immediate rerun
    if terminate_clicked:
        previous_query = st.session_state.get("search_query", query)
        logger.warning(f"Search operation terminated/reset by user. Active query was: '{previous_query}'")
        # Wipe session state values
        st.session_state["search_result"] = None
        st.session_state["search_query"] = ""
        st.session_state["live_analysis_result"] = None
        st.info("Search operation terminated and results cleared.")
        # Trigger immediate full rerun to restore the UI to an initial blank state
        st.rerun()

    # 7. Handle Search Execution
    if search_clicked:
        # Clear previous results immediately upon a new search submission
        st.session_state["search_result"] = None
        st.session_state["live_analysis_result"] = None

        # Sanitize query input (strip leading/trailing whitespace)
        clean_query = query.strip()
        if not clean_query:
            logger.warning("Search attempted with empty query")
            st.warning("Please enter a scene description before searching.")
        else:
            # Store sanitized query in session state for persistence
            st.session_state["search_query"] = clean_query
            logger.info(f"User started search for query: '{clean_query}'")

            # Execute search with a UI visual spinner
            with st.spinner("Searching vector database for matching scene..."):
                t_search_start = time.perf_counter()
                try:
                    # Execute Reciprocal Rank Fusion (RRF) search over ChromaDB & cache
                    image, original_text, description, provenance = get_search_result(
                        collection, data_set, clean_query, model, n_results=2
                    )
                    search_duration = time.perf_counter() - t_search_start

                    # Store output bundle in session state for persistent rendering
                    st.session_state["search_result"] = {
                        "image": image,
                        "original_text": original_text,
                        "description": description,
                        "provenance": provenance,
                        "duration": search_duration
                    }
                    logger.info(
                        f"Search successfully returned match in {search_duration * 1000:.1f}ms "
                        f"for query: '{clean_query}'"
                    )
                except Exception as e:
                    logger.exception(f"Error during vector search for query '{clean_query}': {e}")
                    st.error(f"Error during vector search: {e}")
                    st.session_state["search_result"] = None

    # 8. Render Search Results if present in session state
    res = st.session_state.get("search_result")
    if res:
        # Display search latency banner
        st.success(f"Scene found in {res['duration'] * 1000:.1f}ms!")

        # Unpack result components
        image = res["image"]
        original_text = res["original_text"]
        description = res["description"]
        provenance = res["provenance"]

        # Layout: display image and descriptions side-by-side (3:2 column ratio)
        col1, col2 = st.columns([3, 2], gap="large")

        # Column 1: Found GTA scene image display
        with col1:
            st.subheader("🖼️ Found Scene")
            # Render image with full container width
            st.image(image, caption="Retrieved GTA Scene", width="stretch")
            # Display cosine similarity distance if provided by provenance
            if provenance.get("distance") is not None:
                st.caption(f"Similarity distance: {provenance['distance']:.4f}")

        # Column 2: Textual scene metadata & LLM descriptions
        with col2:
            # Display the raw ground-truth caption from the original dataset
            st.subheader("📝 Original Caption")
            st.info(original_text)

            # Check if pre-computed Gemini description is available (v2 hybrid index)
            if description:
                st.subheader("🤖 Scene Description (Pre-indexed)")
                st.markdown(description)
                st.caption(
                    f"Generated by `{provenance.get('llm_model', 'unknown')}` "
                    f"on {provenance.get('indexed_at', 'unknown')[:10]}"
                )
            else:
                # Legacy index fallback: offer on-demand Gemini vision analysis
                st.subheader("🤖 Scene Description")
                google_api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
                if google_api_key:
                    with st.spinner("Generating scene analysis with Gemini..."):
                        try:
                            # Invoke live Gemini vision model on retrieved image
                            description_by_llm = get_image_description(image)
                            st.markdown(description_by_llm)
                        except Exception as e:
                            logger.warning(f"LLM description generation failed: {e}")
                            st.error(f"LLM description unavailable: {e}")
                else:
                    st.caption("*(Set `GOOGLE_API_KEY` in `.env` to enable Gemini scene descriptions)*")

        # 9. Interactive Multimodal Deep-Dive Section
        # Allows users to ask open-ended questions about the retrieved image
        st.divider()
        with st.expander("🔬 Ask Gemini for a live deep-dive analysis"):
            google_api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
            if not google_api_key:
                st.caption("*(Set `GOOGLE_API_KEY` in `.env` to enable live Gemini analysis)*")
            else:
                # Input question about the scene
                custom_prompt = st.text_input(
                    "Ask anything about this scene:",
                    placeholder="e.g. What vehicles are visible? Is this set during day or night?",
                    key="live_deep_dive_prompt"
                )
                live_col1, live_col2 = st.columns([3, 1])
                with live_col1:
                    analyze_btn = st.button("🧠 Analyze with Gemini", key="live_analysis", width="stretch")
                with live_col2:
                    clear_live_btn = st.button("Clear Analysis", key="clear_live_analysis", width="stretch")

                # Handle clearing of live analysis output
                if clear_live_btn:
                    logger.info("User cleared live deep-dive analysis results")
                    st.session_state["live_analysis_result"] = None
                    st.rerun()

                # Handle live Gemini analysis submission
                if analyze_btn:
                    clean_prompt = custom_prompt.strip()
                    if clean_prompt:
                        logger.info(f"User initiated live Gemini analysis with prompt: '{clean_prompt}'")
                        with st.spinner("Analyzing scene with Gemini..."):
                            t_llm_start = time.perf_counter()
                            try:
                                # Configure Google Generative AI with explicit REST transport
                                genai.configure(api_key=google_api_key, transport='rest')
                                # Select model (env override or default gemini-3.8-flash)
                                llm_model = os.environ.get("GEMINI_MODEL") or "gemini-3.8-flash"
                                vision_model = genai.GenerativeModel(
                                    llm_model,
                                    generation_config={"temperature": 0.2}  # Low temperature for factual precision
                                )
                                # Send both custom user prompt and scene image to Gemini
                                response = vision_model.generate_content(
                                    [clean_prompt, image],
                                    request_options={"timeout": 30}
                                )
                                llm_duration = time.perf_counter() - t_llm_start
                                logger.info(
                                    f"Live analysis completed successfully using '{llm_model}' "
                                    f"in {llm_duration:.2f}s ({len(response.text)} chars)"
                                )
                                # Persist analysis result in session state
                                st.session_state["live_analysis_result"] = response.text
                            except Exception as e:
                                logger.exception(f"Live analysis failed: {e}")
                                st.error(f"Live analysis failed: {e}")
                    else:
                        st.warning("Please enter a question about the scene.")

                # Render live analysis markdown response if present
                if st.session_state.get("live_analysis_result"):
                    st.markdown("### Analysis Result")
                    st.markdown(st.session_state["live_analysis_result"])


# Python standard execution entrypoint
if __name__ == "__main__":
    main()