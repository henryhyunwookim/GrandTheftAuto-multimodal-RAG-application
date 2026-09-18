"""
GTA Multimodal RAG Application Package.

Contains the core RAG engine, vector store management, multimodal CLIP embeddings,
and interactive Streamlit presentation layer.
"""

from __future__ import annotations

from src.rag_engine import (
    CURRENT_EMBEDDING_MODEL,
    CURRENT_LLM_MODEL,
    DEFAULT_DATASET,
    HYBRID_COLLECTION_NAME,
    INDEX_VERSION,
    LEGACY_COLLECTION_NAME,
    get_image_description,
    get_index_metadata,
    get_logger,
    get_search_result,
    initialization,
    load_data,
    set_directories,
)

__all__ = [
    "CURRENT_EMBEDDING_MODEL",
    "CURRENT_LLM_MODEL",
    "DEFAULT_DATASET",
    "HYBRID_COLLECTION_NAME",
    "INDEX_VERSION",
    "LEGACY_COLLECTION_NAME",
    "get_image_description",
    "get_index_metadata",
    "get_logger",
    "get_search_result",
    "initialization",
    "load_data",
    "set_directories",
]
