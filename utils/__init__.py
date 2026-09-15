"""
GTA Multimodal RAG Application - Utilities Package.

This package provides logging, dataset loading, model initialization,
multimodal embedding generation, vector storage management, and Gemini
vision integration.
"""

from utils.utils import (
    CURRENT_EMBEDDING_MODEL,
    CURRENT_LLM_MODEL,
    DEFAULT_DATASET,
    HYBRID_COLLECTION_NAME,
    INDEX_VERSION,
    LEGACY_COLLECTION_NAME,
    get_collection,
    get_embeddings,
    get_image_description,
    get_index_metadata,
    get_logger,
    get_search_result,
    initialization,
    load_data,
    set_directories,
    show_image,
)

__all__ = [
    "CURRENT_EMBEDDING_MODEL",
    "CURRENT_LLM_MODEL",
    "DEFAULT_DATASET",
    "HYBRID_COLLECTION_NAME",
    "INDEX_VERSION",
    "LEGACY_COLLECTION_NAME",
    "get_collection",
    "get_embeddings",
    "get_image_description",
    "get_index_metadata",
    "get_logger",
    "get_search_result",
    "initialization",
    "load_data",
    "set_directories",
    "show_image",
]
