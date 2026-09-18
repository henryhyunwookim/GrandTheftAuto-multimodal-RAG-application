"""
GTA Multimodal RAG Application - Root Compatibility Module.

Re-exports all core infrastructure functions, constants, and classes from
`src.rag_engine` for backward compatibility with root-level scripts and notebooks.
"""

from __future__ import annotations

from pathlib import Path
import sys

# Ensure repository root is on sys.path
REPO_ROOT: Path = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.rag_engine import *  # noqa: F401, F403
from src.rag_engine import (
    CURRENT_EMBEDDING_MODEL,
    CURRENT_LLM_MODEL,
    DEFAULT_DATASET,
    HYBRID_COLLECTION_NAME,
    INDEX_VERSION,
    LEGACY_COLLECTION_NAME,
    REPO_ROOT,
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
