"""
GTA Multimodal RAG Application - Core RAG Engine.

Module:
    rag_engine

Purpose:
    Provides core infrastructure functions for the GTA Multimodal RAG application,
    including persistent logging, dataset ingestion and caching, ChromaDB vector
    storage lifecycle, dual-stream Reciprocal Rank Fusion (RRF) search, and
    Google Gemini multimodal vision inference.

Usage:
    from rag_engine import (
        get_logger,
        initialization,
        get_search_result,
        get_image_description,
        get_index_metadata,
    )
    logger = get_logger("my_script")
    collection, data_set, model, logger = initialization(logger)

Prerequisites & Dependencies:
    - Python 3.10+
    - chromadb, sentence-transformers, google-generativeai, datasets, pillow, numpy, matplotlib
    - Google Gemini API key configured via GOOGLE_API_KEY environment variable (in .env)

Inputs & Outputs:
    - Inputs: Natural language query strings, PIL images, HuggingFace dataset items.
    - Outputs: Matched scene PIL images, captions, rich Gemini scene descriptions,
      and detailed model provenance metadata dictionaries.
"""

from __future__ import annotations

import logging
import os
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import chromadb
from chromadb.api.models.Collection import Collection
from datasets import load_dataset
from dotenv import load_dotenv
import google.generativeai as genai
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


# ===========================================================================
# Model Provenance & Storage Constants
# ===========================================================================
# Absolute path to repository root (parent of 'src/' directory)
REPO_ROOT: Path = Path(__file__).resolve().parent.parent

# SentenceTransformer CLIP model used for embedding visual scenes and text queries
CURRENT_EMBEDDING_MODEL: str = "sentence-transformers/clip-ViT-L-14"

# Default Gemini model used for multimodal visual scene descriptions and analysis
CURRENT_LLM_MODEL: str = os.environ.get("GEMINI_MODEL") or "gemini-3.8-flash"

# Semantic index version string representing dual-vector fusion architecture
INDEX_VERSION: str = "2.0.0"

# Target collection names in ChromaDB:
# - HYBRID_COLLECTION_NAME: v2.0 dual-vector index with fused CLIP embeddings & stored descriptions
# - LEGACY_COLLECTION_NAME: v1.0 image-only vector index for backward compatibility
HYBRID_COLLECTION_NAME: str = "gta_hybrid_v2"
LEGACY_COLLECTION_NAME: str = "image_vectors"

# Default HuggingFace GTA Image Captioning dataset repository
DEFAULT_DATASET: str = "vipulmaheshwari/GTA-Image-Captioning-Dataset"


# ===========================================================================
# Logging Infrastructure
# ===========================================================================
def get_logger(name: str = "gta_multimodal_rag") -> logging.Logger:
    """
    Get or create a configured logger with both file and console handlers.

    Ensures UTF-8 encoding for rotating daily logs under the `log/` directory,
    and prevents handler duplication across repeated calls or Streamlit reruns.

    Args:
        name (str): Identifier name for the logger hierarchy. Defaults to "gta_multimodal_rag".

    Returns:
        logging.Logger: Fully configured Python Logger instance.
    """
    # 1. Ensure log directory exists on disk (anchored to repository root)
    log_path = REPO_ROOT / "log"
    log_path.mkdir(parents=True, exist_ok=True)

    # 2. Build daily rotating log filename (e.g. log/20260915.log)
    cur_date = datetime.now().strftime("%Y%m%d")
    log_filename = log_path / f"{cur_date}.log"

    # 3. Retrieve or create the named logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # 4. Prevent duplicate handlers: in Streamlit, functions are re-invoked on every interaction;
    # attaching handlers repeatedly without this check causes duplicate log messages.
    if not logger.handlers:
        # Standardized log format with timestamp, level, logger name, and message
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # File Handler: set explicit UTF-8 encoding so emojis and multilingual characters are preserved
        file_handler = logging.FileHandler(str(log_filename), encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Stream Handler: outputs formatted log messages directly to console (stdout)
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger


# ===========================================================================
# Directory & Path Management
# ===========================================================================
def set_directories() -> Tuple[Path, Path]:
    """
    Ensure required application directories exist on the local filesystem.

    Creates `./data` and `./data/chroma` if they are missing.

    Returns:
        Tuple[Path, Path]: Tuple containing:
            - data_pickle_path: Full Path to `data/data_set.pkl`.
            - chroma_dir: Full Path to `data/chroma`.
    """
    # Data directory storing downloaded datasets, embedding caches, and ChromaDB (anchored to REPO_ROOT)
    data_dir = REPO_ROOT / "data"
    data_pickle_path = data_dir / "data_set.pkl"
    chroma_dir = data_dir / "chroma"

    # Iterate through paths and safely create parent directories if missing
    for directory in (data_dir, chroma_dir):
        if not directory.exists():
            directory.mkdir(parents=True, exist_ok=True)

    return data_pickle_path, chroma_dir


# ===========================================================================
# Dataset Ingestion & Caching
# ===========================================================================
def load_data(
    data_pickle_path: Union[Path, str],
    dataset: str = DEFAULT_DATASET
) -> Any:
    """
    Load GTA dataset from local pickle cache or download from HuggingFace.

    If the pickled dataset file does not exist, it downloads the dataset from
    HuggingFace, caches it to disk for subsequent launches, and returns it.

    Args:
        data_pickle_path (Union[Path, str]): Path where the dataset pickle is stored.
        dataset (str): HuggingFace repository identifier. Defaults to DEFAULT_DATASET.

    Returns:
        Any: HuggingFace DatasetDict or Dataset instance.
    """
    logger = get_logger()
    t_start = time.perf_counter()
    pickle_path = Path(data_pickle_path)

    # 1. Download from Hugging Face if no local pickle cache is found
    if not pickle_path.exists():
        logger.info(f"Dataset not found locally. Downloading '{dataset}' from HuggingFace...")
        # Stream or fetch dataset rows from Hugging Face Hub
        data_set = load_dataset(dataset)
        # Ensure parent directory exists before writing
        pickle_path.parent.mkdir(parents=True, exist_ok=True)
        # Save dataset to pickle file for fast offline access on future runs
        with open(pickle_path, 'wb') as outfile:
            pickle.dump(data_set, outfile)
        logger.info(f"Dataset downloaded and cached to {pickle_path} in {time.perf_counter() - t_start:.2f}s")
    else:
        # 2. Fast local load from disk
        logger.info(f"Loading cached dataset from {pickle_path}...")
        with open(pickle_path, 'rb') as infile:
            data_set = pickle.load(infile)
        total_items = len(data_set['train']) if 'train' in data_set else 'unknown'
        logger.info(f"Dataset loaded ({total_items} items) in {time.perf_counter() - t_start:.2f}s")

    return data_set


# ===========================================================================
# Embedding Generation
# ===========================================================================
def get_embeddings(
    data: List[Any],
    model: SentenceTransformer
) -> Tuple[List[str], List[List[float]]]:
    """
    Compute dense vector embeddings sequentially for a list of images or text inputs.

    Args:
        data (List[Any]): List of PIL images or text strings to encode.
        model (SentenceTransformer): Loaded SentenceTransformer/CLIP model instance.

    Returns:
        Tuple[List[str], List[List[float]]]: Tuple containing:
            - ids: Unique string identifiers formatted as "image <id>".
            - embeddings: List of float vector embeddings.
    """
    logger = get_logger()
    logger.info(f"Computing embeddings for {len(data)} items...")
    t_start = time.perf_counter()

    ids: List[str] = []
    embeddings: List[List[float]] = []

    # Iterate with progress bar, encoding each image/text sequentially
    for idx, item in tqdm(enumerate(data), total=len(data), desc="Encoding images"):
        ids.append(f"image {idx}")
        # Compute vector representation via SentenceTransformer
        embedding = model.encode(item)
        # Convert numpy array to standard python list for ChromaDB ingestion
        embeddings.append(embedding.tolist())

    logger.info(f"Computed {len(embeddings)} embeddings in {time.perf_counter() - t_start:.2f}s")
    return ids, embeddings


# ===========================================================================
# ChromaDB Collection Management
# ===========================================================================
def get_collection(
    chroma_dir: Union[Path, str],
    model: SentenceTransformer,
    collection_name: str = LEGACY_COLLECTION_NAME,
    data: Optional[List[Any]] = None
) -> Collection:
    """
    Get or create a ChromaDB collection, prioritizing the v2 hybrid collection.

    Attempts to load the v2 hybrid collection first (`gta_hybrid_v2`). If absent
    or empty, falls back to the specified collection (typically legacy `image_vectors`).

    Args:
        chroma_dir (Union[Path, str]): Persistent path to the ChromaDB database directory.
        model (SentenceTransformer): SentenceTransformer model used if embedding generation is needed.
        collection_name (str): Fallback collection name. Defaults to LEGACY_COLLECTION_NAME.
        data (Optional[List[Any]]): Optional image data list for population if collection is empty.

    Returns:
        Collection: ChromaDB collection object ready for query or document retrieval.
    """
    logger = get_logger()
    # Initialize persistent Chroma client pointing to local storage directory
    client = chromadb.PersistentClient(path=str(chroma_dir))

    # Priority 1: Check if the upgraded hybrid v2 collection exists and has documents
    try:
        hybrid_col = client.get_collection(name=HYBRID_COLLECTION_NAME)
        count = hybrid_col.count()
        if count > 0:
            meta = hybrid_col.metadata or {}
            logger.info(
                f"Using hybrid collection '{HYBRID_COLLECTION_NAME}' ({count} documents) | "
                f"Index v{meta.get('index_version', '2.0.0')} | "
                f"Embedding: {meta.get('embedding_model', 'unknown')} | "
                f"LLM: {meta.get('llm_model', 'unknown')}"
            )
            return hybrid_col
    except Exception as e:
        logger.debug(f"Hybrid collection check: {e}")

    # Priority 2: Fall back to legacy collection (e.g. 'image_vectors')
    logger.info(f"Accessing collection '{collection_name}'...")
    collection = client.get_or_create_collection(name=collection_name)

    # Populate embeddings if legacy collection is completely empty and seed data is provided
    if collection.count() == 0 and data is not None:
        logger.info(f"Collection '{collection_name}' is empty. Adding embeddings...")
        ids, embeddings = get_embeddings(data, model)
        collection.add(
            ids=ids,
            embeddings=embeddings
        )
        logger.info(f"Added {len(ids)} embeddings to '{collection_name}'")
    else:
        logger.info(f"Collection '{collection_name}' ready with {collection.count()} documents")

    return collection


def get_index_metadata(collection: Collection) -> Dict[str, Any]:
    """
    Inspect model provenance metadata from a ChromaDB collection.

    Args:
        collection (Collection): ChromaDB collection instance.

    Returns:
        Dict[str, Any]: Dictionary containing index provenance:
            - collection_name: Name of the collection.
            - document_count: Number of indexed vectors.
            - embedding_model: Name of the embedding model used.
            - llm_model: Name of the generative vision model used.
            - index_version: Semantic index version string (e.g., "2.0.0").
            - image_weight: Weight allocated to image representations.
            - text_weight: Weight allocated to text representations.
            - created_at: ISO timestamp of index creation.
            - is_hybrid: Boolean indicating if collection supports dual-vector hybrid search.
    """
    meta = collection.metadata or {}
    return {
        "collection_name": collection.name,
        "document_count": collection.count(),
        "embedding_model": meta.get("embedding_model", "unknown"),
        "llm_model": meta.get("llm_model", "unknown"),
        "index_version": meta.get("index_version", "1.0.0 (legacy)"),
        "image_weight": meta.get("image_weight", "unknown"),
        "text_weight": meta.get("text_weight", "unknown"),
        "created_at": meta.get("created_at", "unknown"),
        # Detect hybrid status based on version prefix "2." or collection suffix "_v2"
        "is_hybrid": meta.get("index_version", "").startswith("2") or collection.name.endswith("_v2"),
    }


# ===========================================================================
# Multimodal Search via Reciprocal Rank Fusion (RRF)
# ===========================================================================
def get_search_result(
    collection: Collection,
    data_set: Any,
    query: str,
    model: SentenceTransformer,
    n_results: int = 2
) -> Tuple[Any, str, str, Dict[str, Any]]:
    """
    Search the multimodal collection using Reciprocal Rank Fusion (RRF).

    Combines two retrieval channels:
    1. Text-to-Image cross-modal similarity (visual appearance via CLIP).
    2. Text-to-Text semantic similarity (scene description/caption via CLIP).

    Reciprocal Rank Fusion prevents single-vector modality gap distortion
    and centroid collapse, yielding highly robust multimodal rankings.

    Args:
        collection (Collection): Active ChromaDB collection.
        data_set (Any): Hugging Face dataset containing images and text captions.
        query (str): Natural language scene description from the user.
        model (SentenceTransformer): Loaded SentenceTransformer/CLIP model instance.
        n_results (int): Number of top candidate results to evaluate. Defaults to 2.

    Returns:
        Tuple[Any, str, str, Dict[str, Any]]: Tuple containing:
            - image: Best matching PIL Image.
            - original_caption: Original dataset caption for the image.
            - description: Pre-computed or stored rich scene description.
            - provenance: Provenance dictionary with model names, version, and scores.
    """
    logger = get_logger()
    index_meta = get_index_metadata(collection)

    logger.info(f"Initiating vector search for query: '{query}' [Collection: {collection.name}]")

    # 1. Encode query text with SentenceTransformer CLIP model
    # normalize_embeddings=True ensures output vector has unit norm (||q|| = 1)
    t_enc_start = time.perf_counter()
    query_embedding = model.encode([query], normalize_embeddings=True)[0]
    enc_duration = time.perf_counter() - t_enc_start
    logger.info(f"Query text encoded in {enc_duration * 1000:.1f}ms")

    t_search_start = time.perf_counter()

    # 2. Check for cached precomputed normalized embeddings on disk
    # This enables dual-stream RRF rank calculation without querying two separate collections
    img_cache_path = REPO_ROOT / "data" / "image_embeddings_cache.npy"
    cap_cache_path = REPO_ROOT / "data" / "caption_embeddings_cache.npy"

    if img_cache_path.exists() and cap_cache_path.exists():
        # Load visual embedding matrix (N x D)
        img_embs = np.load(img_cache_path)
        # Ensure L2 normalization of visual vectors
        img_norms = np.linalg.norm(img_embs, axis=1, keepdims=True)
        img_norms[img_norms == 0] = 1.0
        img_embs = img_embs / img_norms

        # Load caption/text embedding matrix (N x D)
        cap_embs = np.load(cap_cache_path)
        # Ensure L2 normalization of caption vectors
        cap_norms = np.linalg.norm(cap_embs, axis=1, keepdims=True)
        cap_norms[cap_norms == 0] = 1.0
        cap_embs = cap_embs / cap_norms

        # Channel 1: Cosine similarity against Image embeddings (CLIP cross-modal visual matching)
        sim_img = np.dot(img_embs, query_embedding)

        # Channel 2: Cosine similarity against Caption embeddings (CLIP text-to-text semantic matching)
        sim_cap = np.dot(cap_embs, query_embedding)

        # 3. Reciprocal Rank Fusion (RRF with standard smoothing factor k=60)
        # Double argsort produces zero-indexed rank positions (rank 0 = top match)
        rank_img = np.argsort(np.argsort(-sim_img))
        rank_cap = np.argsort(np.argsort(-sim_cap))
        # RRF formula: Score(d) = sum( 1 / (k + rank_i(d)) )
        rrf_scores = 1.0 / (60 + rank_img) + 1.0 / (60 + rank_cap)

        # Identify top candidate indices
        top_indices = np.argsort(-rrf_scores)[:n_results]
        top_idx = int(top_indices[0])

        # Compute cosine distance metric for display (1.0 - max_similarity)
        sim_distance = float(1.0 - max(sim_img[top_idx], sim_cap[top_idx]))
        search_duration = time.perf_counter() - t_search_start
        logger.info(f"Dual-stream RRF search completed in {search_duration * 1000:.1f}ms")

        # 4. Fetch stored rich scene description from ChromaDB for top match
        try:
            doc_res = collection.get(ids=[f"img_{top_idx:04d}"], include=["documents", "metadatas"])
            description = doc_res["documents"][0] if doc_res.get("documents") and doc_res["documents"] else ""
            meta = doc_res["metadatas"][0] if doc_res.get("metadatas") and doc_res["metadatas"] else {}
        except Exception:
            description = ""
            meta = {}

        # 5. Fetch image and caption by index directly from HuggingFace dataset (O(1) random access)
        t_fetch = time.perf_counter()
        row = data_set['train'][top_idx]
        image = row['image']
        original_caption = meta.get("original_caption", row['text'])
        fetch_duration = time.perf_counter() - t_fetch

        # Bundle model provenance and scoring telemetry
        provenance = {
            "llm_model": meta.get("llm_model", index_meta.get("llm_model", "gemini-3.8-flash")),
            "embedding_model": meta.get("embedding_model", index_meta.get("embedding_model", CURRENT_EMBEDDING_MODEL)),
            "index_version": meta.get("index_version", INDEX_VERSION),
            "indexed_at": meta.get("indexed_at", "unknown"),
            "distance": sim_distance,
            "rrf_score": float(rrf_scores[top_idx]),
        }

        logger.info(
            f"RRF Match: image_id={top_idx} | dist={sim_distance:.4f} | rrf={rrf_scores[top_idx]:.5f} | "
            f"caption='{original_caption[:60]}...' | row_fetch={fetch_duration * 1000:.1f}ms"
        )
        return image, original_caption, description, provenance
    else:
        # 6. Fallback path: Direct vector query against the active ChromaDB collection
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        query_duration = time.perf_counter() - t_search_start
        logger.info(f"Fallback ChromaDB search completed in {query_duration * 1000:.1f}ms")

        top_metadata = results['metadatas'][0][0] if results['metadatas'] else {}
        top_document = results['documents'][0][0] if results['documents'] else ""
        image_id = top_metadata.get("image_id", 0)
        distance = results['distances'][0][0] if results['distances'] else None

        # Fetch matched row from dataset
        row = data_set['train'][image_id]
        image = row['image']
        original_caption = top_metadata.get("original_caption", row['text'])

        provenance = {
            "llm_model": top_metadata.get("llm_model", "unknown"),
            "embedding_model": top_metadata.get("embedding_model", "unknown"),
            "index_version": top_metadata.get("index_version", "unknown"),
            "indexed_at": top_metadata.get("indexed_at", "unknown"),
            "distance": distance,
        }

        return image, original_caption, top_document, provenance


# ===========================================================================
# Visual Display Helper
# ===========================================================================
def show_image(image: Any, text: str, query: str) -> None:
    """
    Display an image in a matplotlib interactive figure with text metadata.

    Useful for interactive terminal debugging or Jupyter notebook exploration.

    Args:
        image (Any): PIL Image or numpy array to display.
        text (str): Image caption or description.
        query (str): User search query that matched this image.
    """
    plt.ion()  # Turn on matplotlib interactive mode
    plt.axis("off")  # Suppress pixel coordinate axes
    plt.imshow(image)
    plt.show()
    print(f"User query: {query}")
    print(f"Original description: {text}\n")


# ===========================================================================
# Google Gemini Multimodal Vision Inference
# ===========================================================================
def get_image_description(
    image: Any,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None
) -> str:
    """
    Generate an objective, detailed scene description of an image using Google Gemini.

    Evaluates candidate models in order: explicitly passed model_name -> GEMINI_MODEL
    environment variable -> gemini-3.8-flash -> gemini-2.0-flash -> gemini-1.5-flash.
    Fails fast upon fatal authentication or quota errors.

    Args:
        image (Any): Input PIL Image to describe.
        api_key (Optional[str]): Gemini API key. Defaults to GOOGLE_API_KEY from environment.
        model_name (Optional[str]): Gemini model name. Defaults to GEMINI_MODEL env var or gemini-3.8-flash.

    Returns:
        str: Generated textual scene description.

    Raises:
        ValueError: If no Google API key is provided or configured.
        Exception: If all candidate Gemini models fail during inference.
    """
    logger = get_logger()
    load_dotenv(override=True)
    # Resolve API key from arguments or environment
    api_key = api_key or os.environ.get('GOOGLE_API_KEY')
    if not api_key:
        logger.error("Google API Key not provided. Cannot generate description.")
        raise ValueError("Google API Key not provided. Set GOOGLE_API_KEY in .env or pass it into the app.")

    # Configure Google GenAI SDK with explicit REST transport
    genai.configure(api_key=api_key, transport='rest')

    # Hard-constrained prompt for factual, objective visual description
    prompt = """Describe what you explicitly see in the given image in detail.
Begin your description with "In this image," or "This image is about," to provide context.
Your response should be a hard description of the given image without any thoughts or suggestions."""

    # Model resolution waterfall: explicit argument -> GEMINI_MODEL env var -> gemini-3.8-flash
    active_model = model_name or os.environ.get('GEMINI_MODEL') or "gemini-3.8-flash"

    # Build fallback candidate sequence
    candidate_models: List[str] = [active_model]
    for fallback in ["gemini-3.8-flash", "gemini-2.0-flash", "gemini-1.5-flash"]:
        if fallback not in candidate_models:
            candidate_models.append(fallback)

    last_error: Optional[Exception] = None

    # Attempt inference sequentially across candidate models
    for m_name in candidate_models:
        logger.info(f"Requesting scene description from Gemini model: {m_name}")
        t_gen_start = time.perf_counter()
        try:
            # Temperature 0.0 guarantees deterministic, hallucination-minimized output
            vision_model = genai.GenerativeModel(
                m_name,
                generation_config={"temperature": 0.0}
            )
            # Submit both prompt and raw image input with 15s timeout
            response = vision_model.generate_content([prompt, image], request_options={"timeout": 15})
            gen_duration = time.perf_counter() - t_gen_start
            logger.info(
                f"Gemini description generated successfully with '{m_name}' in "
                f"{gen_duration:.2f}s ({len(response.text)} chars)"
            )
            return response.text
        except Exception as e:
            gen_duration = time.perf_counter() - t_gen_start
            last_error = e
            err_msg = str(e)
            logger.warning(f"Generation attempt failed on '{m_name}' after {gen_duration:.2f}s: {err_msg}")
            # Fail fast on invalid key or authentication error so we don't freeze the UI retrying invalid credentials
            if "API_KEY_INVALID" in err_msg or "API key not valid" in err_msg or "invalid_grant" in err_msg:
                logger.error(f"Fatal auth error encountered with API key: {err_msg}")
                raise e
            continue

    logger.error(f"All candidate Gemini models failed. Last error: {last_error}")
    if last_error:
        raise last_error
    raise RuntimeError("Failed to generate image description: no response returned.")


# ===========================================================================
# Application Resource Initialization Orchestration
# ===========================================================================
def initialization(
    logger: logging.Logger
) -> Tuple[Collection, Any, SentenceTransformer, logging.Logger]:
    """
    Initialize shared application resources required for multimodal search.

    Sets up required directories, loads the GTA dataset, instantiates the CLIP
    SentenceTransformer embedding model, and establishes the ChromaDB collection.

    Args:
        logger (logging.Logger): Initialized logger instance.

    Returns:
        Tuple[Collection, Any, SentenceTransformer, logging.Logger]: Initialized resources:
            - collection: ChromaDB collection (v2 hybrid or legacy).
            - data_set: HuggingFace GTA dataset with train split.
            - model: Loaded SentenceTransformer CLIP model.
            - logger: Application logger.
    """
    logger.info("Initializing application resources...")
    logger.info("-" * 55)

    # Step 1: Establish data and vector database folder structure
    logger.info("Setting up directories...")
    data_pickle_path, chroma_dir = set_directories()

    # Step 2: Load or download the HuggingFace GTA captioning dataset
    logger.info("Loading dataset...")
    data_set = load_data(data_pickle_path)

    # Step 3: Instantiate the CLIP neural model
    logger.info(f"Loading CLIP model: {CURRENT_EMBEDDING_MODEL}...")
    t_clip_start = time.perf_counter()
    model = SentenceTransformer(CURRENT_EMBEDDING_MODEL)
    logger.info(f"CLIP model loaded in {time.perf_counter() - t_clip_start:.2f}s")

    # Step 4: Connect to ChromaDB persistent collection
    logger.info("Initializing vector storage collection...")
    collection = get_collection(chroma_dir, model, collection_name=LEGACY_COLLECTION_NAME, data=None)

    logger.info("-" * 55)
    logger.info("Initialization completed successfully. Ready for search.")

    return collection, data_set, model, logger