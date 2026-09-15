"""
Offline Indexing Pipeline for GTA Multimodal RAG Search.

File:
    index_dataset.py

Purpose:
    Generates an optimized ChromaDB vector database collection for the GTA dataset,
    combining CLIP dual-vector representations (image visual features + rich textual
    scene descriptions generated via Google Gemini) into fused, normalized vectors.
    Supports checkpointing, batch embedding inference, and model provenance tracking.

Usage / CLI Invocation:
    python index_dataset.py
    python index_dataset.py --force-reindex
    python index_dataset.py --sample-size 10
    python index_dataset.py --embedding-model sentence-transformers/clip-ViT-L-14
    python index_dataset.py --llm-model gemini-3.8-flash
    python index_dataset.py --skip-llm

Prerequisites & Dependencies:
    - Python 3.10+
    - chromadb >= 0.5.5
    - sentence-transformers >= 3.0.0
    - google-generativeai >= 0.8.0
    - datasets >= 2.21.0
    - pillow >= 10.4.0, numpy >= 1.26.0, tqdm >= 4.66.5
    - GOOGLE_API_KEY environment variable configured in `.env` (unless --skip-llm is specified).

Inputs & Outputs:
    - Inputs: HuggingFace GTA Image Captioning Dataset, command-line arguments.
    - Outputs:
      - `data/chroma/`: Persistent ChromaDB collection (`gta_hybrid_v2`).
      - `data/descriptions_checkpoint.json`: Checkpointed scene descriptions from Gemini.
      - `data/image_embeddings_cache.npy`: Cached normalized image vector embeddings.
      - `data/caption_embeddings_cache.npy`: Cached normalized caption vector embeddings.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import chromadb
from dotenv import load_dotenv
from datasets import load_dataset
import google.generativeai as genai
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from rag_engine import get_logger

# Initialize pipeline logger
logger = get_logger("index_dataset")

# ===========================================================================
# Pipeline Hyperparameters & Storage Constants
# ===========================================================================
# Semantic index version string recorded in collection metadata
INDEX_VERSION: str = "2.0.0"

# Default visual and textual feature extractor model
DEFAULT_EMBEDDING_MODEL: str = "sentence-transformers/clip-ViT-L-14"

# Default Gemini model used for multimodal visual scene descriptions
DEFAULT_LLM_MODEL: str = "gemini-3.8-flash"

# Target hybrid ChromaDB collection name
DEFAULT_COLLECTION_NAME: str = "gta_hybrid_v2"

# HuggingFace GTA Image Captioning dataset repository
DEFAULT_DATASET_NAME: str = "vipulmaheshwari/GTA-Image-Captioning-Dataset"

# Dual-Vector Fusion Weights:
# In multimodal scene search, users often describe visual scene elements ("sports car at sunset").
# Allocating 60% weight to visual features and 40% to textual scene descriptions strikes an optimal
# balance between cross-modal visual matching and semantic text matching.
IMAGE_WEIGHT: float = 0.6  # Weight for image vector in the fused embedding
TEXT_WEIGHT: float = 0.4   # Weight for text vector in the fused embedding

# Checkpoint and cache file locations
DESCRIPTION_CHECKPOINT_FILE: str = "data/descriptions_checkpoint.json"
IMAGE_CACHE_FILE: str = "data/image_embeddings_cache.npy"
CAPTION_CACHE_FILE: str = "data/caption_embeddings_cache.npy"


# ===========================================================================
# CLI Argument Parsing
# ===========================================================================
def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the offline dataset indexing pipeline.

    Allows configuring embedding models, LLM generators, sample sizes,
    fusion weights, and collection names from the terminal.

    Returns:
        argparse.Namespace: Parsed CLI options and configuration flags.
    """
    parser = argparse.ArgumentParser(
        description="Index GTA dataset into ChromaDB with hybrid fused embeddings."
    )
    # SentenceTransformer CLIP model used for embeddings
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help=f"SentenceTransformer model for embeddings (default: {DEFAULT_EMBEDDING_MODEL})"
    )
    # Gemini model used for generating scene descriptions
    parser.add_argument(
        "--llm-model",
        default=DEFAULT_LLM_MODEL,
        help=f"Gemini model for generating scene descriptions (default: {DEFAULT_LLM_MODEL})"
    )
    # Target ChromaDB collection name
    parser.add_argument(
        "--collection-name",
        default=DEFAULT_COLLECTION_NAME,
        help=f"ChromaDB collection name (default: {DEFAULT_COLLECTION_NAME})"
    )
    # Restrict indexing to first N samples (useful for rapid testing/debugging)
    parser.add_argument(
        "--sample-size",
        type=int,
        default=0,
        help="Only index the first N images (0 = index entire dataset)"
    )
    # Force complete re-indexing by deleting existing ChromaDB collection
    parser.add_argument(
        "--force-reindex",
        action="store_true",
        help="Delete existing collection and re-index from scratch"
    )
    # Bypass Gemini API calls completely and use original captions
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip Gemini description generation (use original dataset captions only)"
    )
    # Relative visual weighting in fused embeddings (0.0 to 1.0)
    parser.add_argument(
        "--image-weight",
        type=float,
        default=IMAGE_WEIGHT,
        help=f"Weight for image vector in fused embedding (default: {IMAGE_WEIGHT})"
    )
    return parser.parse_args()


# ===========================================================================
# Dataset Loading & Serialization
# ===========================================================================
def load_or_download_dataset(
    data_pickle_path: Path,
    dataset_name: str = DEFAULT_DATASET_NAME
) -> Any:
    """
    Load GTA dataset from local pickle cache or download directly from HuggingFace.

    Caches the downloaded dataset locally to disk (`data/data_set.pkl`) to avoid
    repeated multi-gigabyte downloads on subsequent pipeline executions.

    Args:
        data_pickle_path (Path): Path to the serialized pickle file on disk.
        dataset_name (str): Hugging Face repository identifier.

    Returns:
        Any: Loaded Hugging Face dataset dictionary.
    """
    # 1. Check if local pickle cache already exists on disk
    if data_pickle_path.exists():
        print(f"Loading cached dataset from {data_pickle_path}...")
        with open(data_pickle_path, 'rb') as f:
            return pickle.load(f)
    else:
        # 2. Download from HuggingFace Hub
        print(f"Downloading dataset '{dataset_name}' from HuggingFace...")
        data_set = load_dataset(dataset_name)
        # Ensure target data directory exists
        data_pickle_path.parent.mkdir(parents=True, exist_ok=True)
        # Serialize dataset object to pickle file
        with open(data_pickle_path, 'wb') as f:
            pickle.dump(data_set, f)
        return data_set


# ===========================================================================
# Gemini Scene Description Generation with Checkpointing
# ===========================================================================
def generate_descriptions(
    images: List[Any],
    captions: List[str],
    llm_model: str,
    checkpoint_path: str
) -> List[str]:
    """
    Generate detailed scene descriptions using Google Gemini, with automatic checkpointing.

    Checks for previously saved descriptions at `checkpoint_path` and resumes from
    the latest checkpoint, avoiding redundant API costs and time on repeated executions.
    Features automated HTTP 429 rate limit backoff and fallback to dataset captions.

    Args:
        images (List[Any]): List of PIL Images to describe.
        captions (List[str]): Fallback captions from the original dataset.
        llm_model (str): Gemini model identifier (e.g., 'gemini-3.8-flash').
        checkpoint_path (str): Filepath to read/write json checkpoint progress.

    Returns:
        List[str]: List of detailed scene descriptions corresponding to the images.
    """
    # Load environment variables to retrieve GOOGLE_API_KEY
    load_dotenv(override=True)
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("WARNING: GOOGLE_API_KEY not set. Using original captions as descriptions.")
        return list(captions)

    # Configure Google Generative AI SDK with explicit REST transport
    genai.configure(api_key=api_key, transport='rest')

    # 1. Load existing checkpoint progress if present
    checkpoint: Dict[str, str] = {}
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            checkpoint = json.load(f)
        print(f"Loaded {len(checkpoint)} existing descriptions from checkpoint.")

    # 2. Hard-constrained prompt for factual, objective visual description
    prompt = (
        "Describe what you explicitly see in the given image in detail.\n"
        "Begin your description with \"In this image,\" or \"This image is about,\" to provide context.\n"
        "Your response should be a hard description of the given image without any thoughts or suggestions."
    )

    descriptions: List[str] = [""] * len(images)
    generated_count = 0
    error_count = 0

    # 3. Iterate over images with progress bar
    for i in tqdm(range(len(images)), desc="Generating LLM descriptions"):
        key = str(i)

        # Skip API generation if description is already cached in checkpoint
        if key in checkpoint and checkpoint[key]:
            descriptions[i] = checkpoint[key]
            continue

        # Generate new description via Gemini
        try:
            vision_model = genai.GenerativeModel(
                llm_model,
                generation_config={"temperature": 0.0}  # Deterministic temperature
            )
            response = vision_model.generate_content(
                [prompt, images[i]],
                request_options={"timeout": 30}
            )
            descriptions[i] = response.text
            checkpoint[key] = descriptions[i]
            generated_count += 1

            # Save checkpoint every 5 images to prevent work loss on interruption
            if generated_count % 5 == 0:
                with open(checkpoint_path, 'w', encoding='utf-8') as f:
                    json.dump(checkpoint, f, ensure_ascii=False, indent=2)

            # Polite rate-limiting sleep between requests
            time.sleep(0.5)

        except Exception as e:
            err_msg = str(e)
            # Detect fatal authentication errors
            if "API_KEY_INVALID" in err_msg or "API key not valid" in err_msg:
                print("\nERROR: Invalid API key. Stopping description generation.")
                print("Falling back to original captions for remaining images.")
                for j in range(i, len(images)):
                    if str(j) not in checkpoint:
                        descriptions[j] = captions[j]
                break

            # Handle rate limiting / quota exhaustion (HTTP 429) with exponential backoff
            if "429" in err_msg or "ResourceExhausted" in err_msg or "quota" in err_msg.lower():
                print(f"\nRate limit encountered on image {i}. Waiting 10 seconds before retry...")
                time.sleep(10.0)
                try:
                    response = vision_model.generate_content(
                        [prompt, images[i]],
                        request_options={"timeout": 30}
                    )
                    descriptions[i] = response.text
                    checkpoint[key] = descriptions[i]
                    generated_count += 1
                    continue
                except Exception as retry_err:
                    print(f"Retry failed for image {i}: {retry_err}")

            # Fallback on other individual image errors
            print(f"\nWARNING: Failed to generate description for image {i}: {e}")
            descriptions[i] = captions[i]  # Fallback to original caption
            error_count += 1

    # 4. Final checkpoint flush to disk
    with open(checkpoint_path, 'w', encoding='utf-8') as f:
        json.dump(checkpoint, f, ensure_ascii=False, indent=2)

    print(
        f"Descriptions: {generated_count} generated, "
        f"{len(checkpoint) - generated_count} from checkpoint, "
        f"{error_count} errors (fell back to captions)"
    )
    return descriptions


# ===========================================================================
# CLIP Tokenizer Truncation Patch
# ===========================================================================
def enable_clip_truncation(model: SentenceTransformer) -> None:
    """
    Ensure CLIP tokenizer truncates long text to model_max_length (77 tokens).

    Background:
        OpenAI CLIP models use a fixed context length of 77 tokens. When fed rich,
        multi-sentence descriptions produced by Gemini, the default SentenceTransformer
        tokenizer can raise: `RuntimeError: The size of tensor a (X) must match the size of tensor b (77)`.
        This patch safely intercepts the tokenizer invocation to ensure `truncation=True, max_length=77`.

    Args:
        model (SentenceTransformer): SentenceTransformer model to patch.
    """
    first_module = model._first_module()
    # Check if the underlying module exposes a Hugging Face processor and tokenizer
    if hasattr(first_module, 'processor') and hasattr(first_module.processor, 'tokenizer'):
        def safe_tokenize(texts: List[Any], padding: bool = True) -> Dict[str, Any]:
            images: List[Any] = []
            texts_values: List[str] = []
            image_text_info: List[int] = []

            # Separate multimodal batch into images and text strings
            for data in texts:
                if isinstance(data, Image.Image):
                    images.append(data)
                    image_text_info.append(0)  # 0 indicates visual modality
                else:
                    texts_values.append(str(data))
                    image_text_info.append(1)  # 1 indicates textual modality

            encoding: Dict[str, Any] = {}
            # Tokenize text inputs with explicit truncation to 77 tokens
            if len(texts_values):
                encoding = first_module.processor.tokenizer(
                    texts_values,
                    return_tensors="pt",
                    padding=padding,
                    truncation=True,  # Truncate sequences exceeding max_length
                    max_length=77     # Hard limit to match CLIP positional embeddings
                )

            # Process pixel values for image inputs
            if len(images):
                image_features = first_module.processor.image_processor(images, return_tensors="pt")
                encoding["pixel_values"] = image_features.pixel_values

            encoding["image_text_info"] = image_text_info
            return encoding

        # Replace tokenize method on the active module
        first_module.tokenize = safe_tokenize


# ===========================================================================
# Dual-Vector Fused Embedding Generation
# ===========================================================================
def compute_fused_embeddings(
    images: List[Any],
    descriptions: List[str],
    model: SentenceTransformer,
    image_weight: float,
    chroma_client: Optional[chromadb.PersistentClient] = None
) -> Tuple[List[str], List[List[float]]]:
    """
    Compute fused dual-vector embeddings combining image visual and text representations.

    Reuses existing precomputed image embeddings when available (from disk cache or
    legacy ChromaDB collection). Employs batch processing for GPU/CPU acceleration.

    Fusion Formula:
        fused = L2_Normalize(image_weight * image_vec + (1 - image_weight) * text_vec)

    Args:
        images (List[Any]): List of PIL Images.
        descriptions (List[str]): List of scene descriptions.
        model (SentenceTransformer): CLIP SentenceTransformer model.
        image_weight (float): Relative weight for visual vector (0.0 - 1.0).
        chroma_client (Optional[chromadb.PersistentClient]): Optional client to reuse legacy vectors.

    Returns:
        Tuple[List[str], List[List[float]]]: Tuple containing:
            - ids: String identifiers formatted as "img_0000".
            - fused_normalized: Fused L2-normalized float embeddings.
    """
    # Calculate corresponding text weight such that weights sum to 1.0
    text_weight = 1.0 - image_weight
    print(f"Computing embeddings (image_weight={image_weight}, text_weight={text_weight})...")

    img_embeddings: Optional[np.ndarray] = None
    num_images = len(images)

    # 1. Check disk cache for precomputed image embeddings
    if os.path.exists(IMAGE_CACHE_FILE):
        try:
            cached = np.load(IMAGE_CACHE_FILE)
            if len(cached) >= num_images:
                print(f"Loaded {num_images} precomputed image embeddings from disk cache ({IMAGE_CACHE_FILE})")
                img_embeddings = cached[:num_images]
        except Exception as e:
            print(f"Warning: Failed to load image cache: {e}")

    # 2. Check legacy Chroma collection if disk cache was not found
    if img_embeddings is None and chroma_client is not None:
        try:
            legacy_col = chroma_client.get_collection("image_vectors")
            if legacy_col.count() >= num_images:
                print(f"Extracting {num_images} precomputed image embeddings from existing 'image_vectors' collection...")
                expected_ids = [f"image {i}" for i in range(num_images)]
                all_vecs: List[List[float]] = []
                # Fetch in batches of 200 to avoid memory spikes
                for b_start in range(0, num_images, 200):
                    b_end = min(b_start + 200, num_images)
                    batch_res = legacy_col.get(ids=expected_ids[b_start:b_end], include=['embeddings'])
                    id_to_emb = {eid: emb for eid, emb in zip(batch_res['ids'], batch_res['embeddings'])}
                    for eid in expected_ids[b_start:b_end]:
                        all_vecs.append(id_to_emb[eid])
                img_embeddings = np.array(all_vecs)
                # Persist extracted embeddings to disk cache
                os.makedirs(os.path.dirname(IMAGE_CACHE_FILE), exist_ok=True)
                np.save(IMAGE_CACHE_FILE, img_embeddings)
                print(f"Saved extracted image embeddings to disk cache ({IMAGE_CACHE_FILE})")
        except Exception as e:
            print(f"Could not load embeddings from legacy collection: {e}")

    # 3. Compute image embeddings from scratch if not cached anywhere
    if img_embeddings is None:
        print("Encoding images in batches (batch_size=32)...")
        img_embeddings = model.encode(
            images,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True
        )
        # Save to disk cache for fast reuse
        os.makedirs(os.path.dirname(IMAGE_CACHE_FILE), exist_ok=True)
        np.save(IMAGE_CACHE_FILE, img_embeddings)
        print(f"Saved image embeddings to disk cache ({IMAGE_CACHE_FILE})")

    # 4. Enforce unit L2 normalization on visual embeddings
    img_norms = np.linalg.norm(img_embeddings, axis=1, keepdims=True)
    img_norms[img_norms == 0] = 1.0
    img_embeddings = img_embeddings / img_norms

    # 5. Enable safe tokenizer truncation for descriptions (CLIP 77 token boundary)
    enable_clip_truncation(model)

    # 6. Encode textual scene descriptions in batches
    print("Encoding descriptions in batches (batch_size=64)...")
    txt_embeddings = model.encode(
        descriptions,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    # Also persist caption embeddings cache for dual-stream search
    try:
        os.makedirs(os.path.dirname(CAPTION_CACHE_FILE), exist_ok=True)
        np.save(CAPTION_CACHE_FILE, txt_embeddings)
    except Exception as e:
        print(f"Warning: Could not save caption cache: {e}")

    # 7. Compute Fused Dual-Vector: Weighted Linear Combination followed by L2 Normalization
    print("Fusing dual vectors...")
    # Weighted vector addition: v_fused = w_i * v_img + w_t * v_txt
    fused_matrix = image_weight * np.array(img_embeddings) + text_weight * np.array(txt_embeddings)
    # L2 normalization to unit hypersphere
    norms = np.linalg.norm(fused_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    fused_normalized = fused_matrix / norms

    # Generate standardized zero-padded document IDs (e.g. img_0000, img_0001)
    ids = [f"img_{i:04d}" for i in range(len(images))]
    return ids, fused_normalized.tolist()


# ===========================================================================
# End-to-End Offline Indexing Pipeline
# ===========================================================================
def build_index(args: argparse.Namespace) -> None:
    """
    Execute end-to-end dataset indexing pipeline.

    Coordinates dataset loading, scene description generation, fused vector
    computation, and batch upserts into the ChromaDB collection with provenance.

    Args:
        args (argparse.Namespace): Configured pipeline options from CLI.
    """
    curr_dir = Path(os.getcwd())
    data_pickle_path = curr_dir / 'data' / 'data_set.pkl'
    chroma_dir = curr_dir / 'data' / 'chroma'
    chroma_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = str(curr_dir / DESCRIPTION_CHECKPOINT_FILE)

    # Initialize persistent ChromaDB client
    client = chromadb.PersistentClient(path=str(chroma_dir))

    # STEP 1: Load GTA dataset from cache or download from HuggingFace
    print("=" * 60)
    print("STEP 1: Loading dataset")
    print("=" * 60)
    data_set = load_or_download_dataset(data_pickle_path)
    train = data_set['train']

    total_images = len(train)
    # Apply sample size constraint if specified (e.g. --sample-size 10)
    if args.sample_size > 0:
        total_images = min(args.sample_size, total_images)
    print(f"Dataset loaded: {len(train)} images total, indexing {total_images}")

    # Slice image and caption lists
    images = train['image'][:total_images]
    captions = train['text'][:total_images]

    # STEP 2: Generate or load rich descriptions via Gemini
    print("\n" + "=" * 60)
    print("STEP 2: Generating rich scene descriptions")
    print("=" * 60)
    if args.skip_llm:
        print("Skipping LLM descriptions (using original captions only)")
        descriptions = list(captions)
    else:
        descriptions = generate_descriptions(images, captions, args.llm_model, checkpoint_path)

    # STEP 3: Load CLIP model and compute fused dual-vector embeddings
    print("\n" + "=" * 60)
    print("STEP 3: Computing fused embeddings")
    print("=" * 60)
    print(f"Loading embedding model: {args.embedding_model}")
    model = SentenceTransformer(args.embedding_model)
    ids, fused_embeddings = compute_fused_embeddings(
        images,
        descriptions,
        model,
        args.image_weight,
        chroma_client=client
    )

    # STEP 4: Persist collection and documents into ChromaDB
    print("\n" + "=" * 60)
    print("STEP 4: Storing in ChromaDB")
    print("=" * 60)

    # If --force-reindex was supplied, delete previous collection to avoid stale IDs
    if args.force_reindex:
        try:
            client.delete_collection(args.collection_name)
            print(f"Deleted existing collection '{args.collection_name}'")
        except Exception:
            pass

    # Build collection-level metadata for model provenance tracking
    collection_metadata: Dict[str, Any] = {
        "embedding_model": args.embedding_model,
        "llm_model": args.llm_model if not args.skip_llm else "none",
        "index_version": INDEX_VERSION,
        "image_weight": str(args.image_weight),
        "text_weight": str(1.0 - args.image_weight),
        "total_documents": str(total_images),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hnsw:space": "cosine"  # Use cosine similarity space for normalized embeddings
    }

    collection = client.get_or_create_collection(
        name=args.collection_name,
        metadata=collection_metadata
    )

    # Build per-document metadata records
    metadatas: List[Dict[str, Any]] = []
    for i in range(total_images):
        metadatas.append({
            "image_id": i,
            "original_caption": captions[i],
            "embedding_model": args.embedding_model,
            "llm_model": args.llm_model if not args.skip_llm else "none",
            "index_version": INDEX_VERSION,
            "indexed_at": datetime.now(timezone.utc).isoformat()
        })

    # Upsert in batches of 100 to stay within ChromaDB payload recommendations
    batch_size = 100
    for start in range(0, total_images, batch_size):
        end = min(start + batch_size, total_images)
        collection.upsert(
            ids=ids[start:end],
            embeddings=fused_embeddings[start:end],
            documents=descriptions[start:end],
            metadatas=metadatas[start:end]
        )
        logger.info(f"Upserted ChromaDB batch {start}-{end}")
        print(f"  Upserted batch {start}-{end}")

    # Output final summary metrics
    summary_msg = (
        f"Indexing complete! Collection: '{args.collection_name}' | "
        f"Documents: {collection.count()} | Embedding: {args.embedding_model} | "
        f"LLM: {args.llm_model if not args.skip_llm else 'none'} | "
        f"Index Ver: {INDEX_VERSION}"
    )
    logger.info(summary_msg)
    print(f"\n{summary_msg}")


# Standard Python CLI execution
if __name__ == "__main__":
    cli_args = parse_args()
    build_index(cli_args)
