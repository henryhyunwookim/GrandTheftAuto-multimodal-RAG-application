"""
GTA Multimodal RAG Search - Streamlit Root Entrypoint.

Convenience root entrypoint enabling `streamlit run app.py`.
Delegates execution to the modularized UI application in `src/app.py`.

Usage:
    streamlit run app.py
    streamlit run app.py --server.port 8501
"""

from __future__ import annotations

from pathlib import Path
import sys

# Ensure repository root is on sys.path
REPO_ROOT: Path = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import and execute the main Streamlit application
from src.app import main

if __name__ == "__main__":
    main()
