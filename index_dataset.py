"""
Offline Indexing Pipeline Entry Point.

Convenience wrapper enabling `python index_dataset.py` from repository root.
Delegates to `scripts/index_dataset.py`.

Usage / CLI Invocation:
    python index_dataset.py
    python index_dataset.py --force-reindex
    python index_dataset.py --sample-size 10
"""

from __future__ import annotations

from pathlib import Path
import runpy
import sys

# Ensure repository root is on sys.path
REPO_ROOT: Path = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if __name__ == "__main__":
    script_path = REPO_ROOT / "scripts" / "index_dataset.py"
    runpy.run_path(str(script_path), run_name="__main__")
