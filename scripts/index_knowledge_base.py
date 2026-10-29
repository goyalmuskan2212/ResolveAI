"""Index the knowledge base documents into Qdrant.

Usage::

    python scripts/index_knowledge_base.py
"""

import logging
import sys
from pathlib import Path

# Ensure the project root is on sys.path so ``app`` package is importable.
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from app.rag.indexer import run_indexing  # noqa: E402


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger(__name__)

    logger.info("Starting knowledge base indexing...")
    run_indexing()
    logger.info("Indexing finished successfully.")


if __name__ == "__main__":
    main()
