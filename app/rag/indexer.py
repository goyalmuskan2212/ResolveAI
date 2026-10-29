import logging
from pathlib import Path

from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter

from app.rag.llm_settings import configure_llama_index
from app.rag.qdrant_store import get_vector_store

logger = logging.getLogger(__name__)

KNOWLEDGE_BASE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "knowledge_base"


def run_indexing(knowledge_base_dir: Path | None = None) -> VectorStoreIndex:
    """Load documents from the knowledge base, split into chunks, and index into Qdrant.

    Args:
        knowledge_base_dir: Path to the knowledge base directory.
            Defaults to ``data/knowledge_base/`` at the project root.

    Returns:
        The created VectorStoreIndex.
    """
    configure_llama_index()

    kb_dir = knowledge_base_dir or KNOWLEDGE_BASE_DIR
    logger.info("Loading documents from %s", kb_dir)

    documents = SimpleDirectoryReader(input_dir=str(kb_dir)).load_data()
    logger.info("Loaded %d document(s)", len(documents))

    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    nodes = splitter.get_nodes_from_documents(documents)
    logger.info("Split into %d node(s)", len(nodes))

    vector_store = get_vector_store()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
    )
    logger.info("Indexing complete")

    return index
