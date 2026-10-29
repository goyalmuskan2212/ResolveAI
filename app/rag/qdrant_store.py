from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from app.config import settings


def get_qdrant_client() -> QdrantClient:
    """Create and return a QdrantClient connected to the configured Qdrant instance."""
    return QdrantClient(
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY,
    )


def get_vector_store() -> QdrantVectorStore:
    """Create and return a QdrantVectorStore for LlamaIndex."""
    client = get_qdrant_client()
    return QdrantVectorStore(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        client=client,
    )
