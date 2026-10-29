from llama_index.core import VectorStoreIndex
from llama_index.core.postprocessor import LLMRerank, SimilarityPostprocessor
from llama_index.core.query_engine import BaseQueryEngine

from app.rag.llm_settings import configure_llama_index
from app.rag.qdrant_store import get_vector_store


def get_query_engine() -> BaseQueryEngine:
    """Create a query engine with similarity filtering and LLM reranking.

    Pipeline: vector search (top_k=10) → SimilarityPostprocessor (cutoff=0.5)
    → LLMRerank (top_n=5, gemini-2.5-flash). No response synthesis —
    only source nodes are used by rag_search_node.
    """
    configure_llama_index()

    vector_store = get_vector_store()
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

    return index.as_query_engine(
        similarity_top_k=10,
        response_mode="no_text",
        node_postprocessors=[
            SimilarityPostprocessor(similarity_cutoff=0.5),
            LLMRerank(top_n=5),
        ],
    )
