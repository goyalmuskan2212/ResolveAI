from llama_index.core import Settings
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI

from app.config import settings as app_settings


def configure_llama_index() -> None:
    """Configure LlamaIndex global Settings with Gemini models.

    Must be called before any indexing or querying operations.
    Safe to call multiple times (idempotent).
    """
    Settings.embed_model = GoogleGenAIEmbedding(
        model_name="gemini-embedding-2-preview",
        api_key=app_settings.GEMINI_API_KEY,
    )
    Settings.llm = GoogleGenAI(
        model="gemini-2.5-flash",
        api_key=app_settings.GEMINI_API_KEY,
    )
