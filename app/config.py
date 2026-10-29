from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

    GEMINI_API_KEY: str = Field(..., repr=False)
    QDRANT_URL: str
    QDRANT_API_KEY: str = Field(..., repr=False)
    QDRANT_COLLECTION_NAME: str = "support-agent"
    LANGFUSE_SECRET_KEY: str | None = Field(default=None, repr=False)
    LANGFUSE_PUBLIC_KEY: str | None = Field(default=None, repr=False)
    LANGFUSE_BASE_URL: str = "https://cloud.langfuse.com"
    LANGFUSE_ENABLED: bool = True


settings = Settings()
