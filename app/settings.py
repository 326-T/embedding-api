from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

MODEL_DIMENSIONS = {
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "sentence-transformers/paraphrase-MiniLM-L6-v2": 384,
    "sentence-transformers/all-mpnet-base-v2": 768,
    "intfloat/multilingual-e5-large": 1024,
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", frozen=True)
    connection_string: str = "postgresql://pgvector:pgvector@localhost:5432/pgvector"
    sentence_transformer_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    @property
    def vector_dimension(self) -> int:
        return MODEL_DIMENSIONS[self.sentence_transformer_model]


@lru_cache()
def get_settings() -> Settings:
    return Settings()
