from functools import lru_cache
from typing import Annotated, Tuple

import numpy as np
from fastapi import Depends
from sentence_transformers import SentenceTransformer

from app.settings import Settings, get_settings


class SentenceTransformerRepository:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    @lru_cache
    def encode_text(self, text: str) -> np.ndarray:
        return self.model.encode(text)

    @lru_cache
    def encode_texts(self, texts: Tuple[str]) -> np.ndarray:
        return self.model.encode(list(texts))


@lru_cache(maxsize=1)
def get_sentence_transformer_repository(
    settings: Annotated[Settings, Depends(get_settings)],
) -> SentenceTransformerRepository:
    return SentenceTransformerRepository(settings.sentence_transformer_model)
