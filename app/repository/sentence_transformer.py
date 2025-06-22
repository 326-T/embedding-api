from functools import lru_cache
from typing import Tuple

import numpy as np
from sentence_transformers import SentenceTransformer


class SentenceTransformerRepository:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    @lru_cache
    def encode_text(self, text: str) -> np.ndarray:
        return self.model.encode(text)

    @lru_cache
    def encode_texts(self, texts: Tuple[str]) -> np.ndarray:
        return self.model.encode(list(texts))


@lru_cache(maxsize=1)
def get_sentence_transformer_repository(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> SentenceTransformerRepository:
    return SentenceTransformerRepository(model_name)
