from typing import Annotated, List

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.repository import (
    SentenceTransformerRepository,
    get_sentence_transformer_repository,
)


class TextRequest(BaseModel):
    text: str


class TextListRequest(BaseModel):
    texts: List[str]


class EmbeddingResponse(BaseModel):
    embedding: List[float]


class EmbeddingListResponse(BaseModel):
    embeddings: List[List[float]]


router = APIRouter()


@router.get("/health")
async def health_check():
    return


@router.post("/embed")
async def embed_text(
    request: TextRequest,
    service: Annotated[
        SentenceTransformerRepository, Depends(get_sentence_transformer_repository)
    ],
):
    embedding = service.encode_text(request.text)
    return EmbeddingResponse(embedding=embedding.tolist())


@router.post("/embed/batch")
async def embed_texts(
    request: TextListRequest,
    service: Annotated[
        SentenceTransformerRepository, Depends(get_sentence_transformer_repository)
    ],
):
    embeddings = service.encode_texts(request.texts)
    return EmbeddingListResponse(embeddings=embeddings.tolist())
