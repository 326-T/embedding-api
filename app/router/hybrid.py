from datetime import datetime
from typing import Annotated, List, Optional

import numpy as np
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.repository.pgvector import (
    InsertVector,
    PgVectorRepository,
    get_pgvector_repository,
)
from app.repository.sentence_transformer import (
    SentenceTransformerRepository,
    get_sentence_transformer_repository,
)


class SearchRequest(BaseModel):
    category: Optional[str] = None
    query: str


class SearchResponse(BaseModel):
    id: int
    category: str
    title: str
    text: str
    vector_score: float
    text_score: float
    hibrid_score: float
    created_at: datetime


class InsertRequest(BaseModel):
    category: str
    title: str
    text: str


class BulkRequest(BaseModel):
    items: List[InsertRequest]


router = APIRouter()


@router.post("/hybrid/search")
async def hybrid_search(
    request: SearchRequest,
    sentence_transformer: Annotated[
        SentenceTransformerRepository, Depends(get_sentence_transformer_repository)
    ],
    pgvector: Annotated[PgVectorRepository, Depends(get_pgvector_repository)],
):
    embedding = sentence_transformer.encode_text(request.query)
    results = pgvector.hybrid_search(
        embedding=embedding,
        query=request.query,
        category=request.category,
    )
    return [SearchResponse(**result.model_dump()) for result in results]


@router.post("/hybrid/insert")
async def hybrid_insert(
    request: BulkRequest,
    service: Annotated[
        SentenceTransformerRepository, Depends(get_sentence_transformer_repository)
    ],
    pgvector: Annotated[PgVectorRepository, Depends(get_pgvector_repository)],
):
    embeddings: np.ndarray = service.encode_texts(
        tuple(f"{item.title} {item.text}" for item in request.items)
    )
    data: List[InsertVector] = [
        InsertVector(
            category=item.category,
            title=item.title,
            text=item.text,
            embedding=np.array(embedding, dtype=np.float32),
        )
        for item, embedding in zip(request.items, list(embeddings))
    ]
    pgvector.copy(data)
