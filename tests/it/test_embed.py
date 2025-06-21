from fastapi import status
from fastapi.testclient import TestClient
from pytest import mark

from app.main import app
from app.router.embed import EmbeddingListResponse, EmbeddingResponse

client = TestClient(app)


@mark.it
def test_embed_text():
    # given
    text = "Hello, world!"
    # when
    response = client.post("/embed", json={"text": text})
    # then
    assert response.status_code == status.HTTP_200_OK
    body = EmbeddingResponse(**response.json())
    assert len(body.embedding) > 0


@mark.it
def test_embed_texts():
    # given
    texts = ["Hello, world!", "Goodbye, world!"]
    # when
    response = client.post("/embed/batch", json={"texts": texts})
    # then
    assert response.status_code == status.HTTP_200_OK
    body = EmbeddingListResponse(**response.json())
    assert len(body.embeddings) == len(texts)
