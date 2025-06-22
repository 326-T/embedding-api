from unittest.mock import Mock

import numpy as np
from fastapi import status
from fastapi.testclient import TestClient
from pytest import fixture, mark
from pytest_mock import MockerFixture

from app.main import app
from app.repository.sentence_transformer import (
    SentenceTransformerRepository,
    get_sentence_transformer_repository,
)

client = TestClient(app)


@fixture
def sentence_transformer_repository(mocker: MockerFixture):
    repository: Mock = mocker.create_autospec(spec=SentenceTransformerRepository)
    repository.encode_text.return_value = np.array([0.1, 0.2, 0.3])
    repository.encode_texts.return_value = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    app.dependency_overrides[get_sentence_transformer_repository] = lambda: repository
    return repository


@mark.ut
def test_embed_text(sentence_transformer_repository: Mock):
    # given
    text = "Hello, world!"
    # when
    response = client.post("/embed", json={"text": text})
    # then
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {"embedding": [0.1, 0.2, 0.3]}
    sentence_transformer_repository.encode_text.assert_called_once_with(text)


@mark.ut
def test_embed_texts(sentence_transformer_repository: Mock):
    # given
    texts = ["Hello, world!", "Goodbye, world!"]
    # when
    response = client.post("/embed/batch", json={"texts": texts})
    # then
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {"embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]}
    sentence_transformer_repository.encode_texts.assert_called_once_with(tuple(texts))
