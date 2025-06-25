from datetime import datetime
from unittest.mock import Mock

import numpy as np
from fastapi import status
from fastapi.testclient import TestClient
from pytest import fixture, mark
from pytest_mock import MockerFixture

from app.main import app
from app.repository.pgvector import (
    PgVectorRepository,
    VectorRecord,
    get_pgvector_repository,
)
from app.repository.sentence_transformer import (
    SentenceTransformerRepository,
    get_sentence_transformer_repository,
)

client = TestClient(app)


@fixture
def sentence_transformer_repository(mocker: MockerFixture):
    repository: Mock = mocker.create_autospec(spec=SentenceTransformerRepository)
    repository.encode_text.return_value = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    repository.encode_texts.return_value = np.array(
        [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32
    )
    app.dependency_overrides[get_sentence_transformer_repository] = lambda: repository
    return repository


@fixture
def pgvector_repository(mocker: MockerFixture):
    repository: Mock = mocker.create_autospec(spec=PgVectorRepository)
    repository.hybrid_search.return_value = [
        VectorRecord(
            id=1,
            category="test_category",
            title="Test Title",
            text="Test text content",
            vector_score=0.95,
            text_score=0.85,
            hibrid_score=0.90,
            created_at=datetime(2024, 1, 1, 12, 0, 0),
        )
    ]
    repository.copy.return_value = None
    app.dependency_overrides[get_pgvector_repository] = lambda: repository
    return repository


@mark.ut
def test_hybrid_search(
    sentence_transformer_repository: Mock, pgvector_repository: Mock
):
    # given
    request_data = {"category": "test_category", "query": "test query"}

    # when
    response = client.post("/hybrid/search", json=request_data)

    # then
    assert response.status_code == status.HTTP_200_OK
    response_data = response.json()
    assert len(response_data) == 1
    assert response_data[0]["id"] == 1
    assert response_data[0]["category"] == "test_category"
    assert response_data[0]["title"] == "Test Title"
    assert response_data[0]["text"] == "Test text content"
    assert response_data[0]["vector_score"] == 0.95
    assert response_data[0]["text_score"] == 0.85
    assert response_data[0]["hibrid_score"] == 0.90
    assert response_data[0]["created_at"] == "2024-01-01T12:00:00"

    sentence_transformer_repository.encode_text.assert_called_once_with("test query")
    pgvector_repository.hybrid_search.assert_called_once()
    call_kwargs = pgvector_repository.hybrid_search.call_args.kwargs
    assert call_kwargs["category"] == "test_category"
    assert call_kwargs["query"] == "test query"
    assert np.array_equal(
        call_kwargs["embedding"], np.array([0.1, 0.2, 0.3], dtype=np.float32)
    )


@mark.ut
def test_hybrid_insert(
    sentence_transformer_repository: Mock, pgvector_repository: Mock
):
    # given
    request_data = {
        "items": [
            {"category": "cat1", "title": "Title 1", "text": "Text 1"},
            {"category": "cat2", "title": "Title 2", "text": "Text 2"},
        ]
    }

    # when
    response = client.post("/hybrid/insert", json=request_data)

    # then
    assert response.status_code == status.HTTP_200_OK

    sentence_transformer_repository.encode_texts.assert_called_once_with(
        ("Title 1 Text 1", "Title 2 Text 2")
    )

    pgvector_repository.copy.assert_called_once()
    call_args = pgvector_repository.copy.call_args[0][0]
    assert len(call_args) == 2
    assert call_args[0].category == "cat1"
    assert call_args[0].title == "Title 1"
    assert call_args[0].text == "Text 1"
    assert np.array_equal(
        call_args[0].embedding, np.array([0.1, 0.2, 0.3], dtype=np.float32)
    )
    assert call_args[1].category == "cat2"
    assert call_args[1].title == "Title 2"
    assert call_args[1].text == "Text 2"
    assert np.array_equal(
        call_args[1].embedding, np.array([0.4, 0.5, 0.6], dtype=np.float32)
    )
