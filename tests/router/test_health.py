from fastapi import status
from fastapi.testclient import TestClient
from pytest import mark

from app.main import app

client = TestClient(app)


@mark.ut
def test_health_check():
    # when
    response = client.get("/health")
    # then
    assert response.status_code == status.HTTP_200_OK
    assert response.json() is None
