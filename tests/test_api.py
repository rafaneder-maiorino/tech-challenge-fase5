"""API contract tests against the versioned champion in models/."""

import logging

import pytest
from fastapi.testclient import TestClient

from support_copilot.api.main import app
from support_copilot.data.source import MAX_CHARS, QUEUES, TECHNICAL_QUEUES

SECRET = "maria.silva@example.com CPF 123.456.789-00"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # context manager runs the lifespan (model load)
        yield c


def test_health_reports_model_version(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_version"].startswith("ticket-classifier/v")


def test_predict_valid_ticket(client):
    r = client.post("/predict", json={"text": "VPN keeps disconnecting after the router firmware update"})
    assert r.status_code == 200
    body = r.json()
    assert body["queue"] in QUEUES
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["is_technical"] == (body["queue"] in TECHNICAL_QUEUES)
    assert body["route"] == ("rag" if body["is_technical"] else "queue")
    probs = [t["probability"] for t in body["top"]]
    assert len(probs) == 3 and probs == sorted(probs, reverse=True)
    assert body["top"][0]["queue"] == body["queue"]


def test_predict_accepts_exactly_max_chars(client):
    assert client.post("/predict", json={"text": "a" * MAX_CHARS}).status_code == 200


def test_text_is_trimmed_before_length_check(client):
    assert client.post("/predict", json={"text": "  " + "a" * MAX_CHARS + "  "}).status_code == 200


@pytest.mark.parametrize(
    "payload",
    [
        {"text": ""},
        {"text": "    \n\t "},
        {"text": "a" * (MAX_CHARS + 1)},
        {"text": 123},
        {"text": None},
        {},
        {"text": "valid text", "priority": "high"},
    ],
    ids=["empty", "only-whitespace", "too-long", "int", "null", "missing", "extra-field"],
)
def test_invalid_payloads_return_422(client, payload):
    r = client.post("/predict", json=payload)
    assert r.status_code == 422
    assert r.json()["detail"]


def test_422_does_not_echo_the_ticket_text(client):
    r = client.post("/predict", json={"text": SECRET + "x" * MAX_CHARS})
    assert r.status_code == 422
    assert "maria.silva" not in r.text
    assert "123.456.789-00" not in r.text


def test_logs_never_contain_raw_text(client, caplog):
    with caplog.at_level(logging.INFO, logger="support_copilot.api"):
        client.post("/predict", json={"text": f"Cannot log in. {SECRET}"})
    logged = " ".join(r.getMessage() for r in caplog.records)
    assert '"event": "predict"' in logged
    assert "maria.silva" not in logged
    assert "123.456.789-00" not in logged
