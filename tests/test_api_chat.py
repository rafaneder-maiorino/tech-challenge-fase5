"""/chat contract tests with the default fake provider (no API key needed)."""

import logging

import pytest
from fastapi.testclient import TestClient

from support_copilot.api.main import app
from support_copilot.data.source import MAX_CHARS
from support_copilot.rag.prompts import REFUSAL

SECRET = "joao.souza@example.com"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_reports_rag(client):
    rag = client.get("/health").json()["rag"]
    assert rag["chunks"] > 100
    assert rag["provider"] == "fake"
    assert 0 < rag["min_score"] < 1


def test_in_domain_question_is_answered_with_page_citations(client):
    r = client.post("/chat", json={"question": "How do I reset the root password without reinstalling?"})
    assert r.status_code == 200
    body = r.json()
    assert body["grounded"] is True and body["refusal_reason"] is None
    assert body["sources"] and all(s["score"] >= 0.67 for s in body["sources"])
    assert f"[p. {body['sources'][0]['page']}]" in body["answer"]


def test_out_of_domain_question_is_refused_without_sources(client):
    r = client.post("/chat", json={"question": "Can you recommend a good pizza place near the office?"})
    body = r.json()
    assert r.status_code == 200
    assert (body["grounded"], body["refusal_reason"], body["answer"], body["sources"]) == (
        False,
        "no_relevant_context",
        REFUSAL,
        [],
    )


@pytest.mark.parametrize("payload", [{"question": ""}, {"question": "x" * (MAX_CHARS + 1)}, {"q": "hi"}])
def test_invalid_chat_payloads_return_422(client, payload):
    assert client.post("/chat", json=payload).status_code == 422


def test_provider_failure_returns_502_without_leaking_details(client, monkeypatch):
    class Down:
        name = "down"
        model = "none"

        def generate(self, question, hits):
            raise TimeoutError("upstream secret detail")

    monkeypatch.setattr(app.state.chat, "generator", Down())
    r = client.post("/chat", json={"question": "How do I reset the root password without reinstalling?"})
    assert r.status_code == 502
    assert r.json() == {"detail": "llm_unavailable"}
    assert "secret" not in r.text


def test_chat_logs_never_contain_the_question(client, caplog):
    with caplog.at_level(logging.INFO, logger="support_copilot.api"):
        client.post("/chat", json={"question": f"How do I back up the configuration? I am {SECRET}"})
    logged = " ".join(rec.getMessage() for rec in caplog.records)
    assert '"event": "chat"' in logged
    assert SECRET not in logged
