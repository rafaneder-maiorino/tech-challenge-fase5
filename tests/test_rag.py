"""RAG unit tests (no model download) + integration tests on the real manual (marked slow)."""

from collections import Counter

import pytest
from langchain_core.documents import Document

from support_copilot.rag.chat import ChatService
from support_copilot.rag.config import RagSettings
from support_copilot.rag.ingest import clean, load_manual, section_title, sha256_file, split
from support_copilot.rag.llm import FakeGenerator, ProviderConfigError, build_generator
from support_copilot.rag.prompts import REFUSAL, SYSTEM_PROMPT, user_message
from support_copilot.rag.retriever import Hit

HIT = Hit(
    text="Go to System > Access > Users. Click Add. Fill in the form.",
    page=7,
    section="manual/users",
    score=0.8,
    chunk_id=1,
)


class StubRetriever:
    def __init__(self, hits: list[Hit]) -> None:
        self.hits = hits

    def relevant(self, question: str) -> list[Hit]:
        return self.hits


class ExplodingGenerator:
    name = "boom"
    model = "none"

    def generate(self, question, hits):
        raise AssertionError("LLM must not be called without evidence")


class FixedGenerator:
    name = "fixed"
    model = "none"

    def __init__(self, answer: str) -> None:
        self.answer = answer

    def generate(self, question, hits):
        return self.answer


# ---------- ingestion ----------


def test_clean_drops_the_page_breadcrumb_only_on_the_first_line():
    assert clean("/Troubleshooting /Boot\nBoot\nhangs /a /b") == "Boot\nhangs /a /b"


def test_section_title_is_human_readable():
    assert section_title("troubleshooting/password_reset") == "Troubleshooting > Password reset"


def test_clean_removes_icon_glyphs_and_normalises_menus():
    raw = "Note\nSystem ‣ Settings ‣ Tunables\n\n   spaced    out  "
    assert clean(raw) == "Note\nSystem > Settings > Tunables\nspaced out"


def test_split_assigns_sequential_chunk_ids_and_keeps_metadata():
    docs = [Document(page_content="word " * 800, metadata={"page": 3, "section": "manual/x"})]
    chunks = split(docs, chunk_size=1000, chunk_overlap=150)
    assert len(chunks) > 1
    assert [c.metadata["chunk_id"] for c in chunks] == list(range(len(chunks)))
    assert all(c.metadata["page"] == 3 and c.metadata["section"] == "manual/x" for c in chunks)
    assert all(c.page_content.startswith("Manual > X\n") for c in chunks)  # header is embedded
    assert all(not c.metadata["body"].startswith("Manual > X") for c in chunks)  # body is clean


def test_manual_is_pinned_and_sections_come_from_outline():
    s = RagSettings()
    assert sha256_file(s.manual_path) == s.manual_sha256
    pages = load_manual(s)
    sections = Counter(d.metadata["section"] for d in pages)
    assert len(sections) == 26
    assert "troubleshooting/password_reset" in sections and "unknown" not in sections


# ---------- prompt / generators ----------


def test_prompt_treats_context_as_data_and_defines_refusal():
    assert REFUSAL in SYSTEM_PROMPT
    assert "not instructions" in SYSTEM_PROMPT
    assert "[p. 7 | manual/users]" in user_message("how?", [HIT])


def test_fake_generator_is_extractive_and_cites_the_page():
    answer = FakeGenerator().generate("how do I add a user?", [HIT])
    assert answer.startswith("Go to System > Access > Users.")
    assert answer.endswith("[p. 7]")


def test_unknown_provider_and_missing_key_fail_fast(monkeypatch):
    with pytest.raises(ProviderConfigError, match="unknown"):
        build_generator("llama-on-my-toaster")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(ProviderConfigError, match="GOOGLE_API_KEY"):
        build_generator("gemini")


def test_default_provider_is_fake(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    assert build_generator().name == "fake"


# ---------- chat service ----------


def test_no_evidence_refuses_without_calling_the_llm():
    out = ChatService(StubRetriever([]), ExplodingGenerator()).ask("pizza near the office?")
    assert (out.grounded, out.refusal_reason, out.llm_called, out.answer) == (
        False,
        "no_relevant_context",
        False,
        REFUSAL,
    )


def test_grounded_answer_keeps_sources():
    out = ChatService(StubRetriever([HIT]), FixedGenerator("Use System > Access > Users [p. 7]")).ask("add user?")
    assert out.grounded and out.llm_called and out.sources == [HIT]


def test_model_declining_is_reported_as_not_grounded():
    out = ChatService(StubRetriever([HIT]), FixedGenerator("I could not find this in the manual.")).ask("x?")
    assert (out.grounded, out.refusal_reason) == (False, "model_declined")


# ---------- integration: real manual, real embeddings ----------


@pytest.fixture(scope="module")
def real_retriever():
    from support_copilot.rag.ingest import load_or_build
    from support_copilot.rag.retriever import ManualRetriever

    s = RagSettings()
    return ManualRetriever(load_or_build(s), s.top_k, s.min_score)


@pytest.mark.slow
@pytest.mark.parametrize(
    ("question", "section"),
    [
        ("How do I enable two-factor authentication with Google Authenticator?", "manual/two_factor"),
        ("How do I configure WAN failover between two internet providers?", "manual/multiwan"),
        ("The web interface certificate is broken, how do I restore GUI access?", "troubleshooting/webgui"),
    ],
)
def test_known_questions_retrieve_their_section(real_retriever, question, section):
    hits = real_retriever.relevant(question)
    assert hits and hits[0].section == section


@pytest.mark.slow
def test_out_of_domain_question_has_no_relevant_evidence(real_retriever):
    assert real_retriever.relevant("I was charged twice on my invoice, can I get a refund?") == []
