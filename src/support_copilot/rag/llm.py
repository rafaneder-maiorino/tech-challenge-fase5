"""Answer generators. LLM_PROVIDER = fake (default, offline, deterministic) | gemini | openai.

Keys come only from the environment (GOOGLE_API_KEY / OPENAI_API_KEY) and are never logged.
The fake provider is extractive: it returns the opening of the best passage with its page, so the
whole /chat path (retrieval, threshold, citations, guardrails) runs and is testable without a key.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Protocol

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from support_copilot.rag.prompts import SYSTEM_PROMPT, user_message
from support_copilot.rag.retriever import Hit

logger = logging.getLogger(__name__)

DEFAULT_MODELS = {"gemini": "gemini-2.5-flash", "openai": "gpt-4.1-mini"}
KEY_VARS = {"gemini": "GOOGLE_API_KEY", "openai": "OPENAI_API_KEY"}


class ProviderConfigError(RuntimeError):
    """Raised at startup when the selected provider cannot be configured (e.g. missing key)."""


class AnswerGenerator(Protocol):
    name: str
    model: str

    def generate(self, question: str, hits: list[Hit]) -> str: ...


class FakeGenerator:
    """Deterministic extractive answer: first sentences of the top passage, cited."""

    name = "fake"
    model = "extractive"
    _sentence = re.compile(r"(?<=[.!?])\s+")

    def __init__(self, max_chars: int = 400) -> None:
        self.max_chars = max_chars

    def generate(self, question: str, hits: list[Hit]) -> str:
        top = hits[0]
        flat = " ".join(top.text.split())
        out = ""
        for sentence in self._sentence.split(flat):
            if out and len(out) + len(sentence) > self.max_chars:
                break
            out = f"{out} {sentence}".strip()
        return f"{out[: self.max_chars]} [p. {top.page}]"


class ChatModelGenerator:
    def __init__(self, name: str, model: str, chat_model: BaseChatModel) -> None:
        self.name = name
        self.model = model
        self.chat_model = chat_model

    def generate(self, question: str, hits: list[Hit]) -> str:
        reply = self.chat_model.invoke([SystemMessage(SYSTEM_PROMPT), HumanMessage(user_message(question, hits))])
        content = reply.content
        if isinstance(content, list):  # some providers return content blocks
            content = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
        return str(content).strip()


def build_generator(provider: str | None = None) -> AnswerGenerator:
    provider = (provider or os.environ.get("LLM_PROVIDER", "fake")).strip().lower()
    if provider == "fake":
        return FakeGenerator()
    if provider not in DEFAULT_MODELS:
        raise ProviderConfigError(f"unknown LLM_PROVIDER={provider!r}; use fake, gemini or openai")
    key_var = KEY_VARS[provider]
    if not os.environ.get(key_var):
        raise ProviderConfigError(f"LLM_PROVIDER={provider} needs {key_var} in the environment (.env)")

    model = os.environ.get(f"{provider.upper()}_MODEL", DEFAULT_MODELS[provider])
    max_tokens = int(os.environ.get("LLM_MAX_TOKENS", "400"))
    timeout = float(os.environ.get("LLM_TIMEOUT_S", "30"))
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        chat: BaseChatModel = ChatGoogleGenerativeAI(
            model=model, temperature=0, max_output_tokens=max_tokens, timeout=timeout, max_retries=2
        )
    else:
        from langchain_openai import ChatOpenAI

        chat = ChatOpenAI(model=model, temperature=0, max_tokens=max_tokens, timeout=timeout, max_retries=2)
    logger.info("llm provider=%s model=%s max_tokens=%d timeout=%.0fs", provider, model, max_tokens, timeout)
    return ChatModelGenerator(provider, model, chat)
