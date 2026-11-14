"""Guardrails in three layers around the RAG chain (defence in depth, no single layer trusted).

input   -> blocklist + direct prompt injection (before retrieval; blocked = HTTP 400)
PII     -> personal data masked before anything is sent to the LLM (LGPD)
context -> indirect prompt injection in retrieved chunks (poisoned chunks dropped)
output  -> citations must point to the pages sent to the model; PII masked in the answer
"""

from support_copilot.guardrails.context import ContextGuard, ContextVerdict
from support_copilot.guardrails.input import InputGuard, InputVerdict
from support_copilot.guardrails.output import OutputGuard, OutputVerdict
from support_copilot.guardrails.pii import Masked, mask_pii
from support_copilot.guardrails.rules import RuleSet, normalize

__all__ = [
    "ContextGuard",
    "ContextVerdict",
    "InputGuard",
    "InputVerdict",
    "Masked",
    "OutputGuard",
    "OutputVerdict",
    "RuleSet",
    "mask_pii",
    "normalize",
]
