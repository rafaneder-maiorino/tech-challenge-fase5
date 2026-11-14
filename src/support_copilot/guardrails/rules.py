"""Load and match the declarative guardrail rules in ``rules.yaml``.

Text is normalised before matching so trivial obfuscation (accents, casing, zero-width
characters, leetspeak and, when enabled in the rules file, letters spelled out with spaces and
base64 payloads) does not slip through. Normalisation is for matching only: the text that
continues down the pipeline is never altered by this module.
"""

from __future__ import annotations

import base64
import binascii
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

RULES_PATH = Path(__file__).with_name("rules.yaml")

Scope = Literal["input", "context"]

_INVISIBLE = dict.fromkeys(map(ord, "​‌‍⁠﻿­"), None)
_QUOTES = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"'})
_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"})
_SPACES = re.compile(r"\s+")


def normalize(text: str) -> str:
    """NFKD, accents stripped, casefolded, invisible characters removed, whitespace collapsed."""
    t = unicodedata.normalize("NFKD", text.translate(_INVISIBLE).translate(_QUOTES))
    t = "".join(c for c in t if not unicodedata.combining(c)).casefold()
    return _SPACES.sub(" ", t).strip()


_SPACED = re.compile(r"(?<!\S)\w(?: \w){2,}(?!\S)")  # "i g n o r e" (single spaces inside a word)
_BASE64 = re.compile(r"[A-Za-z0-9+/]{16,}={0,2}")


def _despace(text: str) -> str:
    """Join letters spelled out with single spaces; 2+ spaces still separate words."""
    return _SPACED.sub(lambda m: m.group(0).replace(" ", ""), text)


def _decoded_base64(text: str) -> list[str]:
    """Printable UTF-8 payloads hidden as base64 blobs (an encoded instruction is still one)."""
    out = []
    for blob in _BASE64.findall(text):
        try:
            decoded = base64.b64decode(blob + "=" * (-len(blob) % 4), validate=True).decode("utf-8")
        except (binascii.Error, UnicodeDecodeError):
            continue
        if decoded and sum(c.isprintable() or c.isspace() for c in decoded) / len(decoded) > 0.95:
            out.append(decoded)
    return out


def variants(text: str, despace: bool = True, decode_base64: bool = True) -> tuple[str, ...]:
    """Normalised views of the text that rules are matched against.

    The normalised text and its de-leeted copy ("1gn0re" -> "ignore"); optionally the copy with
    spelled-out letters joined ("i g n o r e" -> "ignore") and the content of any base64 blob.
    """
    raw = text.translate(_INVISIBLE)
    candidates = [normalize(raw)]
    if despace:
        candidates.append(normalize(_despace(raw)))
    if decode_base64:
        candidates.extend(normalize(d) for d in _decoded_base64(raw))
    out: list[str] = []
    for c in candidates:
        for v in (c, c.translate(_LEET)):
            if v not in out:
                out.append(v)
    return tuple(out)


@dataclass(frozen=True)
class Rule:
    id: str
    kind: Literal["blocklist", "injection"]
    category: str
    scope: frozenset[str]
    regex: re.Pattern[str]
    case_sensitive: bool = False

    def matches(self, raw: str, normalized: tuple[str, ...]) -> bool:
        if self.case_sensitive:
            return bool(self.regex.search(raw.translate(_INVISIBLE)))
        return any(self.regex.search(v) for v in normalized)


@dataclass(frozen=True)
class Match:
    rule: str
    kind: str
    category: str


class RuleSet:
    def __init__(self, rules: list[Rule], version: int, despace: bool = False, decode_base64: bool = False) -> None:
        self.rules = rules
        self.version = version
        self.despace = despace
        self.decode_base64 = decode_base64

    @classmethod
    def load(cls, path: Path = RULES_PATH) -> RuleSet:
        spec = yaml.safe_load(path.read_text(encoding="utf-8"))
        rules: list[Rule] = []
        for item in spec["rules"]:
            if "terms" in item:
                alternatives = "|".join(re.escape(normalize(t)) for t in item["terms"])
                pattern = rf"(?<![\w-])(?:{alternatives})(?![\w-])"
            else:
                pattern = item["pattern"]
            case_sensitive = bool(item.get("case_sensitive", False))
            rules.append(
                Rule(
                    id=item["id"],
                    kind=item["kind"],
                    category=item["category"],
                    scope=frozenset(item["scope"]),
                    regex=re.compile(pattern, 0 if case_sensitive else re.IGNORECASE),
                    case_sensitive=case_sensitive,
                )
            )
        ids = [r.id for r in rules]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate rule ids in rules.yaml")
        norm = spec.get("normalization") or {}
        return cls(
            rules,
            int(spec.get("version", 1)),
            despace=bool(norm.get("despace", False)),
            decode_base64=bool(norm.get("base64", False)),
        )

    def first_match(self, text: str, scope: Scope) -> Match | None:
        """First rule (in file order) that fires on ``text`` for the given scope."""
        normalized = variants(text, self.despace, self.decode_base64)
        for rule in self.rules:
            if scope in rule.scope and rule.matches(text, normalized):
                return Match(rule.id, rule.kind, rule.category)
        return None
