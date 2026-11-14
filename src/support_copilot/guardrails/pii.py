"""Mask personal data (LGPD) BEFORE any text leaves the machine for the LLM.

Detected types and placeholders: e-mail [EMAIL], card number [CARD] (Luhn-valid), CNPJ [CNPJ],
CPF [CPF] (formatted, or 11 bare digits with valid check digits), phone [PHONE].

IP addresses, hostnames and MAC addresses are deliberately NOT masked: they are the technical
substance of a firewall question, and the manual-only answer needs them. Person names are not
detected (regex cannot do it reliably); a NER model such as Presidio is the prescribed next step.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
CARD = re.compile(r"(?<![\d.])(?:\d[ -]?){12,18}\d(?![\d.])")
CNPJ = re.compile(r"(?<!\d)\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}(?!\d)")
CPF_FORMATTED = re.compile(r"(?<!\d)\d{3}\.\d{3}\.\d{3}-\d{2}(?!\d)")
CPF_BARE = re.compile(r"(?<![\d.])\d{11}(?![\d.])")
# Phones: area code in parentheses or followed by a separator; or a bare Brazilian mobile
# (DDD + 9 + 8 digits); or the US format. A bare 11-digit number that is neither a valid CPF nor
# a mobile (e.g. a ticket id) is left alone.
PHONE = re.compile(
    r"(?<![\w.])(?:\+\d{1,3}[\s.-]?)?(?:\(\d{2,3}\)[\s.-]?|\d{2,3}[\s.-])\d{4,5}[\s.-]?\d{4}(?![\w.])"
    r"|(?<![\w.])(?:\+?55)?\d{2}9\d{8}(?![\w.])"
    r"|(?<![\w.])(?:\+1[\s.-]?)?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}(?![\w.])"
)


def luhn_ok(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
    return total % 10 == 0


def cpf_ok(digits: str) -> bool:
    if len(digits) != 11 or len(set(digits)) == 1:
        return False
    for n in (9, 10):
        s = sum(int(digits[i]) * (n + 1 - i) for i in range(n))
        if (s * 10 % 11) % 10 != int(digits[n]):
            return False
    return True


@dataclass(frozen=True)
class Masked:
    text: str
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def found(self) -> bool:
        return bool(self.counts)


def mask_pii(text: str) -> Masked:
    """Replace personal data with typed placeholders. Order matters: most specific first."""
    counts: Counter[str] = Counter()

    def sub(pattern: re.Pattern[str], label: str, s: str, check=None) -> str:
        def repl(m: re.Match[str]) -> str:
            if check is not None and not check(re.sub(r"\D", "", m.group(0))):
                return m.group(0)
            counts[label] += 1
            return f"[{label}]"

        return pattern.sub(repl, s)

    out = sub(EMAIL, "EMAIL", text)
    out = sub(CARD, "CARD", out, check=lambda d: 13 <= len(d) <= 19 and luhn_ok(d))
    out = sub(CNPJ, "CNPJ", out)
    out = sub(CPF_FORMATTED, "CPF", out)
    out = sub(CPF_BARE, "CPF", out, check=cpf_ok)
    out = sub(PHONE, "PHONE", out)
    return Masked(out, dict(counts))
