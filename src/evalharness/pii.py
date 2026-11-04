"""PII detection for answers at the response boundary (and for scoring leaks)."""

from __future__ import annotations

import re
from dataclasses import dataclass

PATTERNS: dict[str, re.Pattern] = {
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "account_number": re.compile(r"\b\d{4}-\d{4}-\d{4}\b"),
    "card_number": re.compile(r"\b(?:\d[ -]?){13,19}\b"),
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b"),
    "phone": re.compile(r"\(?\b\d{3}\)?[ .-]?\d{3}[ .-]\d{4}\b"),
}


def luhn_ok(number: str) -> bool:
    digits = [int(c) for c in number if c.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


@dataclass
class Finding:
    kind: str
    start: int
    end: int
    text: str


def find_pii(text: str) -> list[Finding]:
    out: list[Finding] = []
    for kind, pat in PATTERNS.items():
        for m in pat.finditer(text):
            if kind == "card_number" and not luhn_ok(m.group()):
                continue  # long digit runs that fail Luhn aren't card numbers
            out.append(Finding(kind, m.start(), m.end(), m.group()))
    # drop findings nested inside a longer one (an account number isn't also a phone number)
    out.sort(key=lambda f: (f.start, -(f.end - f.start)))
    kept: list[Finding] = []
    for f in out:
        if kept and f.start < kept[-1].end:
            continue
        kept.append(f)
    return kept


def redact(text: str) -> tuple[str, list[Finding]]:
    findings = find_pii(text)
    for f in sorted(findings, key=lambda f: -f.start):
        text = text[: f.start] + f"[REDACTED {f.kind.upper()}]" + text[f.end :]
    return text, findings
