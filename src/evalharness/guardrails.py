"""Guardrails at the response boundary: they run on every production answer, not only in evals.

- `PIIGuard`: redacts SSNs, account and card numbers (Luhn-checked), emails and phone numbers. A
  question that asks for personal data gets a refusal instead of a redacted dump.
- `UnsupportedFigureGuard`: a figure in the answer that doesn't appear in the retrieved context is
  treated as fabricated, and the answer is replaced with "can't confirm".

Neither guard can catch a *supported* wrong answer (the right figure for the wrong product): that
is what the eval set is for.
"""

from __future__ import annotations

from dataclasses import replace

from .app import PII_ASK
from .metrics import claims, figures
from .pii import redact
from .types import Response

PII_REFUSAL = "I can't share personal information about customers."
CANT_CONFIRM = "I can't confirm that from our documentation. Please contact support."


class PIIGuard:
    name = "pii"

    def apply(self, question: str, r: Response) -> Response:
        text, findings = redact(r.answer)
        if not findings:
            return r
        kinds = sorted({f.kind for f in findings})
        # asked for personal data: refuse outright; otherwise keep the answer, redacted
        answer = PII_REFUSAL if PII_ASK.search(question) else text
        return replace(
            r,
            answer=answer,
            guardrail_events=r.guardrail_events + [f"pii_blocked:{','.join(kinds)}"],
        )


class UnsupportedFigureGuard:
    name = "unsupported_figure"

    def apply(self, question: str, r: Response) -> Response:
        ctx = figures(" ".join(c.text for c in r.contexts))
        made_up = figures(" ".join(claims(r.answer))) - ctx
        if not made_up:
            return r
        return replace(
            r,
            answer=CANT_CONFIRM,
            guardrail_events=r.guardrail_events
            + [f"unsupported_figure:{','.join(sorted(made_up))}"],
        )


DEFAULT_GUARDS = (PIIGuard(), UnsupportedFigureGuard())
