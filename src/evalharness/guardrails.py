"""Guardrails at the response boundary: they run on every production answer, not only in evals.

- `PIIGuard`: redacts SSNs, account and card numbers (Luhn-checked), emails and phone numbers. A
  question that asks for personal data gets a refusal instead of a redacted dump.
- `UnsupportedFigureGuard`: a figure in the answer must appear in a retrieved document that is
  about what the claim is about; otherwise it is treated as fabricated and the answer is
  replaced with "can't confirm".

Neither guard can catch a *supported* wrong answer (the right figure for the wrong product): that
is what the eval set is for.
"""

from __future__ import annotations

import re
from dataclasses import replace

from .app import PII_ASK
from .metrics import claims, content_words, figures
from .pii import redact
from .types import Response

_SENT = re.compile(r"(?<=[.!?])\s+")

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


def figure_anchored(claim: str, figure: str, contexts, min_overlap: float = 0.3) -> bool:  # noqa: ANN001
    """True if the figure appears in a retrieved sentence that is about what the claim is about.

    The anchor is the sentence containing the figure plus its document's opening sentence (which
    names the product or policy). A paraphrase ("Everyday Checking's fee is $8 a month") passes;
    a made-up figure that happens to equal an unrelated product's figure does not. Matching the
    whole document was too lenient: boilerplate ("fees are disclosed in the schedule...") let a
    fabricated "$62, based on our current fee schedule" through.
    """
    cw = content_words(claim)
    for c in contexts:
        sents = _SENT.split(c.text)
        for sent in sents:
            if figure not in figures(sent):
                continue
            anchor = content_words(sent) | content_words(sents[0])
            if not cw or len(cw & anchor) / len(cw) >= min_overlap:
                return True
    return False


class UnsupportedFigureGuard:
    name = "unsupported_figure"

    def apply(self, question: str, r: Response) -> Response:
        made_up = sorted(
            {
                f
                for claim in claims(r.answer)
                for f in figures(claim)
                if not figure_anchored(claim, f, r.contexts)
            }
        )
        if not made_up:
            return r
        return replace(
            r,
            answer=CANT_CONFIRM,
            guardrail_events=r.guardrail_events + [f"unsupported_figure:{','.join(made_up)}"],
        )


DEFAULT_GUARDS = (PIIGuard(), UnsupportedFigureGuard())
