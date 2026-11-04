"""Shared types: golden cases, responses from the system under test, per-case scores."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path


class CaseKind(StrEnum):
    ANSWERABLE = "answerable"  # the knowledge base contains the answer
    UNANSWERABLE = "unanswerable"  # it doesn't: the right behaviour is to say so
    PII_BAIT = "pii_bait"  # asks for personal data: the right behaviour is to refuse


@dataclass
class GoldenCase:
    case_id: str
    tenant: str
    question: str
    kind: CaseKind
    reference: str = ""  # reference answer, used by the LLM judge
    must_include: list[str] = field(default_factory=list)  # facts a correct answer states
    gold_doc_ids: list[str] = field(default_factory=list)  # docs retrieval should surface
    tags: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> GoldenCase:
        return cls(**{**d, "kind": CaseKind(d["kind"])})


def load_golden(path: str | Path) -> list[GoldenCase]:
    with Path(path).open() as f:
        return [GoldenCase.from_dict(json.loads(line)) for line in f if line.strip()]


def save_golden(cases: list[GoldenCase], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w") as f:
        for c in cases:
            f.write(json.dumps({**asdict(c), "kind": c.kind.value}) + "\n")


@dataclass
class Context:
    doc_id: str
    text: str


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class Response:
    answer: str
    contexts: list[Context]
    usage: Usage
    model: str
    latency_ms: float = 0.0
    guardrail_events: list[str] = field(default_factory=list)


@dataclass
class CaseResult:
    case_id: str
    tenant: str
    kind: str
    answer: str
    retrieval_recall: float | None  # None when the case has no gold docs
    faithfulness: float  # share of answer claims supported by retrieved context
    correct: bool  # answered right, abstained right, or refused right
    hallucinated: bool  # states a figure or claim the context doesn't support
    pii_leak: bool
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float
    judge: dict = field(default_factory=dict)
    guardrail_events: list[str] = field(default_factory=list)
