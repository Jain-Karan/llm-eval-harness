"""Run golden sets against a target and aggregate per tenant."""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .cost import cost_usd
from .judge import Judge
from .metrics import claims, faithfulness, is_correct, retrieval_recall
from .pii import find_pii
from .targets import Target
from .tracing import NullTracer, Tracer
from .types import CaseKind, CaseResult, GoldenCase


def score_case(case: GoldenCase, r, judge: Judge | None = None) -> CaseResult:  # noqa: ANN001
    f = faithfulness(r.answer, r.contexts)
    verdict = {}
    if judge is not None:
        v = judge.grade(case, r)
        verdict = {
            "judge": judge.name,
            "correct": v.correct,
            "faithful": v.faithful,
            "rationale": v.rationale,
        }
    return CaseResult(
        case_id=case.case_id,
        tenant=case.tenant,
        kind=case.kind.value,
        answer=r.answer,
        retrieval_recall=retrieval_recall(case, r.contexts),
        faithfulness=f,
        correct=is_correct(case, r),
        hallucinated=bool(claims(r.answer)) and f < 1.0,
        pii_leak=bool(find_pii(r.answer)),
        input_tokens=r.usage.input_tokens,
        output_tokens=r.usage.output_tokens,
        cost_usd=cost_usd(r.model, r.usage),
        latency_ms=r.latency_ms,
        judge=verdict,
        guardrail_events=r.guardrail_events,
    )


@dataclass
class Summary:
    cases: int
    accuracy: float  # all case kinds
    answer_accuracy: float  # answerable cases only
    abstention_accuracy: float  # unanswerable cases only
    pii_refusal_rate: float  # PII bait refused without leaking
    hallucination_rate: float  # share of cases with any unsupported claim
    pii_leaks: int
    faithfulness: float
    retrieval_recall: float
    cost_per_query_usd: float
    input_tokens_per_query: float
    output_tokens_per_query: float
    p95_latency_ms: float
    guardrail_interventions: int
    judge_agreement: float | None = None


def summarise(results: list[CaseResult]) -> Summary:
    def rate(rs: list[CaseResult], pred) -> float:  # noqa: ANN001
        return sum(1 for r in rs if pred(r)) / len(rs) if rs else 1.0

    by_kind = {k.value: [r for r in results if r.kind == k.value] for k in CaseKind}
    recalls = [r.retrieval_recall for r in results if r.retrieval_recall is not None]
    lat = sorted(r.latency_ms for r in results)
    judged = [r for r in results if r.judge]
    return Summary(
        cases=len(results),
        accuracy=rate(results, lambda r: r.correct),
        answer_accuracy=rate(by_kind["answerable"], lambda r: r.correct),
        abstention_accuracy=rate(by_kind["unanswerable"], lambda r: r.correct),
        pii_refusal_rate=rate(by_kind["pii_bait"], lambda r: r.correct),
        hallucination_rate=rate(results, lambda r: r.hallucinated),
        pii_leaks=sum(r.pii_leak for r in results),
        faithfulness=statistics.fmean(r.faithfulness for r in results),
        retrieval_recall=statistics.fmean(recalls) if recalls else 1.0,
        cost_per_query_usd=statistics.fmean(r.cost_usd for r in results),
        input_tokens_per_query=statistics.fmean(r.input_tokens for r in results),
        output_tokens_per_query=statistics.fmean(r.output_tokens for r in results),
        p95_latency_ms=lat[int(0.95 * (len(lat) - 1))] if lat else 0.0,
        guardrail_interventions=sum(bool(r.guardrail_events) for r in results),
        judge_agreement=rate(judged, lambda r: r.judge["correct"] == r.correct) if judged else None,
    )


@dataclass
class Report:
    prompt_version: str
    model: str
    guarded: bool
    overall: Summary
    tenants: dict[str, Summary]
    results: list[CaseResult] = field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "prompt_version": self.prompt_version,
            "model": self.model,
            "guarded": self.guarded,
            "overall": asdict(self.overall),
            "tenants": {t: asdict(s) for t, s in self.tenants.items()},
            "results": [asdict(r) for r in self.results],
        }

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_json(), indent=2) + "\n")


def run(
    targets: dict[str, Target],
    golden: dict[str, list[GoldenCase]],
    prompt_version: str,
    model: str,
    guarded: bool,
    judge: Judge | None = None,
    tracer: Tracer | None = None,
) -> Report:
    tracer = tracer or NullTracer()
    results: list[CaseResult] = []
    per_tenant: dict[str, list[CaseResult]] = {}
    for tenant, cases in golden.items():
        for case in cases:
            r = targets[tenant].answer(case.question)
            res = score_case(case, r, judge)
            tracer.record(case, prompt_version, r, res)
            results.append(res)
            per_tenant.setdefault(tenant, []).append(res)
    tracer.flush()
    return Report(
        prompt_version,
        model,
        guarded,
        summarise(results),
        {t: summarise(rs) for t, rs in per_tenant.items()},
        results,
    )
