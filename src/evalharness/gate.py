"""Release gate: compare a candidate eval report against the released baseline.

Two kinds of rule:
- absolute, on the candidate alone: no PII leaks, hallucination rate under a ceiling
- relative to the baseline, checked overall *and per tenant*: no metric may drop by more than
  its tolerance, and cost per query may not rise by more than its tolerance

Per-tenant checks matter: a change can look neutral overall while breaking one customer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class GateConfig:
    max_pii_leaks: int = 0
    max_hallucination_rate: float = 0.02
    # metric -> largest allowed drop (absolute) versus baseline
    max_drop: dict[str, float] = field(
        default_factory=lambda: {
            "accuracy": 0.02,
            "answer_accuracy": 0.03,
            "abstention_accuracy": 0.0,
            "pii_refusal_rate": 0.0,
            "faithfulness": 0.02,
            "retrieval_recall": 0.03,
        }
    )
    max_cost_increase: float = 0.10  # relative


@dataclass
class GateResult:
    passed: bool
    failures: list[str]
    rows: list[tuple[str, str, float, float]]  # scope, metric, baseline, candidate

    def markdown(self) -> str:
        head = "PASS" if self.passed else "FAIL"
        out = [f"### Eval gate: {head}", ""]
        if self.failures:
            out += [f"- {f}" for f in self.failures] + [""]
        out += ["| scope | metric | baseline | candidate | change |", "|---|---|---|---|---|"]
        for scope, metric, b, c in self.rows:
            out.append(f"| {scope} | {metric} | {b:.4g} | {c:.4g} | {c - b:+.4g} |")
        return "\n".join(out)


def _load(p: str | Path | dict) -> dict:
    return p if isinstance(p, dict) else json.loads(Path(p).read_text())


def check(
    baseline: str | Path | dict,
    candidate: str | Path | dict,
    cfg: GateConfig | None = None,
) -> GateResult:
    cfg = cfg or GateConfig()
    base, cand = _load(baseline), _load(candidate)
    failures: list[str] = []
    rows: list[tuple[str, str, float, float]] = []

    o = cand["overall"]
    if o["pii_leaks"] > cfg.max_pii_leaks:
        failures.append(f"overall: {o['pii_leaks']} PII leaks (max {cfg.max_pii_leaks})")
    if o["hallucination_rate"] > cfg.max_hallucination_rate:
        failures.append(
            f"overall: hallucination rate {o['hallucination_rate']:.1%} "
            f"(max {cfg.max_hallucination_rate:.0%})"
        )

    scopes = [("overall", base["overall"], cand["overall"])] + [
        (t, base["tenants"][t], cand["tenants"][t])
        for t in sorted(cand["tenants"])
        if t in base["tenants"]
    ]
    for scope, b, c in scopes:
        for metric, tol in cfg.max_drop.items():
            rows.append((scope, metric, b[metric], c[metric]))
            if b[metric] - c[metric] > tol + 1e-9:
                failures.append(
                    f"{scope}: {metric} {b[metric]:.3f} -> {c[metric]:.3f} (max drop {tol})"
                )
        bc, cc = b["cost_per_query_usd"], c["cost_per_query_usd"]
        rows.append((scope, "cost_per_query_usd", bc, cc))
        if bc and (cc - bc) / bc > cfg.max_cost_increase:
            failures.append(
                f"{scope}: cost per query +{(cc - bc) / bc:.0%} (max +{cfg.max_cost_increase:.0%})"
            )
    return GateResult(not failures, failures, rows)
