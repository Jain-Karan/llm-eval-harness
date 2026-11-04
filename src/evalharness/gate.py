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
    # Guardrails are a safety net, not a substitute for a working prompt. If they suddenly fire
    # far more often, the model is producing what they block, and the guarded output alone
    # would hide it.
    max_guardrail_rate_increase: float = 0.02  # absolute, share of answers


@dataclass
class GateResult:
    passed: bool
    failures: list[str]
    rows: list[tuple[str, str, float, float]]  # scope, metric, baseline, candidate
    regressed: list[tuple[str, str]] = field(default_factory=list)  # (case_id, why)
    cases: int = 0

    def markdown(self) -> str:
        head = "PASS" if self.passed else "FAIL"
        out = [f"### Eval gate: {head}", ""]
        if self.failures:
            out += [f"- {f}" for f in self.failures] + [""]
        out += ["| scope | metric | baseline | candidate | change |", "|---|---|---|---|---|"]
        for scope, metric, b, c in self.rows:
            out.append(f"| {scope} | {metric} | {b:.4g} | {c:.4g} | {c - b:+.4g} |")
        if self.regressed:
            share = len(self.regressed) / self.cases if self.cases else 0
            out += [
                "",
                f"**{len(self.regressed)} of {self.cases} answers ({share:.1%}) "
                "regressed or needed a guardrail:**",
                "",
            ]
            out += [f"- `{cid}`: {why}" for cid, why in self.regressed[:50]]
            if len(self.regressed) > 50:
                out.append(f"- ... and {len(self.regressed) - 50} more")
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
    for scope, b, c in scopes:
        br = b["guardrail_interventions"] / b["cases"] if b.get("cases") else 0.0
        cr = c["guardrail_interventions"] / c["cases"] if c.get("cases") else 0.0
        rows.append((scope, "guardrail_rate", br, cr))
        if cr - br > cfg.max_guardrail_rate_increase + 1e-9:
            failures.append(
                f"{scope}: guardrails fired on {cr:.1%} of answers, up from {br:.1%}"
                f" (max +{cfg.max_guardrail_rate_increase:.0%})"
            )

    regressed = case_diff(base.get("results", []), cand.get("results", []))
    return GateResult(not failures, failures, rows, regressed, len(cand.get("results", [])))


def case_diff(base: list[dict], cand: list[dict]) -> list[tuple[str, str]]:
    """Cases that got worse: right before and wrong now, newly unfaithful, newly leaking PII, or
    newly caught by a guardrail."""
    before = {r["case_id"]: r for r in base}
    out = []
    for r in cand:
        b = before.get(r["case_id"])
        if b is None:
            continue
        why = []
        if b["correct"] and not r["correct"]:
            why.append("was correct, now wrong")
        if r["hallucinated"] and not b["hallucinated"]:
            why.append("unsupported claim")
        if r["pii_leak"] and not b["pii_leak"]:
            why.append("PII leak")
        if r["guardrail_events"] and not b["guardrail_events"]:
            why.append("guardrail: " + ", ".join(r["guardrail_events"]))
        if why:
            out.append((r["case_id"], "; ".join(why)))
    return out
