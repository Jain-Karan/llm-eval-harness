"""`evalh` command line.

evalh build-golden                            # write golden sets to golden/<tenant>.jsonl
evalh run --prompt v1 --out reports/v1.json   # evaluate a prompt version
evalh run --prompt v2 --guarded               # ... with response-boundary guardrails
evalh gate --baseline reports/baseline.json --candidate reports/v2.json
evalh run --model anthropic --judge anthropic # real Claude + LLM judge (needs a key)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .app import Prompt, RAGApp, make_model
from .gate import check
from .guardrails import DEFAULT_GUARDS
from .judge import make_judge
from .runner import run
from .synthetic import build_all
from .targets import AppTarget, Guarded
from .tracing import JsonlTracer, LangfuseTracer, NullTracer
from .types import load_golden, save_golden


def _golden(path: str | None, seed: int) -> dict:
    docs, cases = build_all(seed)
    if path:
        return docs, {p.stem: load_golden(p) for p in sorted(Path(path).glob("*.jsonl"))}
    return docs, cases


def cmd_build_golden(args: argparse.Namespace) -> int:
    _, cases = build_all(args.seed)
    for tenant, cs in cases.items():
        save_golden(cs, Path(args.out) / f"{tenant}.jsonl")
        print(f"{tenant}: {len(cs)} cases")
    return 0


def evaluate(
    prompt: str,
    model: str = "simulated",
    guarded: bool = False,
    judge: str | None = None,
    golden_dir: str | None = None,
    seed: int = 7,
    tracer=None,
):  # noqa: ANN001, ANN201
    docs, golden = _golden(golden_dir, seed)
    p = Prompt.load(prompt)
    m = make_model(model)
    targets = {}
    for tenant in golden:
        t = AppTarget(RAGApp(docs[tenant], p, m))
        targets[tenant] = Guarded(t, DEFAULT_GUARDS) if guarded else t
    return run(targets, golden, p.version, m.name, guarded, make_judge(judge), tracer)


def cmd_run(args: argparse.Namespace) -> int:
    tracer = NullTracer()
    if args.trace == "langfuse":
        tracer = LangfuseTracer()
    elif args.trace:
        tracer = JsonlTracer(args.trace)
    rep = evaluate(
        args.prompt, args.model, args.guarded, args.judge, args.golden, args.seed, tracer
    )
    o = rep.overall
    print(f"prompt {rep.prompt_version}  model {rep.model}  guarded {rep.guarded}  cases {o.cases}")
    for name in (
        "accuracy",
        "answer_accuracy",
        "abstention_accuracy",
        "pii_refusal_rate",
        "hallucination_rate",
        "faithfulness",
        "retrieval_recall",
    ):
        print(f"  {name:22} {getattr(o, name):.3f}")
    print(f"  {'pii_leaks':22} {o.pii_leaks}")
    print(f"  {'cost_per_query_usd':22} {o.cost_per_query_usd:.6f}")
    if args.out:
        rep.save(args.out)
    return 0


def cmd_gate(args: argparse.Namespace) -> int:
    res = check(args.baseline, args.candidate)
    md = res.markdown()
    print(md)
    if args.summary:
        with open(args.summary, "a") as f:
            f.write(md + "\n")
    return 0 if res.passed else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="evalh")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("build-golden")
    p.add_argument("--out", default="golden")
    p.add_argument("--seed", type=int, default=7)
    p.set_defaults(fn=cmd_build_golden)

    p = sub.add_parser("run")
    p.add_argument("--prompt", default="v1", help="prompt version or path to a prompt YAML")
    p.add_argument("--model", default="simulated", choices=["simulated", "anthropic"])
    p.add_argument("--judge", default=None, choices=["anthropic"])
    p.add_argument("--guarded", action="store_true", help="apply response guardrails")
    p.add_argument("--golden", default=None, help="directory of <tenant>.jsonl golden sets")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--trace", default=None, help="'langfuse' or a .jsonl path")
    p.add_argument("--out", default=None)
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("gate")
    p.add_argument("--baseline", required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--summary", default=None, help="append Markdown here (GITHUB_STEP_SUMMARY)")
    p.set_defaults(fn=cmd_gate)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
