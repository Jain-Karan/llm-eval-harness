from evalharness.cli import evaluate
from evalharness.gate import GateConfig, check
from evalharness.synthetic import build_tenant
from evalharness.tracing import LangfuseTracer


def _summary(**kw):
    base = dict(
        accuracy=0.9,
        answer_accuracy=0.95,
        abstention_accuracy=0.5,
        pii_refusal_rate=1.0,
        faithfulness=1.0,
        retrieval_recall=1.0,
        hallucination_rate=0.0,
        pii_leaks=0,
        cost_per_query_usd=0.002,
    )
    return {**base, **kw}


def _report(**kw):
    return {"overall": _summary(**kw), "tenants": {"t1": _summary(**kw)}}


def test_identical_reports_pass():
    assert check(_report(), _report()).passed


def test_pii_leak_fails_even_if_metrics_improve():
    r = check(_report(), _report(pii_leaks=1, accuracy=0.99))
    assert not r.passed and any("PII" in f for f in r.failures)


def test_drop_within_tolerance_passes_and_beyond_fails():
    assert check(_report(), _report(accuracy=0.885)).passed
    assert not check(_report(), _report(accuracy=0.87)).passed


def test_per_tenant_regression_fails_even_when_overall_is_flat():
    base = {"overall": _summary(), "tenants": {"a": _summary(), "b": _summary()}}
    cand = {"overall": _summary(), "tenants": {"a": _summary(accuracy=0.8), "b": _summary()}}
    r = check(base, cand)
    assert not r.passed and r.failures == ["a: accuracy 0.900 -> 0.800 (max drop 0.02)"]


def test_cost_increase_limit():
    assert not check(_report(), _report(cost_per_query_usd=0.0025)).passed
    assert check(
        _report(), _report(cost_per_query_usd=0.0025), GateConfig(max_cost_increase=0.5)
    ).passed


def test_gate_catches_the_v2_prompt_regression():
    r = check(evaluate("v1").to_json(), evaluate("v2").to_json())
    assert not r.passed
    assert "### Eval gate: FAIL" in r.markdown()


def test_langfuse_tracer_emits_nested_observations():
    from langfuse import Langfuse
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    lf = Langfuse(
        public_key="pk-lf-test",
        secret_key="sk-lf-test",
        base_url="http://127.0.0.1:9",
        span_exporter=exporter,
    )
    scores = []
    lf.score_current_trace = lambda **kw: scores.append(kw)  # scores go over HTTP; record them

    docs, cases = build_tenant("acme-wealth")
    from evalharness.app import Prompt, RAGApp, SimulatedModel
    from evalharness.runner import run
    from evalharness.targets import AppTarget

    app = RAGApp(docs, Prompt.load("v1"), SimulatedModel())
    run(
        {"acme-wealth": AppTarget(app)},
        {"acme-wealth": cases[:2]},
        "v1",
        "simulated",
        False,
        tracer=LangfuseTracer(lf),
    )
    names = [s.name for s in exporter.get_finished_spans()]
    assert names.count("eval-case") == 2
    assert {"retrieve", "generate"} <= set(names)
    assert {s["name"] for s in scores} == {"faithfulness", "correct", "hallucinated", "pii_leak"}
