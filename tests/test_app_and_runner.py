import json

import httpx

from evalharness.app import BM25, Prompt, RAGApp, SimulatedModel, words
from evalharness.cli import evaluate, main
from evalharness.synthetic import build_all, build_tenant
from evalharness.targets import HttpTarget
from evalharness.tracing import JsonlTracer
from evalharness.types import CaseKind, load_golden, save_golden


def test_golden_sets_are_seeded_and_mixed():
    docs, cases = build_tenant("northwind-bank")
    again = build_tenant("northwind-bank")[1]
    assert [c.question for c in cases] == [c.question for c in again]
    kinds = {c.kind for c in cases}
    assert kinds == {CaseKind.ANSWERABLE, CaseKind.UNANSWERABLE, CaseKind.PII_BAIT}
    figs = [f for c in cases for f in c.must_include]
    assert len(figs) == len(set(figs))  # every figure is unique within the tenant


def test_golden_roundtrip(tmp_path):
    _, cases = build_tenant("acme-wealth")
    save_golden(cases, tmp_path / "g.jsonl")
    assert load_golden(tmp_path / "g.jsonl") == cases


def test_words_expand_abbreviations_and_stem():
    assert "yield" in words("What's the APY?")
    assert "launch" in words("The fund launched in 2019.")


def test_bm25_finds_the_product_doc():
    docs, _ = build_tenant("acme-wealth")
    hit = BM25(docs).search("expense ratio of the Acme Global Equity Fund", 1)[0]
    assert "Acme Global Equity Fund" in hit.text


def test_prompt_versions_load():
    assert Prompt.load("v1").top_k == 6 and Prompt.load("v3").top_k == 3
    assert "don't know" in Prompt.load("v1").system


def test_simulated_model_follows_prompt_rules():
    docs, cases = build_tenant("northwind-bank")
    bait = next(c for c in cases if c.kind == CaseKind.PII_BAIT)
    v1 = RAGApp(docs, Prompt.load("v1"), SimulatedModel()).answer(bait.question)
    v2 = RAGApp(docs, Prompt.load("v2"), SimulatedModel()).answer(bait.question)
    assert "can't share" in v1.answer
    assert "SSN" in v2.answer  # without the personal-data rule, it leaks


def test_v3_cuts_cost_without_losing_quality():
    base, cand = evaluate("v1").overall, evaluate("v3").overall
    assert cand.cost_per_query_usd < base.cost_per_query_usd * 0.7
    assert cand.accuracy == base.accuracy and cand.pii_leaks == 0


def test_jsonl_trace_has_one_line_per_case(tmp_path):
    rep = evaluate("v1", tracer=JsonlTracer(tmp_path / "t.jsonl"))
    lines = (tmp_path / "t.jsonl").read_text().splitlines()
    assert len(lines) == rep.overall.cases
    first = json.loads(lines[0])
    assert {"retrieval", "generation", "scores", "guardrails"} <= first.keys()


def test_http_target_maps_fields():
    def handler(req: httpx.Request) -> httpx.Response:
        assert json.loads(req.content) == {"q": "fee?"}
        return httpx.Response(
            200,
            json={
                "reply": "The fee is $8.",
                "passages": [{"doc_id": "d1", "text": "fee is $8"}],
                "usage": {"input_tokens": 50, "output_tokens": 6},
                "model": "claude-haiku-4-5",
            },
        )

    t = HttpTarget(
        "http://svc/ask",
        question_field="q",
        answer_field="reply",
        contexts_field="passages",
        transport=httpx.MockTransport(handler),
    )
    r = t.answer("fee?")
    assert r.answer == "The fee is $8." and r.contexts[0].doc_id == "d1"
    assert r.usage.input_tokens == 50 and r.model == "claude-haiku-4-5"


def test_cli_run_and_gate(tmp_path, capsys):
    assert main(["run", "--prompt", "v1", "--out", str(tmp_path / "base.json")]) == 0
    assert main(["run", "--prompt", "v2", "--out", str(tmp_path / "v2.json")]) == 0
    assert (
        main(
            [
                "gate",
                "--baseline",
                str(tmp_path / "base.json"),
                "--candidate",
                str(tmp_path / "v2.json"),
            ]
        )
        == 1
    )
    assert "PII leaks" in capsys.readouterr().out


def test_build_golden_writes_files(tmp_path):
    assert main(["build-golden", "--out", str(tmp_path)]) == 0
    _, cases = build_all()
    for t, cs in cases.items():
        assert len(load_golden(tmp_path / f"{t}.jsonl")) == len(cs)
