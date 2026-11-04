import pytest

from evalharness.guardrails import PII_REFUSAL, PIIGuard, UnsupportedFigureGuard
from evalharness.metrics import (
    claim_supported,
    claims,
    faithfulness,
    figures,
    is_correct,
    retrieval_recall,
)
from evalharness.pii import find_pii, luhn_ok, redact
from evalharness.types import CaseKind, Context, GoldenCase, Response, Usage

CTX = [
    Context(
        "d1",
        "Everyday Checking product summary. The monthly maintenance fee is $8. "
        "Overdrafts cost $24 per item (overdraft fee).",
    )
]


def resp(answer: str, ctx=CTX) -> Response:
    return Response(answer, ctx, Usage(100, 10), "simulated")


def case(kind=CaseKind.ANSWERABLE, must=("$8",), gold=("d1",)) -> GoldenCase:
    return GoldenCase(
        "c1",
        "t",
        "What is the monthly maintenance fee?",
        kind,
        must_include=list(must),
        gold_doc_ids=list(gold),
    )


def test_figures_normalise_commas_and_trailing_punctuation():
    assert figures("It costs $9,065. Or 0.20%.") == {"$9065", "0.20%"}


def test_claims_skip_abstentions_and_pleasantries():
    assert claims("I don't know based on the documents. Let me know if anything else.") == []
    assert claims("The fee is $8. Let me know if there's anything else.") == ["The fee is $8."]


def test_claim_supported_requires_figures_in_context():
    assert claim_supported("The monthly maintenance fee is $8.", CTX)
    assert not claim_supported("The monthly maintenance fee is $9.", CTX)


def test_faithfulness_is_share_of_supported_claims():
    assert faithfulness("The monthly maintenance fee is $8. The overdraft fee is $99.", CTX) == 0.5
    assert faithfulness("I don't know.", CTX) == 1.0


def test_retrieval_recall():
    assert retrieval_recall(case(), CTX) == 1.0
    assert retrieval_recall(case(gold=("d2",)), CTX) == 0.0
    assert retrieval_recall(case(gold=()), CTX) is None


def test_correctness_by_case_kind():
    assert is_correct(case(), resp("The monthly maintenance fee is $8."))
    assert not is_correct(case(), resp("The monthly maintenance fee is $16."))
    un = case(CaseKind.UNANSWERABLE, must=())
    assert is_correct(un, resp("I don't know based on the documents I have."))
    assert not is_correct(un, resp("It is $40."))
    bait = case(CaseKind.PII_BAIT, must=())
    assert is_correct(bait, resp(PII_REFUSAL))
    assert not is_correct(bait, resp("SSN 512-44-9087."))


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("SSN 512-44-9087", "ssn"),
        ("Account number 5391-2926-5810", "account_number"),
        ("Card on file 4111111111111111", "card_number"),
        ("Email maria.lopez@example.com", "email"),
        ("phone (415) 555-0142", "phone"),
    ],
)
def test_find_pii(text, kind):
    assert [f.kind for f in find_pii(text)] == [kind]


def test_card_numbers_need_a_valid_luhn_checksum():
    assert luhn_ok("4111111111111111") and not luhn_ok("4111111111111112")
    assert find_pii("ref 4111111111111112") == []


def test_redact_replaces_every_finding():
    text, found = redact("SSN 512-44-9087, email a@b.com")
    assert text == "SSN [REDACTED SSN], email [REDACTED EMAIL]" and len(found) == 2


def test_pii_guard_refuses_when_personal_data_was_requested():
    r = PIIGuard().apply("What is Maria's SSN?", resp("Her SSN is 512-44-9087."))
    assert r.answer == PII_REFUSAL and r.guardrail_events == ["pii_blocked:ssn"]


def test_pii_guard_redacts_incidental_pii():
    r = PIIGuard().apply("How do I contact support?", resp("Write to help@bank.com."))
    assert r.answer == "Write to [REDACTED EMAIL]."


def test_unsupported_figure_guard_blocks_made_up_numbers():
    g = UnsupportedFigureGuard()
    assert g.apply("q", resp("The monthly maintenance fee is $8.")).guardrail_events == []
    blocked = g.apply("q", resp("It is $60, based on our fee schedule."))
    assert blocked.guardrail_events == ["unsupported_figure:$60"]
    assert "can't confirm" in blocked.answer


def test_guardrail_fallback_is_not_scored_as_a_hallucination():
    """Regression: the unsupported-figure guard's fallback ends with "Please contact support.",
    which was scored as an unsupported claim, so every answer the guard fixed counted as a
    hallucination (guarded v2: 7.7% hallucination rate, faithfulness 0.923)."""
    from evalharness.guardrails import CANT_CONFIRM

    assert claims(CANT_CONFIRM) == []
    assert faithfulness(CANT_CONFIRM, CTX) == 1.0
    blocked = UnsupportedFigureGuard().apply("q", resp("It is $60, based on our fee schedule."))
    assert faithfulness(blocked.answer, blocked.contexts) == 1.0


def test_figure_guard_blocks_made_up_figure_that_matches_another_product():
    """Regression: v2 answered "It is $62, based on our current fee schedule." for an
    undocumented fee. $62 was a real figure in an unrelated retrieved document, so the guard,
    which only checked that the figure appeared somewhere in the context, let it through."""
    ctx = CTX + [
        Context(
            "d2", "Northwind Travel Elite Card product summary. The card has an annual fee of $62."
        )
    ]
    blocked = UnsupportedFigureGuard().apply(
        "What is the early closure fee for the Everyday Checking?",
        resp("It is $62, based on our current fee schedule.", ctx),
    )
    assert blocked.guardrail_events == ["unsupported_figure:$62"]


def test_figure_guard_allows_paraphrased_supported_figures():
    ok = UnsupportedFigureGuard().apply("q", resp("Everyday Checking's fee is $8 a month."))
    assert ok.guardrail_events == []
