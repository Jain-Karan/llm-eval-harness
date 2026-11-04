"""LLM-as-judge.

The reference metrics in `metrics.py` are cheap, deterministic and gate CI. An LLM judge adds a
second opinion on correctness and faithfulness against the reference answer, for wording the
heuristics can't read. `runner` reports how often the judge and the heuristics agree, which is the
first thing to check before trusting either.
"""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, Field

from .types import GoldenCase, Response

RUBRIC = """You grade a customer-support answer for a financial services company.
Use only the CONTEXT and the REFERENCE. Decide:
- correct: does the answer give the same facts as the reference? If the reference says the
  information is unavailable or must not be shared, a correct answer declines.
- faithful: is every factual claim in the answer supported by the context?
Return a one-sentence rationale."""


class Verdict(BaseModel):
    correct: bool
    faithful: bool
    rationale: str = Field(default="")


class Judge(Protocol):
    name: str

    def grade(self, case: GoldenCase, r: Response) -> Verdict: ...


class LLMJudge:
    def __init__(self, chat, name: str) -> None:  # noqa: ANN001 - any LangChain chat model
        self.chain = chat.with_structured_output(Verdict)
        self.name = name

    def grade(self, case: GoldenCase, r: Response) -> Verdict:
        ctx = "\n".join(f"- {c.text}" for c in r.contexts)
        return self.chain.invoke(
            [
                ("system", RUBRIC),
                (
                    "human",
                    f"CONTEXT:\n{ctx}\n\nQUESTION: {case.question}\n"
                    f"REFERENCE: {case.reference}\nANSWER: {r.answer}",
                ),
            ]
        )


def make_judge(kind: str | None, model: str = "claude-sonnet-4-5") -> Judge | None:
    if not kind:
        return None
    if kind == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return LLMJudge(ChatAnthropic(model=model, temperature=0, max_tokens=300), model)
    raise ValueError(f"unknown judge {kind!r}")
