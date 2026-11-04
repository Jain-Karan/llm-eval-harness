"""Tracing: every eval case becomes a trace with retrieval, generation and guardrail steps plus
its scores. `JsonlTracer` writes locally (CI artefact); `LangfuseTracer` sends the same structure
to Langfuse, where per-query cost and scores can be filtered by tenant and prompt version.
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Protocol

from .types import CaseResult, GoldenCase, Response


class Tracer(Protocol):
    def record(
        self, case: GoldenCase, prompt_version: str, r: Response, result: CaseResult
    ) -> None: ...

    def flush(self) -> None: ...


class NullTracer:
    def record(self, *a: Any) -> None:
        pass

    def flush(self) -> None:
        pass


class JsonlTracer:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = self.path.open("w")

    def record(
        self, case: GoldenCase, prompt_version: str, r: Response, result: CaseResult
    ) -> None:
        self._f.write(
            json.dumps(
                {
                    "trace": case.case_id,
                    "tenant": case.tenant,
                    "prompt": prompt_version,
                    "question": case.question,
                    "retrieval": [c.doc_id for c in r.contexts],
                    "generation": {
                        "model": r.model,
                        "input_tokens": r.usage.input_tokens,
                        "output_tokens": r.usage.output_tokens,
                        "cost_usd": result.cost_usd,
                    },
                    "guardrails": r.guardrail_events,
                    "answer": r.answer,
                    "scores": {
                        "faithfulness": result.faithfulness,
                        "correct": result.correct,
                        "hallucinated": result.hallucinated,
                        "pii_leak": result.pii_leak,
                        "retrieval_recall": result.retrieval_recall,
                    },
                }
            )
            + "\n"
        )

    def flush(self) -> None:
        self._f.flush()


class LangfuseTracer:
    """Maps each case to a Langfuse trace: chain -> retriever, generation, guardrail, + scores.
    Credentials come from LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST, or pass a
    configured client (tests pass one with an in-memory exporter)."""

    def __init__(self, client: Any = None) -> None:
        if client is None:
            from langfuse import Langfuse

            if not os.environ.get("LANGFUSE_PUBLIC_KEY"):
                raise RuntimeError("LangfuseTracer needs LANGFUSE_PUBLIC_KEY and SECRET_KEY")
            client = Langfuse()
        self.lf = client

    @contextmanager
    def _obs(self, **kw: Any):  # noqa: ANN202
        with self.lf.start_as_current_observation(**kw) as o:
            yield o

    def record(
        self, case: GoldenCase, prompt_version: str, r: Response, result: CaseResult
    ) -> None:
        meta = {"tenant": case.tenant, "kind": case.kind.value, "case_id": case.case_id}
        with self._obs(
            name="eval-case",
            as_type="chain",
            input=case.question,
            metadata=meta,
            version=prompt_version,
        ) as root:
            with self._obs(
                name="retrieve",
                as_type="retriever",
                input=case.question,
                output=[c.doc_id for c in r.contexts],
            ):
                pass
            with self._obs(
                name="generate",
                as_type="generation",
                model=r.model,
                usage_details={"input": r.usage.input_tokens, "output": r.usage.output_tokens},
                cost_details={"total": result.cost_usd},
                output=r.answer,
            ):
                pass
            if r.guardrail_events:
                with self._obs(
                    name="guardrails",
                    as_type="guardrail",
                    output=r.guardrail_events,
                    level="WARNING",
                ):
                    pass
            root.update(output=r.answer)
            for name, value in (
                ("faithfulness", result.faithfulness),
                ("correct", float(result.correct)),
                ("hallucinated", float(result.hallucinated)),
                ("pii_leak", float(result.pii_leak)),
            ):
                self.lf.score_current_trace(name=name, value=value)

    def flush(self) -> None:
        self.lf.flush()
