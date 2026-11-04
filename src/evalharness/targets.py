"""Systems under test. Anything with `answer(question) -> Response` can be evaluated.

- `AppTarget`: the bundled demo RAG app (offline)
- `HttpTarget`: any HTTP endpoint returning answer, contexts and usage. The JSON field names are
  configurable, so an existing service can be evaluated without changing its API.
- `Guarded`: wraps a target with the response-boundary guardrails, exactly as production would.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Protocol

import httpx

from .app import RAGApp
from .types import Context, Response, Usage


class Target(Protocol):
    def answer(self, question: str) -> Response: ...


class AppTarget:
    def __init__(self, app: RAGApp) -> None:
        self.app = app

    def answer(self, question: str) -> Response:
        return self.app.answer(question)


class HttpTarget:
    def __init__(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        question_field: str = "question",
        answer_field: str = "answer",
        contexts_field: str = "contexts",
        model: str = "unknown",
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.url = url
        self.fields = (question_field, answer_field, contexts_field)
        self.model = model
        self.client = httpx.Client(headers=headers or {}, timeout=60, transport=transport)

    def answer(self, question: str) -> Response:
        qf, af, cf = self.fields
        t0 = time.perf_counter()
        r = self.client.post(self.url, json={qf: question})
        r.raise_for_status()
        d = r.json()
        ctx = [
            Context(str(c.get("doc_id", i)), c.get("text", "")) for i, c in enumerate(d.get(cf, []))
        ]
        u = d.get("usage", {})
        return Response(
            d[af],
            ctx,
            Usage(u.get("input_tokens", 0), u.get("output_tokens", 0)),
            d.get("model", self.model),
            (time.perf_counter() - t0) * 1000,
        )


class Guarded:
    def __init__(self, target: Target, guards: Sequence) -> None:
        self.target, self.guards = target, list(guards)

    def answer(self, question: str) -> Response:
        r = self.target.answer(question)
        for g in self.guards:
            r = g.apply(question, r)
        return r
