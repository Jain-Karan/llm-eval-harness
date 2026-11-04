"""The demo system under test: a small RAG support assistant, one per tenant.

The harness doesn't care what it evaluates (see `targets.py` for HTTP targets). This app exists
so the harness has something realistic to evaluate offline: BM25 retrieval over the tenant's
documents, a versioned prompt (`prompts/v*.yaml`), and a model.

Models
- `SimulatedModel`: deterministic and offline, for CI. It reads the system prompt the way an
  instruction-following model would, and its failure modes are triggered by what the prompt
  leaves out: without the grounding instruction it answers unsupported questions with a
  plausible, made-up figure; without the personal-data instruction it repeats PII from the
  context. That makes prompt regressions *injectable*, and the harness has to catch them.
- `ChatModel`: any LangChain chat model; `make_model("anthropic")` gives Claude.
"""

from __future__ import annotations

import hashlib
import math
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import yaml

from .synthetic import Doc
from .types import Context, Response, Usage

PROMPTS_DIR = Path(__file__).with_name("prompts")
_WORD = re.compile(r"[a-z0-9]+(?:[.'%-][a-z0-9]+)*")
STOP = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "have",
        "how",
        "i",
        "in",
        "is",
        "it",
        "its",
        "me",
        "my",
        "of",
        "on",
        "or",
        "the",
        "this",
        "to",
        "was",
        "what",
        "whats",
        "what's",
        "which",
        "with",
        "you",
        "your",
        "can",
        "tell",
        "much",
        "does",
        "do",
        "give",
        "s",
    ]
)


ABBREV = {"apy": "annual percentage yield", "apr": "annual percentage rate"}


def _stem(w: str) -> str:
    for suf in ("ed", "es", "s"):
        if len(w) > 4 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def words(text: str) -> list[str]:
    t = text.lower()
    for short, long in ABBREV.items():
        t = re.sub(rf"\b{short}\b", f"{short} {long}", t)
    return [_stem(w) for w in _WORD.findall(t) if w not in STOP]


def n_tokens(text: str) -> int:
    """~4 characters per token: close to Claude and GPT tokenizers on English prose."""
    return max(1, math.ceil(len(text) / 4))


@dataclass
class Prompt:
    version: str
    system: str
    top_k: int

    @classmethod
    def load(cls, version_or_path: str) -> Prompt:
        p = Path(version_or_path)
        if not p.exists():
            p = PROMPTS_DIR / f"{version_or_path}.yaml"
        d = yaml.safe_load(p.read_text())
        return cls(d["version"], d["system"].strip(), int(d.get("top_k", 6)))


class BM25:
    def __init__(self, docs: list[Doc], k1: float = 1.2, b: float = 0.75) -> None:
        self.docs = docs
        self.tf = [Counter(words(f"{d.title} {d.text}")) for d in docs]
        self.len = [sum(t.values()) for t in self.tf]
        self.avg = sum(self.len) / len(self.len)
        self.df = Counter(w for t in self.tf for w in t)
        self.k1, self.b = k1, b

    def search(self, query: str, k: int) -> list[Doc]:
        n = len(self.docs)
        scores = []
        for i, tf in enumerate(self.tf):
            s = 0.0
            for w in set(words(query)):
                if w in tf:
                    idf = math.log(1 + (n - self.df[w] + 0.5) / (self.df[w] + 0.5))
                    norm = tf[w] + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg)
                    s += idf * tf[w] * (self.k1 + 1) / norm
            scores.append((s, i))
        scores.sort(key=lambda x: (-x[0], x[1]))
        return [self.docs[i] for s, i in scores[:k] if s > 0]


# --- models -------------------------------------------------------------------------------

PII_ASK = re.compile(r"\b(account number|ssn|social security|card number|email|phone)\b", re.I)
_SENT = re.compile(r"(?<=[.!?])\s+")


class SimulatedModel:
    name = "simulated"

    def __init__(self, support_threshold: float = 0.5) -> None:
        self.threshold = support_threshold

    @staticmethod
    def _follows(system: str) -> dict[str, bool]:
        s = system.lower()
        return {
            "grounded": "answer only from the context" in s and "don't know" in s,
            "no_pii": "never include account numbers" in s,
            "concise": "be concise" in s,
        }

    @staticmethod
    def _fabricate(question: str) -> str:
        h = int(hashlib.sha256(question.encode()).hexdigest()[:8], 16)
        return f"${10 + h % 90}"

    def generate(self, system: str, contexts: list[Context], question: str) -> tuple[str, Usage]:
        rules = self._follows(system)
        q = set(words(question))
        best, best_score = "", 0.0
        for c in contexts:
            sents = _SENT.split(c.text)
            head = set(words(sents[0]))  # "<Product> product summary." names the product
            product = len(q & head) / (len(head) or 1)
            for sent in sents[1:] or sents:
                attr = q - head
                hit = len(attr & set(words(sent))) / (len(attr) or 1)
                # an instruction-following model needs both the right product and the right fact
                score = min(1.0, product) * 0.5 + hit * 0.5
                if score > best_score:
                    best, best_score = sent, score
        if PII_ASK.search(question):
            crm = next(
                (
                    c
                    for c in contexts
                    if "Customer profile" in c.text and q & set(words(c.text.split(".")[0]))
                ),
                None,
            )
            if crm:
                answer = (
                    "I can't share personal information about customers."
                    if rules["no_pii"]
                    else crm.text
                )
            else:
                answer = (
                    "I don't know based on the documents I have."
                    if rules["grounded"]
                    else f"The details on file are {self._fabricate(question)}."
                )
        elif best_score >= self.threshold:
            answer = best
        elif rules["grounded"]:
            answer = "I don't know based on the documents I have."
        else:
            answer = f"It is {self._fabricate(question)}, based on our current fee schedule."
        if not rules["concise"]:
            answer += " Let me know if there is anything else I can help you with today."
        prompt_text = system + "".join(c.text for c in contexts) + question
        return answer, Usage(n_tokens(prompt_text), n_tokens(answer))


class ChatModel:
    """Any LangChain chat model. Token usage comes from the provider's usage metadata."""

    def __init__(self, chat, name: str) -> None:  # noqa: ANN001
        self.chat, self.name = chat, name

    def generate(self, system: str, contexts: list[Context], question: str) -> tuple[str, Usage]:
        ctx = "\n\n".join(f"[{i}] {c.text}" for i, c in enumerate(contexts, 1))
        msg = self.chat.invoke(
            [("system", system), ("human", f"Context:\n{ctx}\n\nQuestion: {question}")]
        )
        um = getattr(msg, "usage_metadata", None) or {}
        return str(msg.content), Usage(um.get("input_tokens", 0), um.get("output_tokens", 0))


def make_model(kind: str = "simulated", model: str = "claude-sonnet-4-5"):  # noqa: ANN201
    if kind == "simulated":
        return SimulatedModel()
    if kind == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatModel(ChatAnthropic(model=model, temperature=0, max_tokens=300), model)
    raise ValueError(f"unknown model {kind!r}")


class RAGApp:
    def __init__(self, docs: list[Doc], prompt: Prompt, model) -> None:  # noqa: ANN001
        self.retriever = BM25(docs)
        self.prompt = prompt
        self.model = model

    def answer(self, question: str) -> Response:
        t0 = time.perf_counter()
        hits = self.retriever.search(question, self.prompt.top_k)
        contexts = [Context(d.doc_id, d.text) for d in hits]
        text, usage = self.model.generate(self.prompt.system, contexts, question)
        return Response(
            text, contexts, usage, self.model.name, latency_ms=(time.perf_counter() - t0) * 1000
        )
