"""Per-query cost from token usage.

Prices are USD per million tokens and are configuration, not facts baked into results: check
your provider's current price list and pass your own table. The simulated model is priced like
the default Claude model so cost comparisons between prompt versions are meaningful offline.
"""

from __future__ import annotations

from .types import Usage

DEFAULT_PRICES: dict[str, tuple[float, float]] = {
    # model: (input $/MTok, output $/MTok)
    "claude-sonnet-4-5": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
}
SIMULATED_AS = "claude-sonnet-4-5"


def cost_usd(
    model: str, usage: Usage, prices: dict[str, tuple[float, float]] | None = None
) -> float:
    prices = prices or DEFAULT_PRICES
    key = SIMULATED_AS if model == "simulated" else model
    if key not in prices:
        raise KeyError(f"no price for model {model!r}; pass a price table")
    pin, pout = prices[key]
    return (usage.input_tokens * pin + usage.output_tokens * pout) / 1_000_000
