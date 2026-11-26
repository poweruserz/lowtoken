"""USD per million tokens, Anthropic first-party list prices (cached 2026-11-03).

Cache writes are 1.25x input (5 minute TTL) or 2x input (1 hour TTL).
"""
from __future__ import annotations

# model prefix -> (input, output, cache_read)
_TABLE = [
    ("claude-fable-5-2", 8.0, 40.0, 0.20),
    ("claude-mythos-5-2", 8.0, 40.0, 0.20),
    ("claude-fable-5", 10.0, 50.0, 1.00),
    ("claude-mythos-5", 10.0, 50.0, 1.00),
    ("claude-opus-5-5", 4.0, 20.0, 0.20),
    ("claude-opus", 5.0, 25.0, 0.50),
    ("claude-sonnet-5-5", 2.0, 10.0, 0.10),
    ("claude-sonnet-5", 2.0, 10.0, 0.20),
    ("claude-sonnet", 3.0, 15.0, 0.30),
    ("claude-haiku-5-5", 0.10, 0.50, 0.01),
    ("claude-haiku", 1.0, 5.0, 0.10),
]
DEFAULT = "claude-opus-5-5"


WEB_SEARCH = 0.01  # USD per search


def known(model: str | None) -> bool:
    m = (model or "").lower()
    return any(m.startswith(prefix) for prefix, *_ in _TABLE)


def price(model: str | None) -> tuple[float, float, float]:
    """(input, output, cache_read) in USD per token. Unknown models get the default's prices;
    callers report them with known()."""
    m = (model or DEFAULT).lower()
    for prefix, i, o, r in _TABLE:
        if m.startswith(prefix):
            return i / 1e6, o / 1e6, r / 1e6
    return price(DEFAULT)


def write_multiplier(usage: dict) -> float:
    """Cache write price as a multiple of input: 1.25 for 5-minute writes, 2 for 1-hour, weighted."""
    cc = usage.get("cache_creation") or {}
    w5, w1 = cc.get("ephemeral_5m_input_tokens") or 0, cc.get("ephemeral_1h_input_tokens") or 0
    return (1.25 * w5 + 2.0 * w1) / (w5 + w1) if w5 + w1 else 1.25


def bill(model: str | None, usage: dict) -> float:
    """Exact USD of one API call from its `usage` block."""
    i, o, r = price(model)
    cc = usage.get("cache_creation") or {}
    w5 = cc.get("ephemeral_5m_input_tokens")
    w1 = cc.get("ephemeral_1h_input_tokens")
    total_w = usage.get("cache_creation_input_tokens") or 0
    if w5 is None and w1 is None:
        w5, w1 = total_w, 0
    fast = 2.0 if usage.get("speed") == "fast" else 1.0  # fast mode: twice the standard rates
    searches = (usage.get("server_tool_use") or {}).get("web_search_requests") or 0
    return fast * ((usage.get("input_tokens") or 0) * i
                   + (usage.get("cache_read_input_tokens") or 0) * r
                   + (w5 or 0) * i * 1.25 + (w1 or 0) * i * 2.0
                   + (usage.get("output_tokens") or 0) * o) + searches * WEB_SEARCH
