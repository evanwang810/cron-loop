"""Features for a trading decision, computed only from what was visible at decision time.

Two kinds of decision point share one feature list (missing parts are NaN):
- move: a bet just moved the price (3pp+); bet and bettor features are filled in.
- snapshot: a routine look at a market with nothing special happening, for buy-and-hold calls.
The same function runs in the backtest and live, so the model never sees a feature the live
bot could not compute. Bet rows are the compact form from pull.bets_for:
[time, userId, amount, outcome, probBefore, probAfter, isApi, isRedemption, isLimit].
"""
from __future__ import annotations

import math
import re
import time
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field
from itertools import accumulate

MINUTE = 60_000
HOUR = 60 * MINUTE
DAY = 24 * HOUR

NAN = float("nan")
TOPICS = {
    "sports": r"\bvs\.?\b|\bMLB\b|\bNBA\b|\bNFL\b|\bNHL\b|\bUFC\b|\bF1\b|Premier League|World Cup|\bgame \d",
    "ai": r"\bAI\b|GPT|OpenAI|Anthropic|Claude|Gemini|\bLLM|DeepSeek|xAI|Grok|benchmark",
    "politics": r"Trump|Biden|Vance|election|Senate|Congress|president|governor|primary|Democrat|Republican",
    "geo": r"Russia|Ukraine|China|Israel|Iran|Gaza|Taiwan|Venezuela|\bwar\b|ceasefire",
    "crypto": r"bitcoin|\bBTC\b|crypto|\bETH\b|solana|token",
    "meta": r"Manifold|\bmana\b|this market|\bI\b|\bmy\b",
    "deadline": r"\bby\b|before|\bin 20\d\d|end of",
}
TOPIC_RE = {k: re.compile(v, re.I) for k, v in TOPICS.items()}

FEATURES = [
    # the move itself (NaN on snapshots)
    "is_move", "delta", "abs_delta", "p_before", "amount_log", "is_api", "is_limit", "amount_over_liq",
    "revert_so_far", "opp_bets_since", "bets_since",
    "bettor_age_d_log", "bettor_is_bot", "bettor_bets_here", "bettor_mana_here_log", "bettor_dir_here",
    # market state at decision time
    "p_now", "p_extremity", "chg_1h", "chg_24h", "chg_7d", "path_24h", "liq_log", "mkt_age_h_log",
    "to_close_h_log", "life_left", "n_bets_log", "n_traders_log", "bets_1h", "bets_24h", "mana_24h_log",
    "secs_since_last_bet_log", "q_len", "hour", "weekday", *[f"topic_{k}" for k in TOPICS],
]


@dataclass
class History:
    """One market's bets with prefix sums, so every feature is a binary search away."""
    meta: dict
    bets: list[list]
    times: list[int] = field(init=False)

    def __post_init__(self) -> None:
        # Redemption rows are share bookkeeping, not trades, and their probAfter is often stale.
        self.bets = b = [x for x in self.bets if not x[7]]
        self.times = [x[0] for x in b]
        # A limit order's amount is its final filled total, which is not known when it is placed.
        self.mana = [0.0, *accumulate(0.0 if x[8] else abs(x[2]) for x in b)]
        self.path = [0.0, *accumulate(abs(x[5] - x[4]) for x in b)]
        seen: set[str] = set()
        self.uniq = [0]
        for x in b:
            seen.add(x[1])
            self.uniq.append(len(seen))
        self.by_user: dict[str, list[int]] = {}
        for k, x in enumerate(b):
            self.by_user.setdefault(x[1], []).append(k)

    def prob_at(self, ms: float) -> float | None:
        j = bisect_right(self.times, ms) - 1
        return self.bets[j][5] if j >= 0 else None

    def upto(self, ms: float) -> int:
        """Number of bets placed at or before ms."""
        return bisect_right(self.times, ms)


def log1p(x: float) -> float:
    return math.log1p(max(x, 0.0))


def market_features(h: History, now_ms: float) -> dict[str, float]:
    j = h.upto(now_ms)
    p_now = h.bets[j - 1][5] if j else h.meta.get("p") or 0.5
    lo_1h, lo_24h = bisect_left(h.times, now_ms - HOUR, 0, j), bisect_left(h.times, now_ms - DAY, 0, j)
    created = h.meta["createdTime"]
    close = min(h.meta.get("closeTime") or now_ms + 3650 * DAY, now_ms + 3650 * DAY)
    liq = float(h.meta.get("totalLiquidity") or 0.0)
    q = h.meta.get("question") or ""
    gm = time.gmtime(now_ms / 1000)

    def chg(span: float) -> float:
        before = h.prob_at(now_ms - span)
        return p_now - before if before is not None else NAN

    return {
        "p_now": p_now,
        "p_extremity": abs(p_now - 0.5),
        "chg_1h": chg(HOUR), "chg_24h": chg(DAY), "chg_7d": chg(7 * DAY),
        "path_24h": h.path[j] - h.path[lo_24h],
        "liq_log": log1p(liq),
        "mkt_age_h_log": log1p((now_ms - created) / HOUR),
        "to_close_h_log": log1p((close - now_ms) / HOUR),
        "life_left": (close - now_ms) / max(close - created, 1),
        "n_bets_log": log1p(j),
        "n_traders_log": log1p(h.uniq[j]),
        "bets_1h": float(j - lo_1h),
        "bets_24h": float(j - lo_24h),
        "mana_24h_log": log1p(h.mana[j] - h.mana[lo_24h]),
        "secs_since_last_bet_log": log1p((now_ms - h.times[j - 1]) / 1000) if j else NAN,
        "q_len": float(len(q)),
        "hour": float(gm.tm_hour),
        "weekday": float(gm.tm_wday),
        **{f"topic_{k}": float(bool(rx.search(q))) for k, rx in TOPIC_RE.items()},
    }


def move_features(h: History, i: int, now_ms: float, users: dict[str, dict]) -> dict[str, float]:
    """Market state as of the decision, plus what the move at bet i looked like."""
    t, uid, amount, _outcome, pb, pa, is_api, _red, is_limit = h.bets[i]
    d = 1.0 if pa > pb else -1.0
    j = h.upto(now_ms)
    since = h.bets[i + 1:j]
    feats = market_features(h, now_ms)
    mine = h.by_user.get(uid, [])
    k = bisect_left(mine, i)  # this bettor's earlier bets in the market
    earlier = [h.bets[m] for m in mine[:k]]
    user = users.get(uid) or {}
    age_d = (t - user["createdTime"]) / DAY if user.get("createdTime") else NAN
    liq = float(h.meta.get("totalLiquidity") or 0.0)
    feats.update({
        "is_move": 1.0,
        "delta": pa - pb,
        "abs_delta": abs(pa - pb),
        "p_before": pb,
        "amount_log": NAN if is_limit else log1p(amount),
        "is_api": float(is_api),
        "is_limit": float(is_limit),
        "amount_over_liq": amount / liq if liq and not is_limit else NAN,
        "revert_so_far": (pa - feats["p_now"]) * d,
        "opp_bets_since": float(sum((b[5] - b[4]) * d < 0 for b in since)),
        "bets_since": float(len(since)),
        "bettor_age_d_log": log1p(age_d) if age_d == age_d else NAN,
        "bettor_is_bot": float(bool(user.get("isBot"))) if user else NAN,
        "bettor_bets_here": float(k),
        "bettor_mana_here_log": log1p(sum(abs(b[2]) for b in earlier if not b[8])),
        "bettor_dir_here": float(sum((b[5] - b[4]) * d > 0 for b in earlier) - sum((b[5] - b[4]) * d < 0 for b in earlier)),
    })
    return feats


def snapshot_features(h: History, now_ms: float) -> dict[str, float]:
    feats = market_features(h, now_ms)
    feats["is_move"] = 0.0
    return feats


def row(feats: dict[str, float]) -> list[float]:
    return [feats.get(k, NAN) for k in FEATURES]
