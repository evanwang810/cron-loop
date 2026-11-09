"""Manifold cpmm-1 maths, the same formulas the backtest priced every trade with."""
from __future__ import annotations

import numpy as np

EPS = 1e-4


def pool_for(prob: np.ndarray, liq: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    prob = np.clip(prob, EPS, 1 - EPS)
    ratio = prob * (1 - p) / (p * (1 - prob))
    y = liq / ratio ** (1 - p)
    return y, ratio * y


def prob_of(y: np.ndarray, n: np.ndarray, p: np.ndarray) -> np.ndarray:
    return p * n / ((1 - p) * y + p * n)


def buy(y: np.ndarray, n: np.ndarray, yes: np.ndarray, amount: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    k = y ** p * n ** (1 - p)
    y1, n1 = y + amount, n + amount
    y_new = (k / n1 ** (1 - p)) ** (1 / p)
    n_new = (k / y1 ** p) ** (1 / (1 - p))
    shares = np.where(yes, y1 - y_new, n1 - n_new)
    return shares, np.where(yes, y_new, y1), np.where(yes, n1, n_new)


def sell(y: np.ndarray, n: np.ndarray, yes: np.ndarray, shares: np.ndarray, p: np.ndarray) -> np.ndarray:
    k = y ** p * n ** (1 - p)
    y2, n2 = y + np.where(yes, shares, 0.0), n + np.where(yes, 0.0, shares)
    lo, hi = np.zeros_like(y2), np.minimum(y2, n2)
    for _ in range(45):
        m = (lo + hi) / 2
        over = (y2 - m) ** p * (n2 - m) ** (1 - p) > k
        lo, hi = np.where(over, m, lo), np.where(over, hi, m)
    return lo


def sell_value(prob: float, liq: float, p: float, yes: bool, shares: np.ndarray | float) -> np.ndarray:
    s = np.atleast_1d(np.asarray(shares, float))
    y, n = pool_for(np.full(len(s), prob), np.full(len(s), liq), np.full(len(s), p))
    return sell(y, n, np.full(len(s), yes), s, np.full(len(s), p))


def buy_shares(prob: float, liq: float, p: float, yes: bool, amounts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Shares bought for each amount, and the price the pool ends at."""
    a = np.asarray(amounts, float)
    y, n = pool_for(np.full(len(a), prob), np.full(len(a), liq), np.full(len(a), p))
    shares, y1, n1 = buy(y, n, np.full(len(a), yes), a, np.full(len(a), p))
    return shares, prob_of(y1, n1, np.full(len(a), p))
