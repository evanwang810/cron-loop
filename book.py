"""Positions, sizing and exits. Mirrors v2.simulate so paper results can be compared with the backtest."""
from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import amm

SIZES = np.array([5.0, 10.0, 25.0, 50.0, 100.0])
MIN_TRADERS = 8
MAX_FRAC = 0.20
FIELDS = ["opened", "closed", "market", "url", "side", "stake", "returned", "pnl", "exit", "entry_price", "exit_price", "edge"]


def iso(ms: float) -> str:
    return datetime.fromtimestamp(ms / 1000, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class Position:
    side: str
    shares: float
    cost: float
    opened: int
    entry_price: float
    edge: float
    question: str
    url: str
    created: int
    next_daily: int


@dataclass
class Entry:
    side: str
    size: float
    shares: float
    after: float
    edge: float


class Book:
    def __init__(self, folder: Path, mode: str, bankroll: float) -> None:
        self.folder, self.mode = folder, mode
        folder.mkdir(parents=True, exist_ok=True)
        self.file = folder / f"{mode}.json"
        self.cash, self.positions, self.last_move, self.last_seen = bankroll, {}, {}, 0
        if self.file.exists():
            s = json.loads(self.file.read_text(encoding="utf-8"))
            self.cash, self.last_move, self.last_seen = s["cash"], s["last_move"], s["last_seen"]
            self.positions = {k: Position(**v) for k, v in s["positions"].items()}

    def equity(self) -> float:
        return self.cash + sum(p.cost for p in self.positions.values())

    def save(self) -> None:
        cutoff = self.last_seen - 3_600_000
        self.last_move = {k: v for k, v in self.last_move.items() if v >= cutoff}
        self.file.write_text(json.dumps({"cash": self.cash, "last_seen": self.last_seen, "last_move": self.last_move,
                                         "positions": {k: asdict(v) for k, v in self.positions.items()}}), encoding="utf-8")

    def log_equity(self, now: int) -> None:
        path = self.folder / f"{self.mode}_equity.csv"
        new = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["time", "cash", "open_cost", "equity", "positions"])
            w.writerow([iso(now), round(self.cash, 2), round(self.equity() - self.cash, 2), round(self.equity(), 2), len(self.positions)])

    def close(self, cid: str, proceeds: float, now: int, why: str, exit_price: float | None) -> float:
        p = self.positions.pop(cid)
        self.cash += proceeds
        path = self.folder / f"{self.mode}_trades.csv"
        new = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(FIELDS)
            w.writerow([iso(p.opened), iso(now), p.question, p.url, p.side, round(p.cost, 2), round(proceeds, 2), round(proceeds - p.cost, 2),
                        why, round(p.entry_price, 4), "" if exit_price is None else round(exit_price, 4), round(p.edge, 4)])
        return proceeds - p.cost

    def choose_entry(self, price: float, liq: float, pp: float, hold_yes: float, hold_no: float) -> Entry | None:
        """Best side and stake by predicted profit after worst-case slippage, capped by Kelly and 20% of equity."""
        if hold_yes <= price and hold_no <= 1.0 - price:
            return None
        equity = self.equity()
        sizes = SIZES[SIZES <= min(self.cash, MAX_FRAC * equity)]
        if not len(sizes):
            return None
        best: Entry | None = None
        top = 0.0
        for yes in (True, False):
            shares, after = amm.buy_shares(price, liq, pp, yes, sizes)
            q = hold_yes if yes else 1.0 - hold_no
            profit = amm.sell_value(q, liq, pp, yes, shares) - sizes
            c = price if yes else 1.0 - price
            kelly = max((hold_yes if yes else hold_no) - c, 0.0) / max(1.0 - c, 1e-9)
            profit = np.where(sizes <= kelly * equity, profit, -np.inf)
            k = int(np.argmax(profit))
            if profit[k] > top:
                top = float(profit[k])
                best = Entry("YES" if yes else "NO", float(sizes[k]), float(shares[k]), float(after[k]), (hold_yes if yes else hold_no) - c)
        return best

    @staticmethod
    def exit_now(pos: Position, price: float, liq: float, pp: float, hold_yes: float, hold_no: float) -> tuple[bool, float]:
        yes = pos.side == "YES"
        q = hold_yes if yes else 1.0 - hold_no
        now_val = float(amm.sell_value(price, liq, pp, yes, pos.shares)[0])
        return now_val >= float(amm.sell_value(q, liq, pp, yes, pos.shares)[0]), now_val


def limit_for(side: str, after: float) -> float:
    """Worst acceptable price for a buy, on Manifold's 1% grid, so a faster trader cannot make us overpay."""
    v = math.ceil(after * 100) / 100 if side == "YES" else math.floor(after * 100) / 100
    return min(max(v, 0.01), 0.99)
