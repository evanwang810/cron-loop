"""Watch Manifold's bet feed, act a fixed delay after each 3pp+ move, and manage positions. Paper trades without a key."""
from __future__ import annotations

import heapq
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import amm
import api
from book import MIN_TRADERS, Book, Position, limit_for
from features import DAY, History, move_features, snapshot_features
from model import Bettors, Model

HERE = Path(__file__).parent
STATE = HERE / "state"
CFG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
MAX_RUNTIME = float(os.environ.get("MAX_RUNTIME", 20_400))
POLL_SEC, SAVE_SEC, SETTLE_SEC = 4.0, 900.0, 300.0
MOVE_MIN, COOLDOWN, MAX_PAGES, STALE_MS = 0.03, 600_000, 8, 300_000
MODE = "live" if api.KEY and os.environ.get("LIVE") == "1" else "paper"  # a stored key alone never places a bet
stop = False


def now_ms() -> int:
    return int(time.time() * 1000)


def say(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S', time.gmtime())} {msg}", flush=True)


def on_term(*_: object) -> None:
    global stop
    stop = True


def tradable(m: dict) -> bool:
    return (m.get("outcomeType") == "BINARY" and m.get("mechanism") == "cpmm-1" and not m.get("isResolved") and (m.get("closeTime") or 0) > now_ms()
            and m.get("token", "MANA") == "MANA" and "--cash" not in (m.get("url") or "") and bool((m.get("pool") or {}).get("YES")))


def fetch_rows(cid: str) -> list[list] | None:
    """Same rows the backtest was built from, oldest first. None for markets too long to page, which training also skipped."""
    rows: list[list] = []
    before = None
    for _ in range(MAX_PAGES):
        page = api.get("bets", contractId=cid, limit=1000, before=before)
        rows += [[b["createdTime"], b["userId"], round(b.get("amount") or 0, 2), b.get("outcome"), round(b.get("probBefore") or 0, 4), round(b.get("probAfter") or 0, 4),
                  int(bool(b.get("isApi"))), int(bool(b.get("isRedemption"))), int("limitProb" in b)] for b in page]
        if len(page) < 1000:
            rows.sort(key=lambda r: r[0])
            return rows
        before = page[-1]["id"]
    return None


class Trader:
    def __init__(self) -> None:
        self.model = Model(HERE / "model.npz")
        self.bettors = Bettors(HERE / "users.npz")
        self.users: dict[str, dict] = {}
        bankroll = float(CFG.get("bankroll", 500))
        if MODE == "live":
            bankroll = min(bankroll, float(api.get("me")["balance"]))
        self.book = Book(STATE, MODE, bankroll)

    def account(self, uid: str) -> None:
        if uid not in self.users:
            try:
                self.users[uid] = {"createdTime": api.get(f"user/by-id/{uid}").get("createdTime")}
            except Exception:
                self.users[uid] = {}

    def inputs(self, feats: dict, liq: float, uid: str | None) -> np.ndarray:
        feats = dict(feats, liq_log=math.log1p(liq))
        if feats.get("is_move") == 1.0:
            amount_log = feats.get("amount_log", math.nan)
            feats["amount_over_liq"] = math.expm1(amount_log) / liq if amount_log == amount_log else math.nan
            feats.update(self.bettors.inputs(uid))
        return np.array([feats.get(n, math.nan) for n in self.model.names], float)

    def buy(self, cid: str, m: dict, entry, price: float, now: int) -> None:  # noqa: ANN001
        size, shares = entry.size, entry.shares
        if MODE == "live":
            r = api.post("bet", {"contractId": cid, "amount": size, "outcome": entry.side, "limitProb": limit_for(entry.side, entry.after), "expiresMillisAfter": 1000})
            size, shares = float(r.get("amount") or 0), float(r.get("shares") or 0)
            if size < 1 or shares <= 0:
                say(f"no fill {entry.side} {m['question'][:60]}")
                return
        self.book.cash -= size
        side_price = price if entry.side == "YES" else 1.0 - price
        self.book.positions[cid] = Position(entry.side, shares, size, now, side_price, entry.edge, m.get("question") or "", m.get("url") or "", int(m["createdTime"]), self.next_daily(int(m["createdTime"]), now))
        say(f"BUY {entry.side} {size:.0f} at {side_price:.3f} edge {entry.edge:+.3f} | {m['question'][:70]}")

    @staticmethod
    def next_daily(created: int, now: int) -> int:
        return created + ((now - created) // DAY + 1) * DAY

    def sell(self, cid: str, pos: Position, value: float, price: float, now: int, why: str) -> None:
        proceeds = value
        if MODE == "live":
            r = api.post(f"market/{cid}/sell", {"outcome": pos.side})  # no share count: sells the whole position, so rounding can't reject it
            proceeds = -float(r.get("amount") or -value)
        pnl = self.book.close(cid, proceeds, now, why, price if pos.side == "YES" else 1.0 - price)
        say(f"SELL {pos.side} {pos.cost:.0f} pnl {pnl:+.1f} ({why}) | {pos.question[:70]}")

    def look(self, cid: str, spike: dict | None) -> None:
        now = now_ms()
        m = api.get(f"market/{cid}")
        if not tradable(m):
            return
        rows = fetch_rows(cid)
        if not rows:
            return
        pool, pp = m["pool"], float(m.get("p") or 0.5)
        liq = float(pool["YES"]) ** pp * float(pool["NO"]) ** (1 - pp)
        price = float(m["probability"])
        h = History({**m, "id": cid}, rows)
        uid = None
        if spike is not None:
            i = next((k for k, b in enumerate(h.bets) if b[0] == spike["createdTime"] and b[1] == spike["userId"] and abs(b[5] - b[4]) >= MOVE_MIN), None)
            if i is None:
                return
            uid = spike["userId"]
            self.account(uid)
            feats = move_features(h, i, now, self.users)
        else:
            feats = snapshot_features(h, now)
        hold_yes, hold_no = self.model.hold_values(self.inputs(feats, liq, uid), price)
        pos = self.book.positions.get(cid)
        if pos is not None:
            sell_it, value = self.book.exit_now(pos, price, liq, pp, hold_yes, hold_no)
            hold_ms = CFG.get("hold_days", 0) * DAY
            if sell_it:
                self.sell(cid, pos, value, price, now, "sold")
            elif hold_ms and now - pos.opened >= hold_ms:
                self.sell(cid, pos, value, price, now, "time cap")
            return
        if h.uniq[h.upto(now)] < MIN_TRADERS or liq < 1 or (spike is None and CFG.get("no_daily")):
            return
        entry = self.book.choose_entry(price, liq, pp, hold_yes, hold_no)
        if entry is not None:
            self.buy(cid, m, entry, price, now)

    def settle(self) -> None:
        now = now_ms()
        for cid in list(self.book.positions):
            pos = self.book.positions[cid]
            try:
                m = api.get(f"market/{cid}")
                if m.get("isResolved"):
                    res = m.get("resolution")
                    value = {"YES": 1.0, "NO": 0.0}.get(res)
                    if res == "MKT":
                        value = float(m.get("resolutionProbability") or m.get("probability") or 0.5)
                    if value is None:  # N/A pays back what went in
                        self.book.close(cid, pos.cost, now, "refund", None)
                    else:
                        self.book.close(cid, pos.shares * (value if pos.side == "YES" else 1.0 - value), now, "resolved", value if pos.side == "YES" else 1.0 - value)
                    say(f"RESOLVED {res} | {pos.question[:70]}")
                elif pos.next_daily <= now:
                    pos.next_daily = self.next_daily(pos.created, now)
                    self.look(cid, None)
            except Exception as e:  # noqa: BLE001
                say(f"settle error {cid}: {e}")


def push_state() -> None:
    if not (STATE / ".git").exists():
        return
    git = ["git", "-C", str(STATE), "-c", "user.name=github-actions[bot]", "-c", "user.email=41898282+github-actions[bot]@users.noreply.github.com"]
    subprocess.run([*git, "add", "-A"], check=False)
    if subprocess.run([*git, "diff", "--cached", "--quiet"], check=False).returncode:
        subprocess.run([*git, "commit", "-q", "-m", "update"], check=False)
        subprocess.run([*git, "push", "-q", "origin", "HEAD:state"], check=False)


def main() -> None:
    signal.signal(signal.SIGTERM, on_term)
    t = Trader()
    b = t.book
    delay = t.model.delay_ms
    say(f"start: {MODE}, cash {b.cash:.1f}, {len(b.positions)} open, entry delay {delay / 1000:.0f}s")
    start, saved, settled = time.time(), time.time(), 0.0
    since = max(b.last_seen, now_ms() - 60_000)
    seen: set[str] = set()
    due: list[tuple[int, int, str, dict]] = []
    tick = 0
    while not stop and time.time() - start < MAX_RUNTIME:
        try:
            feed = api.get("bets", limit=1000, afterTime=since - 5000)
            for x in reversed(feed):
                if x["id"] in seen:
                    continue
                seen.add(x["id"])
                bt = x["createdTime"]
                b.last_seen = max(b.last_seen, bt)
                if x.get("answerId") or x.get("isRedemption") or abs(round(x.get("probAfter") or 0, 4) - round(x.get("probBefore") or 0, 4)) < MOVE_MIN:
                    continue
                if bt - b.last_move.get(x["contractId"], 0) < COOLDOWN:
                    continue
                b.last_move[x["contractId"]] = bt
                tick += 1
                heapq.heappush(due, (bt + delay, tick, x["contractId"], x))
            since = max(since, b.last_seen)
            if len(seen) > 200_000:
                seen.clear()
        except Exception as e:  # noqa: BLE001
            say(f"feed error: {e}")
        while due and due[0][0] <= now_ms() and not stop:
            when, _, cid, spike = heapq.heappop(due)
            if now_ms() - when > STALE_MS:
                continue
            try:
                t.look(cid, spike)
            except Exception as e:  # noqa: BLE001
                detail = e.read()[:300].decode("utf-8", "replace") if hasattr(e, "read") else ""
                say(f"look error {cid}: {e} {detail}")
                with (STATE / f"{MODE}_errors.log").open("a", encoding="utf-8") as f:
                    f.write(f"{now_ms()} {cid} {e} {detail}\n")
        if time.time() - settled > SETTLE_SEC:
            settled = time.time()
            t.settle()
        if time.time() - saved > SAVE_SEC:
            saved = time.time()
            b.log_equity(now_ms())
            b.save()
            push_state()
            say(f"cash {b.cash:.1f}, equity {b.equity():.1f}, {len(b.positions)} open")
        time.sleep(POLL_SEC)
    b.log_equity(now_ms())
    b.save()
    push_state()
    say("done")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
