"""The trained net as plain numpy, plus the bettor track-record table. No torch needed at run time."""
from __future__ import annotations

from pathlib import Path

import numpy as np

UNKNOWN_BETTOR = (0.0, 0.0, 0.2, 0.0)  # what a bettor with no history gets in training


class Model:
    def __init__(self, path: Path) -> None:
        z = np.load(path, allow_pickle=False)
        self.names = [str(n) for n in z["names"]]
        self.delay_ms = int(z["delay_ms"])
        self.lam, self.med, self.mu, self.sd = z["lam"], z["med"], z["mu"], z["sd"]
        self.w0, self.b0, self.w1, self.b1, self.w2, self.b2 = (z[k] for k in ("0_weight", "0_bias", "2_weight", "2_bias", "4_weight", "4_bias"))

    def hold_values(self, x: np.ndarray, price: float) -> tuple[float, float]:
        """Predicted value of holding one YES share and one NO share, shrunk toward the price as in the backtest."""
        z = (np.where(np.isnan(x), self.med, x) - self.mu) / self.sd
        h = np.maximum(self.w0 @ z + self.b0, 0.0)
        h = np.maximum(self.w1 @ h + self.b1, 0.0)
        net = np.clip(self.w2 @ h + self.b2, 0.0, 1.0)
        base = np.array([price, 1.0 - price])
        out = base + self.lam * (net - base)
        return float(out[0]), float(out[1])


class Bettors:
    def __init__(self, path: Path) -> None:
        z = np.load(path, allow_pickle=False)
        self.index = {str(u): i for i, u in enumerate(z["ids"])}
        self.table = z["U"]

    def inputs(self, uid: str | None) -> dict[str, float]:
        row = self.table[self.index[uid]] if uid in self.index else UNKNOWN_BETTOR
        return dict(zip(("u_bets_log", "u_spikes_log", "u_rev", "u_skill"), map(float, row)))
