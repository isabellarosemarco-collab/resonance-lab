"""Phase 1: run every swing hypothesis, write results/scoreboard.csv and results/report.md.

    python run_battery.py                    # discovery period only (default)
    python run_battery.py --unlock-holdout   # one-time look at 2023-11 onward
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from lab import config as C
from lab.data import load_daily, load_universe
from lab.hypotheses import HYPOTHESES, drop_overlaps
from lab.portfolio import cagr, sharpe, simulate
from lab.stats import Market, Trades, block_consistency, shift_p, summarize

RESULTS = Path(__file__).resolve().parent / "results"
LOCK = RESULTS / "HOLDOUT_UNLOCKED"


def build_market(data: dict[str, pd.DataFrame]) -> tuple[Market, np.ndarray]:
    symbols = list(data)
    dates = data["SPY"].index
    op = np.vstack([data[s]["open"].reindex(dates).to_numpy() for s in symbols])
    cl = np.vstack([data[s]["close"].reindex(dates).to_numpy() for s in symbols])
    cost = np.array([C.cost_bp(s) / 1e4 for s in symbols])
    disc_end = int(np.searchsorted(dates, pd.Timestamp(C.DISCOVERY_END), side="right") - 1)
    return Market(dates, symbols, op, cost, disc_end), cl


def collect(h, data: dict[str, pd.DataFrame], mkt: Market) -> tuple[Trades, Trades]:
    out = {"trades": ([], [], []), "twin": ([], [], [])}
    for si, s in enumerate(mkt.symbols):
        df = data[s]
        sig = h.fn(df)
        for key, trades in (("trades", sig.trades), ("twin", sig.twin)):
            if not h.per_day:
                trades = drop_overlaps(trades)
            trades = [t for t in trades if t.entry < len(df)]
            if not trades:
                continue
            pos = mkt.dates.get_indexer(df.index[[t.entry for t in trades]])
            keep = pos >= 0
            out[key][0].extend([si] * int(keep.sum()))
            out[key][1].extend(pos[keep].tolist())
            out[key][2].extend(np.array([t.hold for t in trades])[keep].tolist())
    mk = lambda k: Trades(*(np.asarray(a, dtype=int) for a in out[k]))
    return mk("trades"), mk("twin")


def in_discovery(mkt: Market, tr: Trades) -> Trades:
    return tr.subset(tr.pos + tr.hold <= mkt.disc_end)


def in_holdout(mkt: Market, tr: Trades) -> Trades:
    return tr.subset((tr.pos > mkt.disc_end) & (tr.pos + tr.hold < len(mkt.dates)))


def benchmark(data: dict[str, pd.DataFrame], dates: pd.DatetimeIndex) -> pd.Series:
    """$1,000 in SPY until VOO existed, then VOO."""
    spy = data["SPY"]["close"].reindex(dates)
    voo = load_daily("VOO")["close"].reindex(dates)
    r = spy.pct_change()
    after = dates >= pd.Timestamp(C.VOO_INCEPTION)
    r[after] = voo.pct_change()[after]
    return 1000 * (1 + r.fillna(0)).cumprod()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--unlock-holdout", action="store_true")
    ap.add_argument("--only", help="comma-separated hypothesis codes")
    ap.add_argument("--symbols", help="comma-separated symbols (smoke tests)")
    args = ap.parse_args()
    if args.unlock_holdout and LOCK.exists():
        print(f"Holdout was already unlocked on {LOCK.read_text().strip()}. It is used once.")
        return 1

    RESULTS.mkdir(exist_ok=True)
    syms = args.symbols.split(",") if args.symbols else C.UNIVERSE
    if "SPY" not in syms:
        syms = ["SPY"] + syms
    print(f"Loading {len(syms)} symbols...")
    data = load_universe(syms)
    mkt, close = build_market(data)
    bench = benchmark(data, mkt.dates)
    hyps = [h for h in HYPOTHESES if not args.only or h.code in args.only.split(",")]

    rows = []
    for h in hyps:
        print(f"  {h.code} {h.name}...", flush=True)
        tr_all, twin_all = collect(h, data, mkt)
        tr = in_discovery(mkt, tr_all)
        row = {"code": h.code, "hypothesis": h.name, "counted": h.counted}
        if len(tr) < 30:
            rows.append({**row, "trades": len(tr), "status": "TOO FEW TRADES"})
            continue
        s = summarize(mkt, tr, h.per_day)
        exc = s.pop("_exc")
        row.update(s)
        row["luck_p"] = round(shift_p(mkt, tr, 0, mkt.disc_end, C.N_SHIFTS, C.MIN_SHIFT,
                                      C.SEED, h.per_day), 4)
        row["consistency_pct"] = round(100 * block_consistency(mkt, tr, exc), 0)
        twin = in_discovery(mkt, twin_all)
        if len(twin) >= 30:
            ts = summarize(mkt, twin, h.per_day)
            row["twin_excess_bp"] = ts["excess_bp"]
            row["magic_minus_twin_bp"] = round(row["excess_bp"] - ts["excess_bp"], 2)

        disc_dates = mkt.dates[:mkt.disc_end + 1]
        eq = simulate(disc_dates, mkt.open[:, :mkt.disc_end + 1], close[:, :mkt.disc_end + 1],
                      mkt.cost, tr.sym, tr.pos, tr.hold)
        row["lab_cagr_pct"] = round(100 * cagr(eq), 2)
        row["lab_sharpe"] = round(sharpe(eq), 2)
        row["voo_sharpe"] = round(sharpe(bench.iloc[:mkt.disc_end + 1]), 2)

        passed = [row["avg_net_pct"] > 0, row["luck_p"] < C.BAR,
                  row["consistency_pct"] > 50, row["lab_sharpe"] > row["voo_sharpe"]]

        if args.unlock_holdout:
            ho = in_holdout(mkt, tr_all)
            if len(ho) >= 10:
                hs = summarize(mkt, ho, h.per_day)
                row["holdout_trades"] = hs["trades"]
                row["holdout_excess_bp"] = hs["excess_bp"]
                passed.append(np.sign(hs["excess_bp"]) == np.sign(row["excess_bp"])
                              and abs(hs["excess_bp"]) >= 0.5 * abs(row["excess_bp"]))
            else:
                passed.append(False)
            full = simulate(mkt.dates, mkt.open, close, mkt.cost, tr_all.sym, tr_all.pos, tr_all.hold)
            row["full_lab_sharpe"] = round(sharpe(full), 2)
            row["full_voo_sharpe"] = round(sharpe(bench), 2)
            passed[3] = row["full_lab_sharpe"] > row["full_voo_sharpe"]
            row["status"] = "PASS" if all(passed) else "FAIL"
        else:
            row["status"] = "CANDIDATE" if all(passed) else "FAIL"
        if not h.counted:
            row["status"] = f"reference ({row['status'].lower()})"
        rows.append(row)

    board = pd.DataFrame(rows)
    stamp = "holdout" if args.unlock_holdout else "discovery"
    board.to_csv(RESULTS / f"scoreboard_{stamp}.csv", index=False)
    (RESULTS / f"report_{stamp}.md").write_text(report(board, stamp))
    if args.unlock_holdout and not args.symbols and not args.only:
        LOCK.write_text(datetime.now().isoformat(timespec="seconds"))
    cols = ["code", "hypothesis", "trades_per_yr", "avg_net_pct", "excess_bp", "luck_p",
            "consistency_pct", "lab_sharpe", "voo_sharpe", "status"]
    print(board[[c for c in cols if c in board]].to_string(index=False))
    return 0


def report(board: pd.DataFrame, stamp: str) -> str:
    lines = [f"# Resonance Lab — {stamp} results",
             f"_Generated {datetime.now():%Y-%m-%d %H:%M}. Luck bar: p < {C.BAR:.4f}. "
             f"Discovery ends {C.DISCOVERY_END}._", "",
             "| Signal | Trades/yr | Avg trade | vs random | Luck p | Consistent | Lab Sharpe | VOO Sharpe | Result |",
             "|---|---|---|---|---|---|---|---|---|"]
    for _, r in board.iterrows():
        g = lambda k, f="{}": (f.format(r[k]) if k in r and pd.notna(r[k]) else "–")
        lines.append(f"| {r['code']} {r['hypothesis']} | {g('trades_per_yr')} | {g('avg_net_pct', '{:+.2f}%')} "
                     f"| {g('excess_bp', '{:+.1f} bp')} | {g('luck_p')} | {g('consistency_pct', '{:.0f}%')} "
                     f"| {g('lab_sharpe')} | {g('voo_sharpe')} | **{r['status']}** |")
    lines += ["", "**How to read this:** *vs random* is how much better each trade did than a random",
              "trade in the same stock held for the same time (1 bp = 0.01%). *Luck p* is the chance a",
              "random calendar would have looked this good. *Consistent* is the share of 2-year periods",
              "where it beat random. A signal must clear every bar to be traded."]
    if "magic_minus_twin_bp" in board:
        lines += ["", "**Does the magic number matter?** (signal minus its non-3-6-9 twin)", ""]
        for _, r in board.dropna(subset=["magic_minus_twin_bp"]).iterrows():
            lines.append(f"- {r['code']} {r['hypothesis']}: {r['magic_minus_twin_bp']:+.1f} bp")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.exit(main())
