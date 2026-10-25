"""Does T4 work on yesterday's giants that lagged? Removes the 'we know they won' bias."""
import sys; sys.path.insert(0, '.')
import numpy as np
from lab import config as C
from lab.data import load_universe
from lab.hypotheses import HYPOTHESES
from lab.portfolio import simulate, cagr, sharpe
from run_battery import build_market, collect, benchmark
LAGGARDS = ["GE", "CSCO", "INTC", "C", "PFE", "IBM", "T", "VZ", "MMM", "BA", "WFC", "KO", "MRK", "DIS", "BAC"]
h = [x for x in HYPOTHESES if x.code == 'T4'][0]
data = load_universe(['SPY'] + LAGGARDS); mkt, close = build_market(data); bench = benchmark(data, mkt.dates)
tr, _ = collect(h, data, mkt); tr = tr.subset(tr.sym != 0)
eq = simulate(mkt.dates, mkt.open, close, mkt.cost, tr.sym, tr.pos, tr.hold)
hold_eq = 1000 * np.nanmean(close[1:] / close[1:, [0]] if False else close[1:] / np.where(np.isnan(close[1:, :1]), np.nan, 1), axis=0)
bh = np.nanmean([close[i] / close[i][np.isfinite(close[i])][0] for i in range(1, len(mkt.symbols))], axis=0)
import pandas as pd
bh = pd.Series(1000 * bh, index=mkt.dates)
for nm, sl in [('25 yrs', slice(None)), ('last 3', slice(mkt.disc_end + 1, None))]:
    e, b, s = eq.iloc[sl], bench.iloc[sl], bh.iloc[sl]
    print(f"{nm:7s} T4 on laggards {100*cagr(e):5.1f}%/yr | just holding laggards {100*cagr(s):5.1f}%/yr | VOO {100*cagr(b):5.1f}%/yr")
