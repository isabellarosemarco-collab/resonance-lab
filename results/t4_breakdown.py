import numpy as np, sys
sys.path.insert(0, '.')
from lab import config as C
from lab.data import load_universe
from lab.hypotheses import Trade, after_close, down_streak, drop_overlaps
from run_battery import build_market
from lab.stats import Trades, summarize
data = load_universe(C.UNIVERSE); mkt, _ = build_market(data)
for L in range(1, 11):
    S, P, H = [], [], []
    for si, s in enumerate(mkt.symbols):
        df = data[s]; st = down_streak(df['close'].to_numpy())
        tr = [t for t in drop_overlaps(after_close(st == L, 3)) if t.entry < len(df)]
        if not tr: continue
        pos = mkt.dates.get_indexer(df.index[[t.entry for t in tr]]); k = pos >= 0
        S += [si]*int(k.sum()); P += pos[k].tolist(); H += [3]*int(k.sum())
    tr = Trades(np.array(S), np.array(P), np.array(H)); tr = tr.subset(tr.pos + 3 <= mkt.disc_end)
    s = summarize(mkt, tr, False)
    print(f"streak {L:2d}: trades {s['trades']:6d}  excess {s['excess_bp']:+6.1f} bp  win {s['win_pct']}%")
