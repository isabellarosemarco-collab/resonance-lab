import sys; sys.path.insert(0,'.')
import numpy as np
from lab import config as C
from lab.data import load_universe
from lab.hypotheses import HYPOTHESES
from lab.portfolio import simulate, cagr, sharpe
from run_battery import build_market, collect, benchmark
h=[x for x in HYPOTHESES if x.code=='T4'][0]
def run(syms, extra_cost=0.0, label=''):
    data=load_universe(syms if 'SPY' in syms else ['SPY']+syms); mkt,close=build_market(data); bench=benchmark(data,mkt.dates)
    tr,_=collect(h,data,mkt)
    if 'SPY' not in syms:
        keep = np.array([mkt.symbols[s] in syms for s in tr.sym]); tr=tr.subset(keep)
    eq=simulate(mkt.dates,mkt.open,close,mkt.cost+extra_cost,tr.sym,tr.pos,tr.hold)
    dd=lambda x:(x/x.cummax()-1).min()
    for nm,sl in [('full',slice(None)),('holdout',slice(mkt.disc_end+1,None))]:
        e=eq.iloc[sl]; b=bench.iloc[sl]
        print(f"{label:28s} {nm:8s} lab CAGR {100*cagr(e):5.1f}% sharpe {sharpe(e):.2f} maxDD {100*dd(e):5.1f}% | VOO {100*cagr(b):5.1f}% / {sharpe(b):.2f}")
run(C.ETFS, label='ETFs only (no survivorship)')
run(C.UNIVERSE, extra_cost=0.0015, label='all, +15bp extra cost')
