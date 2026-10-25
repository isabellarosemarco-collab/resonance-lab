import sys; sys.path.insert(0,'.')
import numpy as np
from lab import config as C
from lab.data import load_universe
from lab.hypotheses import HYPOTHESES
from lab.portfolio import simulate, cagr, sharpe
from run_battery import build_market, collect, benchmark
data=load_universe(C.UNIVERSE); mkt,close=build_market(data); bench=benchmark(data,mkt.dates)
h=[x for x in HYPOTHESES if x.code=='T4'][0]; tr,_=collect(h,data,mkt)
eq=simulate(mkt.dates,mkt.open,close,mkt.cost,tr.sym,tr.pos,tr.hold)
for name,sl in [('full',slice(None)),('holdout',slice(mkt.disc_end+1,None)),('2015+',slice(int(np.searchsorted(mkt.dates,np.datetime64('2015-01-01'))),None))]:
    e=eq.iloc[sl]; b=bench.iloc[sl]
    dd=lambda x:(x/x.cummax()-1).min()
    print(f"{name:8s} lab CAGR {100*cagr(e):5.1f}% sharpe {sharpe(e):.2f} maxDD {100*dd(e):5.1f}% | VOO CAGR {100*cagr(b):5.1f}% sharpe {sharpe(b):.2f} maxDD {100*dd(b):5.1f}%")
ho=eq.iloc[mkt.disc_end+1:]; print('holdout $1000 ->', round(1000*ho.iloc[-1]/ho.iloc[0]), ' VOO ->', round(1000*bench.iloc[-1]/bench.iloc[mkt.disc_end+1]))
