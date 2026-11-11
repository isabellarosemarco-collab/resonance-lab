# Resonance Lab

*Can Tesla's 3-6-9, numerology and a few cosmic hunches beat the stock market? I tested them against 25 years of data and let the numbers decide.*

By [Bella Marco](https://bellamarco.co).

## The idea

Every idea goes in as a hypothesis, and the backtest decides. Mystical and ordinary signals get exactly the same treatment:

- **Tesla:** 3-6-9 days, 3-6-9 holding periods, 3-6-9 price levels, 3-6-9 losing streaks, plus "frequency, resonance, oscillation" turned into cycle math.
- **Numerology:** date roots, the 22nd of the month, my personal day numbers (life path 9, master number 22), and 222 prices.
- **Pattern recognition:** "the last 9 days look like these moments in the past, so do what happened next."

To count as an edge, a signal has to beat random trades and buy-and-hold VOO after costs, pass a luck test strict enough for testing many ideas at once, and hold up on three years of data that stayed locked away until the very end.

## What happened

60 large US stocks and ETFs, 1999–2026. Results: [`results/report_holdout.md`](results/report_holdout.md).

- **Every numerology and 3-6-9 calendar signal failed.** 3-6-9 days, date roots, the 22nd, my personal days and 222 prices all came out indistinguishable from random.
- **One signal passed: buying after a 3-6-9 day losing streak.** It beat random trades by about 0.19% per 3-day trade, and it did almost exactly as well on the hidden three years.
- **The magic numbers didn't matter.** The bounce just grows the longer a stock has been falling, and streaks of 4, 5, 7 and 8 days work as well as 3, 6 and 9. What I'd actually found was a well-known effect called short-term reversal.
- **Whether it beats simply holding the market depends on the market.** On former giants that lagged (GE, Intel, Cisco...) it made 11% a year over 25 years, against 9% for VOO. Over the last three years of a strong bull run, VOO won.

Next: trading the surviving signal with a real $1,000.

## Round 2: the Edge Lab

After the mystical ideas, I tested six well-known market effects the same way: momentum, sector rotation, the turn of the month, overnight returns, a 200-day trend filter, and post-earnings drift. Code: [`edge_lab.py`](edge_lab.py), [`edge_lab2.py`](edge_lab2.py), [`edge_lab3.py`](edge_lab3.py), [`edge_lab4.py`](edge_lab4.py), [`fib_lab.py`](fib_lab.py). Full results: [`results/edge_lab_report.md`](results/edge_lab_report.md).

- **Most "edges" were hindsight.** Momentum and buying gap-ups looked amazing on today's megacaps, but on former giants that lagged, gap-ups made 1.3% a year against 6.9% for just holding them.
- **Turn of the month and overnight returns failed** after trading costs.
- **Trend and sector rotation didn't make more money, but they cut the worst crash from -55% to about -20% to -30%.**
- **One effect held up: post-earnings drift.** Across 5,277 real earnings reports (2002-2026), stocks with the biggest earnings beats kept outperforming the biggest misses on the same stocks by about 2.7% over the next three months (t 3.9), after skipping the first-day jump. It held on the laggards too, smaller (+1.5%). In the last three years it was still positive but too small a sample to confirm.

- **A walk-forward prediction model looked great and wasn't.** It mostly learned "buy volatile stocks," which only works on stocks we already know won. On the laggards it lost to holding them (5.9% vs 9.7% a year).
- **Hindsight-proof strategies on broad ETFs** (volatility-managed SPY, low-vol sectors, global momentum, pre-holiday days) gave smoother rides but never beat SPY significantly.
- **Fibonacci retracements did nothing.** About 14,000 touches: the "magic" levels bounced exactly like random control levels (~50%).

**Now paper-trading** ([`paper_bot.py`](paper_bot.py), since Nov 10, 2026): the two survivors, post-earnings drift and the losing-streak bounce, run side by side on a paper account with no real money.

## Run it

```bash
pip install -r requirements.txt
pytest                         # 40 tests, including a no-peeking-at-the-future check for every signal
python run_battery.py          # discovery period (through Oct 2023)
```

`--unlock-holdout` opens the hidden years once. A lock file records when that happened, and any rerun after a bug fix is logged with its reason.

## Layout

| Path | What it does |
|---|---|
| `lab/hypotheses.py` | every signal, pre-registered |
| `lab/numerology.py` | digit roots, personal days, 222 |
| `lab/cycles.py` | FFT cycles, resonance |
| `lab/stats.py` | excess return, calendar-shift luck test |
| `lab/portfolio.py` | the $1,000 four-slot lab account |
| `edge_lab*.py`, `fib_lab.py` | the Edge Lab rounds |
| `paper_bot.py`, `run_paper.sh` | the paper-trading bot (Alpaca paper only) |
| `docs/superpowers/specs/` | the design, written before any results |

*Not investment advice. It's an experiment, and most of the ideas in it lost.*
