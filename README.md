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
| `docs/superpowers/specs/` | the design, written before any results |

*Not investment advice. It's an experiment, and most of the ideas in it lost.*
