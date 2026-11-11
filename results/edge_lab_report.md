# Edge Lab results (2026-11-09)

Script: `edge_lab.py`. Same data and split as the Resonance Lab (discovery 1999 to Oct 2023, holdout Nov 2023 to Oct 2026). Costs: ETFs 2 bp round trip, stocks 5 bp. All rules were fixed before the run.

| Strategy | Discovery CAGR | Holdout CAGR | Verdict |
|---|---|---|---|
| Buy & hold SPY | 7.0% | 23.8% | benchmark |
| S1 ETF dual momentum | 8.9% (MaxDD -29% vs -55%) | 20.7% | Not significant (t 0.43). It cuts crash risk; it doesn't add return. |
| S2 Stock momentum | 18.8% | 37.3% | Hindsight-universe bias. Only +2.7%/yr vs equal-weight hold of the same stocks (t 1.1). |
| S3 Turn of month | 3.0% | 4.1% | Fail |
| S4 Overnight SPY | 2.2% | 10.2% | Fail (daily costs eat it) |
| S5 Trend filter 200d | 5.2% (MaxDD -28%) | 19.5% | Less return, half the drawdown. Risk tool only. |
| S6 Gap drift (PEAD proxy) | 21.9% | 41.1% | Looked best, but FAILED on the 15 laggard former giants (1.3% vs 6.9% for holding them). It's hindsight bias. |

Bottom line: on unbiased tests, nothing here beats buy-and-hold. T4 (Resonance Lab) is still the only signal that held up on the laggard universe.

# Round 2 (better data): `edge_lab2.py`
New data: FRED 3-month T-bill (cash earns interest), the 9 original sector SPDRs (no survivorship), and 5,277 real EPS surprises for the 56 stocks from Yahoo (2002-2026).

- **Post-earnings drift (PEAD) is the one real edge.** Measured 60 trading days after the reaction-day close (day-one jump excluded), net of 5 bp, vs SPY. Top-quintile beats beat bottom-quintile misses on the same stocks by **+2.68% per trade in discovery (t 3.86)** and +0.91% in the holdout (t 0.42, n=124/quintile, so too small to confirm). On the 15 laggard former giants the spread was +1.50% (t 1.39). Positive in 15 of 25 years.
- **R5 PEAD portfolio** (hold stocks for 60 days after a top-quintile beat, using trailing cutoffs, T-bills otherwise): 21.0% vs 15.3% for an equal-weight hold of the same stocks in discovery (t 2.49), and 29.7% vs 26.3% in the holdout (t 0.65). Universe bias still inflates the absolute CAGR.
- Sector momentum: matched SPY in discovery (9.0% vs 8.1%) with a -30% vs -55% max drawdown, and was slightly ahead in the holdout. No significant excess.
- Trend filter with T-bill cash: 6.7% vs 8.1%, with a -21% vs -55% drawdown. Risk tool.

# Round 3: stacking and a walk-forward model (`edge_lab3.py`)
- **Walk-forward ridge model** (retrained each January on past data only, top 10 of 56 monthly): 25.0% vs 15.6% for an equal-weight hold (t 3.09) in 2005-2023, and 38.0% vs 26.3% in the holdout. Its biggest weight was **high volatility**, though, and on the 15 laggards it made **5.9% vs 9.7%** for holding them. It learned hindsight, not an edge. On the laggards, the only positive weights left were the losing streak (T4) and the EPS surprise (PEAD).
- **Beat + near the 52-week high:** no consistent extra effect (-1.1% in discovery, +3.4% in the holdout, both insignificant).
- **T4 + PEAD 50/50:** the best Sharpe of the honest strategies (0.86 vs 0.82 for an equal-weight hold), but no fewer losing years (3/19). The return is about the same as holding.
- Every strategy had losing years. None beat buy-and-hold every year.

# Round 4: hindsight-proof universes (`edge_lab4.py`)
Broad ETFs and asset classes only (SPY, sectors, EFA, IEF, GLD), T-bill cash, 2004 onward.
- **Volatility-managed SPY:** 10.2% vs 9.0% with a -37% vs -55% drawdown in discovery (Sharpe 0.69 vs 0.55), and even with SPY in the holdout. A better ride, but the extra return isn't significant (t 0.24).
- **T4 on 13 ETFs:** 7.6% vs 9.0%. The bounce is weaker on diversified ETFs than on single stocks.
- **Low-vol sectors** and **global asset momentum**: smaller drawdowns, lower returns, below SPY in the holdout.
- **Pre-holiday SPY:** a very high Sharpe, but it's only invested about 9 days a year, so the total return is tiny (2.2%).
- Nothing beat SPY significantly. None of the strategies avoided losing years.

# Fibonacci retracements (`fib_lab.py`)
About 14,000 first touches of retracement levels after 20%+ run-ups on 60 symbols, measured as the 10-day return vs SPY. Fibonacci levels did +0.55% vs +0.54% for the control levels in discovery (t 0.13), and +0.20% vs -0.01% in the holdout (t 0.69). Bounce rates were about 50% at every level. No Fibonacci effect.

# Paper trading (`paper_bot.py`, live since 2026-11-10)
The two signals that survived every test now run side by side on an Alpaca paper account (no real money):
- **T4 losing-streak bounce:** a stock or ETF closes down 3+ days in a row: buy at the next open, sell 3 trading days later. Up to 4 positions of $500.
- **PEAD:** a top-20% EPS surprise (vs the trailing-year cutoff): buy at the close of the first reaction day, hold 60 trading days. Up to 10 positions of $500.

It runs twice a trading day (after the open and near the close) and writes a daily summary. AMZN is excluded. The base URL is hard-wired to the paper endpoint.

# S&P 500 holdout (`sp500_lab.py`, pre-registered in `docs/prereg/2026-11-11-sp500-holdout.md`)
482 of the 503 S&P 500 members on 2023-11-01 (21 later acquired or renamed couldn't be priced), Nov 2023 to Nov 2026. Rules unchanged. Bar: p < 0.0125.

| Test | Trades | Avg excess | t | Luck p | 2024 / 2025 / 2026 | Result |
|---|---|---|---|---|---|---|
| T4b losing streak 3+ | 37,418 | +0.05% | -0.40 | 0.17 | -0.11% / +0.20% / +0.00% | FAIL |
| T7 snap-back | 9,043 | +0.39% | 1.29 | 0.044 | +0.08% / +0.96% / -0.04% | FAIL |
| T6 cycles align | 839 | -0.23% | -0.73 | 0.88 | +0.20% / -0.27% / -0.63% | FAIL |
| PEAD top minus bottom quintile | 2,157 | +2.00% | 2.94 (p 0.0017) | | +0.96% / +2.96% / +2.56% | **PASS** |

- **Post-earnings drift is confirmed out of sample.** Beats beat misses by 2.0% over 60 days, positive every year, and the quintiles line up in order (-2.65%, -2.43%, -1.20%, -0.51%, -0.65% vs SPY).
- **But most of the edge is in avoiding the misses.** Long-only, the top quintile still trailed SPY (-0.65%), because SPY was driven by a few megacaps. It beat the average S&P stock in the sample by about 0.8%.
- **T4 did not generalize.** On 9x more stocks over the same three years it was flat. Its edge looks specific to the 60 megacaps it was found on.
