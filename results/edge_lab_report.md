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
