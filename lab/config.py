"""Pre-registered constants. Changing anything here after a run invalidates that run."""
from __future__ import annotations

ETFS = ["SPY", "QQQ", "IWM", "DIA"]
STOCKS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "BRK-B", "JPM", "V", "MA",
    "UNH", "JNJ", "XOM", "CVX", "PG", "KO", "PEP", "WMT", "HD", "COST",
    "MRK", "ABBV", "LLY", "PFE", "ORCL", "CSCO", "INTC", "ADBE", "CRM", "AMD",
    "QCOM", "TXN", "IBM", "BAC", "WFC", "GS", "MS", "C", "T", "VZ",
    "DIS", "NKE", "MCD", "SBUX", "BA", "CAT", "GE", "MMM", "HON", "AVGO",
    "NFLX", "TSLA", "LOW", "UPS", "AMGN", "GILD",
]
UNIVERSE = ETFS + STOCKS            # 60 symbols, fixed before testing

START = "1999-01-01"
DISCOVERY_END = "2023-10-31"        # everything after this is the locked holdout
HOLDOUT_START = "2023-11-01"
VOO_INCEPTION = "2010-09-09"        # benchmark is SPY before this date, VOO after


def cost_bp(symbol: str) -> float:
    """Round-trip trading cost in basis points."""
    return 2.0 if symbol in ETFS else 5.0


N_TESTS = 16                        # 12 swing + 4 intraday, pre-registered
BAR = 0.05 / N_TESTS                # Bonferroni significance bar
N_SHIFTS = 2000                     # random calendar shifts for the luck test
MIN_SHIFT = 30                      # bars; keeps a shift from landing back on the real dates
SEED = 20261025

BIRTH_MONTH, BIRTH_DAY = 6, 2                  # life path 9

LAB_CAPITAL = 1000.0
SLOTS = 4
SLOT_SIZE = LAB_CAPITAL / SLOTS
