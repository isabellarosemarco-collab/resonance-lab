"""Tesla's 'frequency, resonance, oscillation' turned into causal price math."""
from __future__ import annotations

import numpy as np


def dominant_cycle(x: np.ndarray, pmin: float = 3, pmax: float = 60) -> tuple[float, float]:
    """Strongest sinusoid in a linearly detrended window, by FFT.

    Returns (period_in_bars, phase) such that the cycle's value at window index n is
    proportional to cos(2*pi*n/period + phase).
    """
    n = len(x)
    t = np.arange(n)
    slope, intercept = np.polyfit(t, x, 1)
    resid = x - (slope * t + intercept)
    spec = np.fft.rfft(resid)
    freqs = np.fft.rfftfreq(n)
    with np.errstate(divide="ignore"):
        periods = np.where(freqs > 0, 1.0 / freqs, np.inf)
    power = np.where((periods >= pmin) & (periods <= pmax), np.abs(spec), -1.0)
    k = int(np.argmax(power))
    return float(periods[k]), float(np.angle(spec[k]))


def trough_at(period: float, phase: float, n: int) -> bool:
    """Is window index n (the bar after the window ends) within half a bar of the cycle's low?"""
    theta = (2 * np.pi * n / period + phase) % (2 * np.pi)
    return abs(theta - np.pi) <= np.pi / period


def band(close: np.ndarray, k: int) -> np.ndarray:
    """Causal k-bar oscillator: SMA(k) - SMA(2k). NaN until 2k bars exist."""
    def sma(a: np.ndarray, w: int) -> np.ndarray:
        c = np.cumsum(np.insert(a, 0, 0.0))
        out = np.full(len(a), np.nan)
        out[w - 1:] = (c[w:] - c[:-w]) / w
        return out
    return sma(close, k) - sma(close, 2 * k)


def turns_up(o: np.ndarray) -> np.ndarray:
    """True at t when o rose at t after falling or flat at t-1 (a local trough just passed)."""
    out = np.zeros(len(o), dtype=bool)
    out[2:] = (o[2:] > o[1:-1]) & (o[1:-1] <= o[:-2])
    return out & ~np.isnan(o)
