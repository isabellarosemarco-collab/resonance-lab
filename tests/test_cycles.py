import numpy as np

from lab.cycles import band, dominant_cycle, trough_at, turns_up


def test_dominant_cycle_finds_planted_period():
    t = np.arange(252)
    x = 0.01 * t + np.sin(2 * np.pi * t / 21)
    period, _ = dominant_cycle(x)
    assert abs(period - 21) < 1.5


def test_trough_projection_on_pure_cosine():
    p = 252 / 12  # exact FFT bin, period 21
    t = np.arange(252)
    # value at index n: cos(2*pi*n/p + phi). Choose phi so the low lands at n = 252.
    phi = np.pi - 2 * np.pi * 252 / p
    x = np.cos(2 * np.pi * t / p + phi)
    period, phase = dominant_cycle(x)
    assert trough_at(period, phase, 252)
    assert not trough_at(period, phase, 252 + 10)


def test_band_is_causal_and_nan_until_warm():
    c = np.arange(1.0, 30.0)
    o = band(c, 3)
    assert np.isnan(o[:5]).all() and not np.isnan(o[5])
    # changing the future must not change the past
    c2 = c.copy()
    c2[-1] = 1000
    assert np.allclose(band(c2, 3)[:-1], o[:-1], equal_nan=True)


def test_turns_up():
    o = np.array([3.0, 2.0, 1.0, 2.0, 3.0])
    assert turns_up(o).tolist() == [False, False, False, True, False]
