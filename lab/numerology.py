"""Numerology predicates. Pure functions, no market data."""
from __future__ import annotations

from datetime import date

from lab.config import BIRTH_DAY, BIRTH_MONTH

MASTERS = {11, 22, 33}


def digit_sum(n: int) -> int:
    return sum(int(c) for c in str(abs(int(n))))


def digit_root(n: int) -> int:
    """Repeated digit sum down to one digit: 2061 -> 9. digit_root(0) == 0."""
    n = abs(int(n))
    return 0 if n == 0 else 1 + (n - 1) % 9


def reduce_keep_masters(n: int) -> int:
    """Like digit_root, but stops at the master numbers 11, 22, 33."""
    n = abs(int(n))
    while n > 9 and n not in MASTERS:
        n = digit_sum(n)
    return n


def is_369(n: int) -> bool:
    return digit_root(n) in (3, 6, 9)


def date_root(d: date) -> int:
    """Digit root of month + day + year: 2026-10-25 -> 10+25+2026 = 2061 -> 9."""
    return digit_root(d.month + d.day + d.year)


def personal_day(d: date, birth_month: int = BIRTH_MONTH, birth_day: int = BIRTH_DAY) -> int:
    """Classic personal day: personal year -> personal month -> personal day, masters kept."""
    py = reduce_keep_masters(birth_month + birth_day + digit_sum(d.year))
    pm = reduce_keep_masters(py + d.month)
    return reduce_keep_masters(pm + d.day)


def has_222(price: float) -> bool:
    """True if the traded price ends in .22 cents or contains the digits 222."""
    s = f"{price:.2f}"
    return s.endswith(".22") or "222" in s.replace(".", "")
