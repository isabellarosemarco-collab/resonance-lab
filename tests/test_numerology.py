from datetime import date

from lab.numerology import (date_root, digit_root, has_222, is_369, personal_day,
                            reduce_keep_masters)


def test_digit_root():
    assert digit_root(2061) == 9
    assert digit_root(12) == 3
    assert digit_root(0) == 0
    assert digit_root(10) == 1


def test_is_369_days_of_month():
    assert [d for d in range(1, 32) if is_369(d)] == [3, 6, 9, 12, 15, 18, 21, 24, 27, 30]


def test_reduce_keeps_masters():
    assert reduce_keep_masters(22) == 22
    assert reduce_keep_masters(29) == 11
    assert reduce_keep_masters(36) == 9


def test_bella_life_path_is_9():
    assert digit_root(36) == 9


def test_date_root():
    assert date_root(date(2026, 10, 25)) == 9


def test_personal_day():
    # personal year: 6+2+(2+0+2+6)=18 -> 9; month: 9+10=19 -> 1; day: 1+25=26 -> 8
    assert personal_day(date(2026, 10, 25)) == 8


def test_has_222():
    assert has_222(12.22)
    assert has_222(222.10)
    assert has_222(2.22)
    assert not has_222(10.21)
    assert not has_222(123.45)
