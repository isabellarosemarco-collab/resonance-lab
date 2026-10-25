import pandas as pd

from lab.data import unsplit


def test_unsplit_restores_pre_split_price():
    idx = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"])
    adjusted = pd.Series([50.0, 50.0, 50.0, 50.0], index=idx)
    splits = pd.Series([2.0], index=pd.to_datetime(["2020-01-06"]))
    assert unsplit(adjusted, splits).tolist() == [100.0, 100.0, 50.0, 50.0]


def test_unsplit_compounds_multiple_splits():
    idx = pd.to_datetime(["2020-01-02", "2021-01-04", "2022-01-03"])
    adjusted = pd.Series([10.0, 10.0, 10.0], index=idx)
    splits = pd.Series([2.0, 3.0], index=pd.to_datetime(["2021-01-04", "2022-01-03"]))
    assert unsplit(adjusted, splits).tolist() == [60.0, 30.0, 10.0]
