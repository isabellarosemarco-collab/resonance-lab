"""Paper-trading bot for the two signals that held up in testing. Alpaca PAPER account only.

  T4    losing-streak bounce : a stock closes down 3+ days in a row -> buy at the next open,
                               sell after 3 trading days.                       (Resonance Lab)
  PEAD  post-earnings drift  : S&P 500. Top-20% EPS surprise (vs trailing-year cutoff) -> buy at the
                               close of the first reaction day; bottom-20% surprise -> SHORT it.
                               Close either after 60 trading days. (Edge Lab; confirmed on the
                               S&P 500 holdout 2023-26: top minus bottom +2.0% per 60 days, t 2.94)

Runs twice a trading day (launchd):
  python paper_bot.py open    ~9:35 ET  : sell expired T4 positions, buy new T4 signals, and scan the
                                          S&P 500 for today's earnings reactions (cached for the close)
  python paper_bot.py close   ~15:50 ET : close expired PEAD positions, buy today's big beats and
                                          short today's big misses, write the daily summary
Add --dry to print what it would do without placing orders.

Sizing: T4 4 slots x $500; PEAD 30 long + 30 short slots x $500 (biggest surprises first).
AMZN is excluded (Amazon employee trading rules). Keys come from ~/.alpaca/paper.env and the
base URL is hard-wired to the PAPER endpoint, so this cannot touch a live account.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import date, datetime, timedelta

import pandas as pd
import requests
import yfinance as yf

from lab.config import ETFS, STOCKS

BASE = "https://paper-api.alpaca.markets"          # PAPER ONLY. Never change this to the live URL.
HERE = os.path.dirname(os.path.abspath(__file__))
STATE = os.path.join(HERE, "paper", "state.json")
LOG = os.path.join(HERE, "paper", "trades.csv")
SUMMARY_DIR = os.path.join(HERE, "paper", "daily")

UNIVERSE = [s for s in ETFS + STOCKS if s != "AMZN"]
PEAD_UNIVERSE = open(os.path.join(HERE, "data_cache/sp500_current.txt")).read().split()   # AMZN excluded
T4_SLOTS, PEAD_SLOTS, SLOT_USD = 4, 30, 500.0
SCAN = os.path.join(HERE, "paper", "pead_scan.json")
LOCK = os.path.join(HERE, "paper", ".lock")
T4_STREAK, T4_HOLD, PEAD_HOLD, PEAD_PCTL = 3, 3, 60, 0.80


# ---------- Alpaca ----------
def _keys() -> dict:
    env = {}
    for line in open(os.path.expanduser("~/.alpaca/paper.env")):
        if "=" in line:
            k, v = line.strip().split("=", 1)
            env[k] = v
    return {"APCA-API-KEY-ID": env["APCA_API_KEY_ID"], "APCA-API-SECRET-KEY": env["APCA_API_SECRET_KEY"]}


def api(method: str, path: str, **kw):
    r = requests.request(method, BASE + path, headers=_keys(), timeout=30, **kw)
    if r.status_code >= 400:
        raise RuntimeError(f"Alpaca {method} {path}: {r.status_code} {r.text[:200]}")
    return r.json() if r.text else {}


def trading_days(start: date, end: date) -> list[str]:
    cal = api("GET", "/v2/calendar", params={"start": start.isoformat(), "end": end.isoformat()})
    return [c["date"] for c in cal]


def is_trading_day(d: date) -> bool:
    return d.isoformat() in trading_days(d, d)


def days_held(entry: str, today: date) -> int:
    """Trading days from the entry day to today (entry day = 0)."""
    return len(trading_days(date.fromisoformat(entry), today)) - 1


def buy_notional(sym: str, usd: float, dry: bool):
    if dry:
        return {"id": "DRY", "symbol": sym}
    return api("POST", "/v2/orders", json={"symbol": sym, "notional": round(usd, 2), "side": "buy",
                                           "type": "market", "time_in_force": "day"})


def short_qty(sym: str, usd: float, dry: bool):
    """Short whole shares worth about `usd` (Alpaca can't short fractional/notional). None if not shortable."""
    a = api("GET", f"/v2/assets/{sym}")
    if not (a.get("shortable") and a.get("easy_to_borrow")):
        return None
    q = yf.Ticker(sym).fast_info.get("lastPrice") or 0
    n = int(usd // q) if q else 0
    if n < 1:
        return None
    if dry:
        return {"id": "DRY", "symbol": sym, "qty": n}
    o = api("POST", "/v2/orders", json={"symbol": sym, "qty": str(n), "side": "sell", "type": "market", "time_in_force": "day"})
    o["qty"] = n
    return o


def cover_qty(sym: str, qty: float, dry: bool):
    if dry or qty <= 0:
        return {"id": "DRY", "symbol": sym}
    pos = {p["symbol"]: float(p["qty"]) for p in api("GET", "/v2/positions")}
    q = min(qty, -pos.get(sym, 0.0))
    if q <= 0:
        return {"id": "NONE", "symbol": sym}
    return api("POST", "/v2/orders", json={"symbol": sym, "qty": str(int(q)), "side": "buy",
                                           "type": "market", "time_in_force": "day"})


def sell_qty(sym: str, qty: float, dry: bool):
    if dry or qty <= 0:
        return {"id": "DRY", "symbol": sym}
    pos = {p["symbol"]: float(p["qty"]) for p in api("GET", "/v2/positions")}
    q = min(qty, pos.get(sym, 0.0))
    if q <= 0:
        return {"id": "NONE", "symbol": sym}
    return api("POST", "/v2/orders", json={"symbol": sym, "qty": str(round(q, 9)), "side": "sell",
                                           "type": "market", "time_in_force": "day"})


def filled_qty(order_id: str) -> float:
    try:
        return float(api("GET", f"/v2/orders/{order_id}").get("filled_qty") or 0)
    except Exception:
        return 0.0


# ---------- state & logging ----------
def load_state() -> dict:
    if os.path.exists(STATE):
        st = json.load(open(STATE))
        st.setdefault("pead_short", [])
        return st
    return {"t4": [], "pead": [], "pead_short": [], "pending": []}


def save_state(st: dict):
    os.makedirs(os.path.dirname(STATE), exist_ok=True)
    json.dump(st, open(STATE, "w"), indent=2)


def log_trade(row: dict):
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    new = not os.path.exists(LOG)
    pd.DataFrame([row]).to_csv(LOG, mode="a", header=new, index=False)


def settle_pending(st: dict):
    """Record filled quantities for buys placed on the previous run."""
    still = []
    for p in st.get("pending", []):
        q = abs(filled_qty(p["order_id"]))
        if q > 0:
            st[p["strategy"]].append({"symbol": p["symbol"], "entry": p["entry"], "qty": q, "usd": p["usd"]})
        elif p.get("tries", 0) < 3:
            p["tries"] = p.get("tries", 0) + 1
            still.append(p)
    st["pending"] = still


# ---------- signals ----------
def t4_signals(today: date) -> list[str]:
    hist = yf.download(UNIVERSE, period="2mo", auto_adjust=True, progress=False)["Close"]
    hist = hist[hist.index.date < today]                     # completed sessions only
    out = []
    for s in UNIVERSE:
        c = hist[s].dropna()
        if len(c) < T4_STREAK + 1:
            continue
        down = (c.diff() < 0).iloc[::-1]
        streak = 0
        for d in down:
            if d:
                streak += 1
            else:
                break
        if streak >= T4_STREAK:
            out.append((streak, s))
    return [s for _, s in sorted(out, reverse=True)]         # longest streaks first


def pead_cutoffs() -> tuple[float, float]:
    """20th / 80th percentile EPS surprise over the trailing year of S&P 500 reports (known today)."""
    import glob
    fs = glob.glob(os.path.join(HERE, "data_cache/sp500/*.earn.parquet"))
    e = pd.concat([pd.read_parquet(f) for f in fs]).dropna(subset=["surprise_pct"])
    scan = json.load(open(SCAN)) if os.path.exists(SCAN) else {}
    extra = [r for d in scan.get("history", {}).values() for r in d]
    if extra:
        e = pd.concat([e, pd.DataFrame(extra)])
    ts = pd.to_datetime(e["date"], utc=True)
    recent = e[ts >= ts.max() - pd.Timedelta(days=365)]
    return float(recent["surprise_pct"].quantile(1 - PEAD_PCTL)), float(recent["surprise_pct"].quantile(PEAD_PCTL))


def pead_scan(today: date) -> list[dict]:
    """Every S&P 500 stock whose first reaction day to an earnings report is TODAY: reports before
    today's open, or after yesterday's close. Known by the open, so it runs in the open job."""
    prev = trading_days(today - timedelta(days=7), today)[-2]
    out = []
    for s in PEAD_UNIVERSE:
        try:
            e = yf.Ticker(s).get_earnings_dates(limit=4)
        except Exception:
            continue
        if e is None or e.empty:
            continue
        for ts, row in e.iterrows():
            sp = row.get("Surprise(%)")
            if pd.isna(sp):
                continue
            t = pd.Timestamp(ts).tz_convert("America/New_York")
            if (t.hour < 16 and t.date() == today) or (t.hour >= 16 and t.date().isoformat() == prev):
                out.append({"symbol": s, "surprise_pct": float(sp), "date": t.isoformat()})
    return out


def pead_signals(today: date) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
    """(longs, shorts) for today from the open-run scan: top-20% beats, bottom-20% misses."""
    scan = json.load(open(SCAN)) if os.path.exists(SCAN) else {}
    rows = scan.get("history", {}).get(today.isoformat())
    if rows is None:                                      # open scan didn't run; scan now
        rows = pead_scan(today)
    lo, hi = pead_cutoffs()
    longs = sorted([(r["symbol"], r["surprise_pct"]) for r in rows if r["surprise_pct"] >= hi], key=lambda x: -x[1])
    shorts = sorted([(r["symbol"], r["surprise_pct"]) for r in rows if r["surprise_pct"] <= lo], key=lambda x: x[1])
    return longs, shorts


# ---------- runs ----------
def run_open(dry: bool) -> list[str]:
    today = date.today()
    st = load_state(); settle_pending(st); notes = []
    keep = []
    for p in st["t4"]:
        if days_held(p["entry"], today) >= T4_HOLD:
            o = sell_qty(p["symbol"], p["qty"], dry)
            log_trade({"time": datetime.now().isoformat(timespec="seconds"), "strategy": "T4", "side": "sell",
                       "symbol": p["symbol"], "qty": p["qty"], "order": o.get("id")})
            notes.append(f"T4 sell {p['symbol']} (held {T4_HOLD} days)")
        else:
            keep.append(p)
    st["t4"] = keep
    open_slots = T4_SLOTS - len(st["t4"]) - sum(1 for p in st["pending"] if p["strategy"] == "t4")
    held = {p["symbol"] for p in st["t4"]}
    for s in [x for x in t4_signals(today) if x not in held][:max(open_slots, 0)]:
        o = buy_notional(s, SLOT_USD, dry)
        st["pending"].append({"strategy": "t4", "symbol": s, "entry": today.isoformat(), "usd": SLOT_USD, "order_id": o.get("id")})
        log_trade({"time": datetime.now().isoformat(timespec="seconds"), "strategy": "T4", "side": "buy",
                   "symbol": s, "usd": SLOT_USD, "order": o.get("id")})
        notes.append(f"T4 buy {s} ${SLOT_USD:.0f} (3+ down days)")
    try:
        rows = pead_scan(today)
        scan = json.load(open(SCAN)) if os.path.exists(SCAN) else {"history": {}}
        scan.setdefault("history", {})[today.isoformat()] = rows
        if not dry:
            json.dump(scan, open(SCAN, "w"))
        notes.append(f"earnings scan: {len(rows)} reporting today")
    except Exception as ex:
        notes.append(f"earnings scan failed ({str(ex)[:60]}); close run will retry")
    if not dry:
        save_state(st)
    return notes


def run_close(dry: bool, allow_buys: bool = True) -> list[str]:
    today = date.today()
    st = load_state(); settle_pending(st); notes = []
    keep = []
    for p in st["pead"]:
        if days_held(p["entry"], today) >= PEAD_HOLD:
            o = sell_qty(p["symbol"], p["qty"], dry)
            log_trade({"time": datetime.now().isoformat(timespec="seconds"), "strategy": "PEAD", "side": "sell",
                       "symbol": p["symbol"], "qty": p["qty"], "order": o.get("id")})
            notes.append(f"PEAD sell {p['symbol']} (held {PEAD_HOLD} days)")
        else:
            keep.append(p)
    st["pead"] = keep
    keep = []
    for p in st["pead_short"]:
        if days_held(p["entry"], today) >= PEAD_HOLD:
            o = cover_qty(p["symbol"], p["qty"], dry)
            log_trade({"time": datetime.now().isoformat(timespec="seconds"), "strategy": "PEAD_SHORT", "side": "cover",
                       "symbol": p["symbol"], "qty": p["qty"], "order": o.get("id")})
            notes.append(f"PEAD cover short {p['symbol']} (held {PEAD_HOLD} days)")
        else:
            keep.append(p)
    st["pead_short"] = keep
    longs, shorts = pead_signals(today) if allow_buys else ([], [])
    held = {p["symbol"] for p in st["pead"] + st["pead_short"]}
    open_slots = PEAD_SLOTS - len(st["pead"]) - sum(1 for p in st["pending"] if p["strategy"] == "pead")
    for s, sp in [x for x in longs if x[0] not in held][:max(open_slots, 0)]:
        o = buy_notional(s, SLOT_USD, dry)
        st["pending"].append({"strategy": "pead", "symbol": s, "entry": today.isoformat(), "usd": SLOT_USD, "order_id": o.get("id")})
        log_trade({"time": datetime.now().isoformat(timespec="seconds"), "strategy": "PEAD", "side": "buy",
                   "symbol": s, "usd": SLOT_USD, "surprise_pct": sp, "order": o.get("id")})
        notes.append(f"PEAD buy {s} ${SLOT_USD:.0f} (EPS surprise {sp:+.1f}%)")
    short_slots = PEAD_SLOTS - len(st["pead_short"]) - sum(1 for p in st["pending"] if p["strategy"] == "pead_short")
    n = 0
    for s, sp in [x for x in shorts if x[0] not in held]:
        if n >= max(short_slots, 0):
            break
        try:
            o = short_qty(s, SLOT_USD, dry)
        except Exception as ex:
            o = None; notes.append(f"PEAD short {s} failed ({str(ex)[:50]})")
        if not o:
            continue
        n += 1
        st["pending"].append({"strategy": "pead_short", "symbol": s, "entry": today.isoformat(), "usd": SLOT_USD, "order_id": o.get("id")})
        log_trade({"time": datetime.now().isoformat(timespec="seconds"), "strategy": "PEAD_SHORT", "side": "short",
                   "symbol": s, "qty": o.get("qty"), "surprise_pct": sp, "order": o.get("id")})
        notes.append(f"PEAD short {s} {o.get('qty')} sh (EPS surprise {sp:+.1f}%)")
    if not dry:
        save_state(st)
        write_summary(st, notes)
    return notes


def write_summary(st: dict, notes: list[str]):
    acct = api("GET", "/v2/account")
    positions = api("GET", "/v2/positions")
    upl = sum(float(p["unrealized_pl"]) for p in positions)
    lines = [f"Paper trading bot: daily summary for {date.today():%a %b %d, %Y}", "",
             "Today: " + ("; ".join(notes) if notes else "no trades"), "",
             f"Open positions: T4 {len(st['t4'])}/{T4_SLOTS}, earnings drift long {len(st['pead'])}/{PEAD_SLOTS}, short {len(st['pead_short'])}/{PEAD_SLOTS}",
             f"Unrealized P/L on open positions: ${upl:,.2f}",
             f"Paper account equity: ${float(acct['equity']):,.2f} (started at $100,000)", ""]
    for p in positions:
        lines.append(f"  {p['symbol']}: {float(p['qty']):.4f} sh, P/L ${float(p['unrealized_pl']):,.2f} ({float(p['unrealized_plpc'])*100:+.1f}%)")
    os.makedirs(SUMMARY_DIR, exist_ok=True)
    path = os.path.join(SUMMARY_DIR, f"{date.today().isoformat()}.txt")
    open(path, "w").write("\n".join(lines) + "\n")


RUNS = os.path.join(HERE, "paper", "runs.json")


def auto(dry: bool) -> str:
    """Catch-up scheduler: called every ~10 min and at login. Runs each daily job once.
    open  : any time the market is open and today's open run hasn't happened yet.
    close : in the last 12 minutes of the session; if the close was missed entirely, a late run
            still sells expired positions and writes the summary but skips new earnings buys."""
    today = date.today()
    if os.path.exists(LOCK) and (datetime.now().timestamp() - os.path.getmtime(LOCK)) < 1800:
        return ""                                          # a previous run (e.g. the 500-stock scan) is still going
    open(LOCK, "w").write(str(os.getpid()))
    try:
        return _auto(dry, today)
    finally:
        os.remove(LOCK)


def _auto(dry: bool, today: date) -> str:
    if not is_trading_day(today):
        return ""
    runs = json.load(open(RUNS)) if os.path.exists(RUNS) else {}
    clock = api("GET", "/v2/clock")
    now = pd.Timestamp(clock["timestamp"])
    close_t = pd.Timestamp(clock["next_close"]) if clock["is_open"] else None
    msgs = []
    if clock["is_open"] and runs.get("open") != today.isoformat():
        msgs.append("open: " + ("; ".join(run_open(dry)) or "no trades"))
        runs["open"] = today.isoformat()
    session_over = (not clock["is_open"]) and now.tz_convert("America/New_York").hour >= 16
    near_close = clock["is_open"] and (close_t - now) <= pd.Timedelta(minutes=12)
    if runs.get("close") != today.isoformat() and (near_close or session_over):
        msgs.append(("close" if near_close else "late close (no new buys)") + ": "
                    + ("; ".join(run_close(dry, allow_buys=near_close)) or "no trades"))
        runs["close"] = today.isoformat()
    if not dry and msgs:
        os.makedirs(os.path.dirname(RUNS), exist_ok=True)
        json.dump(runs, open(RUNS, "w"))
    return " | ".join(msgs)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    dry = "--dry" in sys.argv
    if mode == "auto":
        out = auto(dry)
        if out:
            print(out)
        return
    if mode not in ("open", "close"):
        print(__doc__); return
    if not is_trading_day(date.today()):
        print("market closed today"); return
    notes = run_open(dry) if mode == "open" else run_close(dry)
    print(f"{mode}{' (dry)' if dry else ''}: " + ("; ".join(notes) if notes else "no trades"))


if __name__ == "__main__":
    main()
