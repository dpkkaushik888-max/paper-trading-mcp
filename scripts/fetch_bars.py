"""Fetch full daily kline history into data/bars/{com,us}/<SYMBOL>.csv (gitignored).
Ported from the S35 deep-dive analysis/fetch_data.py. Usage: python scripts/fetch_bars.py [--venue com|us]"""
import argparse
import os
import time

import pandas as pd
import requests

from trading_engine.paper.config import UNIVERSE

VENUES = {"us": "https://api.binance.us/api/v3/klines", "com": "https://api.binance.com/api/v3/klines"}


def fetch(url, sym, start=1483228800000):  # 2017-01-01
    rows = []
    while True:
        r = requests.get(url, params={"symbol": sym, "interval": "1d", "startTime": start, "limit": 1000}, timeout=20)
        if r.status_code != 200:
            return rows, r.status_code
        raw = r.json()
        if not raw:
            break
        rows += raw
        if len(raw) < 1000:
            break
        start = raw[-1][0] + 1
        time.sleep(0.2)
    return rows, 200


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--venue", choices=sorted(VENUES), action="append")
    ap.add_argument("--out", default="data/bars")
    a = ap.parse_args()
    now_ms = int(time.time() * 1000)
    for v in a.venue or sorted(VENUES):
        os.makedirs(f"{a.out}/{v}", exist_ok=True)
        for sym in UNIVERSE:
            rows, st = fetch(VENUES[v], sym)
            rows = [x for x in rows if x[6] <= now_ms]   # closed bars only
            if not rows:
                print(v, sym, "NO DATA", st)
                continue
            df = pd.DataFrame([[x[0], *map(float, x[1:6])] for x in rows],
                              columns=["t", "Open", "High", "Low", "Close", "Volume"])
            df["date"] = pd.to_datetime(df.t, unit="ms").dt.strftime("%Y-%m-%d")
            df[["date", "Open", "High", "Low", "Close", "Volume"]].drop_duplicates("date").to_csv(
                f"{a.out}/{v}/{sym}.csv", index=False)
            print(v, sym, len(df), df.date.iloc[0], df.date.iloc[-1])


if __name__ == "__main__":
    main()
