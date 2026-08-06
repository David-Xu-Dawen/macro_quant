"""从 akshare 拉取多资产日线，统一落盘为 data/raw/{asset}.csv。"""

from __future__ import annotations

import argparse
import time
from typing import Callable

import akshare as ak
import pandas as pd

from config import ASSETS, END_DATE, RAW_DIR, START_DATE


def _to_date(s) -> pd.Timestamp:
    return pd.to_datetime(s).normalize()


def _normalize_ohlcv(df: pd.DataFrame, date_col: str, close_col: str) -> pd.DataFrame:
    """统一成 date/open/high/low/close/volume 格式。"""
    out = df.copy()
    rename = {}
    for src, dst in [
        (date_col, "date"),
        (close_col, "close"),
        ("开盘", "open"),
        ("开盘价", "open"),
        ("最高", "high"),
        ("最高价", "high"),
        ("最低", "low"),
        ("最低价", "low"),
        ("收盘", "close"),
        ("收盘价", "close"),
        ("最新价", "close"),
        ("latest", "close"),
        ("成交量", "volume"),
        ("volume", "volume"),
        ("持仓量", "open_interest"),
    ]:
        if src in out.columns and dst not in out.columns:
            rename[src] = dst
    out = out.rename(columns=rename)
    out["date"] = out["date"].map(_to_date)

    for col in ["open", "high", "low", "close", "volume"]:
        if col not in out.columns:
            out[col] = pd.NA if col != "close" else out["close"]

    keep = [c for c in ["date", "open", "high", "low", "close", "volume", "open_interest", "amount"] if c in out.columns]
    out = out[keep].sort_values("date").drop_duplicates("date", keep="last")
    out = out.dropna(subset=["close"])
    return out.reset_index(drop=True)


def _retry(fn: Callable, retries: int = 4, sleep: float = 1.5):
    last = None
    for i in range(retries):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(sleep * (i + 1))
    raise last


def fetch_em_index(symbol: str, start: str, end: str | None) -> pd.DataFrame:
    start_s = start.replace("-", "")
    end_s = (end or "20500101").replace("-", "")

    def _call():
        return ak.stock_zh_index_daily_em(symbol=symbol, start_date=start_s, end_date=end_s)

    try:
        df = _retry(_call)
        return _normalize_ohlcv(df, "date", "close")
    except Exception:
        # 东方财富失败时回退新浪（无起止过滤，后处理裁剪）
        sina_symbol = symbol if symbol.startswith(("sh", "sz", "bj")) else f"sh{symbol}"
        df = _retry(lambda: ak.stock_zh_index_daily(symbol=sina_symbol))
        out = _normalize_ohlcv(df, "date", "close")
        out = out[out["date"] >= _to_date(start)]
        if end:
            out = out[out["date"] <= _to_date(end)]
        return out.reset_index(drop=True)


def fetch_hk_index(symbol: str, start: str, end: str | None) -> pd.DataFrame:
    try:
        df = _retry(lambda: ak.stock_hk_index_daily_em(symbol=symbol))
        out = _normalize_ohlcv(df, "date", "latest" if "latest" in df.columns else "close")
    except Exception:
        df = _retry(lambda: ak.stock_hk_index_daily_sina(symbol=symbol))
        out = _normalize_ohlcv(df, "date", "close")
    out = out[out["date"] >= _to_date(start)]
    if end:
        out = out[out["date"] <= _to_date(end)]
    return out.reset_index(drop=True)


def fetch_us_index(symbol: str, start: str, end: str | None) -> pd.DataFrame:
    df = _retry(lambda: ak.index_us_stock_sina(symbol=symbol))
    out = _normalize_ohlcv(df, "date", "close")
    out = out[out["date"] >= _to_date(start)]
    if end:
        out = out[out["date"] <= _to_date(end)]
    return out.reset_index(drop=True)


def fetch_futures_sina(symbol: str, start: str, end: str | None) -> pd.DataFrame:
    start_s = start.replace("-", "")
    end_s = (end or "22220101").replace("-", "")
    df = _retry(lambda: ak.futures_main_sina(symbol=symbol, start_date=start_s, end_date=end_s))
    return _normalize_ohlcv(df, "日期", "收盘价")


def fetch_cbond_comp(start: str, end: str | None) -> pd.DataFrame:
    df = _retry(lambda: ak.bond_new_composite_index_cbond(indicator="财富", period="总值"))
    out = df.rename(columns={"date": "date", "value": "close"})
    out["date"] = out["date"].map(_to_date)
    out["open"] = out["high"] = out["low"] = out["close"]
    out["volume"] = pd.NA
    out = out[["date", "open", "high", "low", "close", "volume"]]
    out = out[out["date"] >= _to_date(start)]
    if end:
        out = out[out["date"] <= _to_date(end)]
    return out.sort_values("date").drop_duplicates("date").reset_index(drop=True)


FETCHERS = {
    "em_index": lambda meta, start, end: fetch_em_index(meta["symbol"], start, end),
    "hk_index": lambda meta, start, end: fetch_hk_index(meta["symbol"], start, end),
    "us_index": lambda meta, start, end: fetch_us_index(meta["symbol"], start, end),
    "futures_sina": lambda meta, start, end: fetch_futures_sina(meta["symbol"], start, end),
    "cbond_comp": lambda meta, start, end: fetch_cbond_comp(start, end),
}


def fetch_one(asset_id: str, start: str, end: str | None) -> pd.DataFrame:
    meta = ASSETS[asset_id]
    fetcher = FETCHERS[meta["source"]]
    df = fetcher(meta, start, end)
    df.insert(0, "asset", asset_id)
    df.insert(1, "cn_name", meta["cn_name"])
    df.insert(2, "asset_class", meta["asset_class"])
    return df


def main():
    parser = argparse.ArgumentParser(description="拉取 akshare 多资产日线数据")
    parser.add_argument("--start", default=START_DATE)
    parser.add_argument("--end", default=END_DATE)
    parser.add_argument("--assets", nargs="*", default=list(ASSETS.keys()))
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    summary = []

    for asset_id in args.assets:
        print(f"[fetch] {asset_id} ({ASSETS[asset_id]['cn_name']}) ...", flush=True)
        try:
            df = fetch_one(asset_id, args.start, args.end)
            path = RAW_DIR / f"{asset_id}.csv"
            df.to_csv(path, index=False)
            summary.append(
                {
                    "asset": asset_id,
                    "cn_name": ASSETS[asset_id]["cn_name"],
                    "rows": len(df),
                    "start": str(df["date"].min().date()) if len(df) else None,
                    "end": str(df["date"].max().date()) if len(df) else None,
                    "path": str(path),
                    "status": "ok",
                }
            )
            print(f"  -> {len(df)} rows, {df['date'].min().date()} ~ {df['date'].max().date()}")
        except Exception as exc:  # noqa: BLE001
            summary.append(
                {
                    "asset": asset_id,
                    "cn_name": ASSETS[asset_id]["cn_name"],
                    "rows": 0,
                    "start": None,
                    "end": None,
                    "path": None,
                    "status": f"fail: {exc}",
                }
            )
            print(f"  !! FAIL: {exc}")
        time.sleep(0.6)

    summary_df = pd.DataFrame(summary)
    summary_path = RAW_DIR / "_fetch_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print("\n=== summary ===")
    print(summary_df.to_string(index=False))
    print(f"\nsaved: {summary_path}")


if __name__ == "__main__":
    main()
