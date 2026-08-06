#!/usr/bin/env python3
"""拉取因子暴露分析所需 14 类资产日度时间序列（默认自 2021-01-01 起）。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import akshare as ak
import pandas as pd
import requests
import yfinance as yf

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cbond_index import fetch_cbond_index, fetch_treasury_index

DEFAULT_START_DATE = "2021-01-01"
OUTPUT_DIR = Path(__file__).resolve().parent / "data"


def _filter_from_start(
    df: pd.DataFrame, start_date: str, date_col: str = "date"
) -> pd.DataFrame:
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    return out[out[date_col] >= start_date].sort_values(date_col).reset_index(drop=True)


def _normalize_columns(df: pd.DataFrame, asset: str, source: str) -> pd.DataFrame:
    rename_map = {
        "Date": "date",
        "tradeDate": "date",
        "日期": "date",
    }
    df = df.rename(columns=rename_map)
    if "date" not in df.columns:
        raise ValueError(f"{asset}: 缺少日期列")

    if "close" not in df.columns:
        for candidate in ("收盘", "Close", "value", "最新价"):
            if candidate in df.columns:
                df = df.rename(columns={candidate: "close"})
                break

    keep = ["date", "open", "high", "low", "close", "volume"]
    cols = [c for c in keep if c in df.columns]
    out = df[cols].copy()
    out["asset"] = asset
    out["source"] = source
    return out


def fetch_sse_index(name: str, symbol: str, start_date: str) -> pd.DataFrame:
    df = ak.stock_zh_index_daily(symbol=symbol)
    return _normalize_columns(
        _filter_from_start(df, start_date), name, f"akshare.stock_zh_index_daily({symbol})"
    )


def fetch_hsi(start_date: str) -> pd.DataFrame:
    df = ak.stock_hk_index_daily_sina(symbol="HSI")
    return _normalize_columns(
        _filter_from_start(df, start_date), "恒生指数", "akshare.stock_hk_index_daily_sina(HSI)"
    )


def fetch_bond_treasury(start_date: str) -> pd.DataFrame:
    df = fetch_treasury_index(indicator="财富", period="5Y")
    df = df.rename(columns={"value": "close"})
    return _normalize_columns(
        _filter_from_start(df, start_date), "中债国债", "chinabond.treasury_index"
    )


def fetch_bond_corporate(start_date: str) -> pd.DataFrame:
    df = fetch_cbond_index(index_category="企业债总指数", indicator="财富", period="总值")
    df = df.rename(columns={"value": "close"})
    return _normalize_columns(
        _filter_from_start(df, start_date), "中债企业债", "chinabond.corp_index"
    )


def fetch_convertible_bond(start_date: str) -> pd.DataFrame:
    url = "https://www.csindex.com.cn/csindex-home/perf/index-perf"
    params = {
        "indexCode": "000832",
        "startDate": start_date.replace("-", ""),
        "endDate": pd.Timestamp.today().strftime("%Y%m%d"),
    }
    resp = requests.get(url, params=params, headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
    resp.raise_for_status()
    payload = resp.json()
    if payload.get("code") != "200" or not payload.get("data"):
        raise RuntimeError(f"中证指数接口异常: {payload}")

    df = pd.DataFrame(payload["data"])
    return _normalize_columns(
        _filter_from_start(df, start_date, "tradeDate"), "中证转债", "csindex 000832"
    )


def fetch_nanhua_return(symbol: str, name: str, start_date: str) -> pd.DataFrame:
    url = f"https://www.nanhua.net/ianalysis/varietyindex/return/{symbol}.json"
    resp = requests.get(url, timeout=30)
    if resp.status_code != 200 or not resp.text.strip().startswith("["):
        raise RuntimeError(
            f"南华指数公开接口不可用 (HTTP {resp.status_code})。"
            "需使用 Wind / 南华官方数据服务，或等待 akshare 恢复相关接口。"
        )

    data = resp.json()
    df = pd.DataFrame(data, columns=["timestamp", "close"])
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
    return _normalize_columns(
        _filter_from_start(df, start_date), name, f"nanhua return {symbol}"
    )


def fetch_yfinance(name: str, ticker: str, start_date: str) -> pd.DataFrame:
    raw = yf.download(ticker, start=start_date, progress=False, auto_adjust=True)
    if raw.empty:
        raise RuntimeError(f"yfinance 未返回数据: {ticker}")

    df = raw.reset_index()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    df.columns = [str(col).lower() for col in df.columns]
    return _normalize_columns(
        _filter_from_start(df, start_date), name, f"yfinance {ticker}"
    )


def fetch_shfe_gold(start_date: str) -> pd.DataFrame:
    df = ak.futures_zh_daily_sina(symbol="AU0")
    return _normalize_columns(
        _filter_from_start(df, start_date), "沪金", "akshare.futures_zh_daily_sina(AU0)"
    )


def fetch_brent_oil(start_date: str) -> pd.DataFrame:
    df = ak.futures_foreign_hist(symbol="OIL")
    return _normalize_columns(
        _filter_from_start(df, start_date), "布伦特原油", "akshare.futures_foreign_hist(OIL)"
    )


def build_fetchers(start_date: str) -> dict[str, callable]:
    return {
        "上证50": lambda: fetch_sse_index("上证50", "sh000016", start_date),
        "沪深300": lambda: fetch_sse_index("沪深300", "sh000300", start_date),
        "中证500": lambda: fetch_sse_index("中证500", "sh000905", start_date),
        "中证1000": lambda: fetch_sse_index("中证1000", "sh000852", start_date),
        "恒生指数": lambda: fetch_hsi(start_date),
        "中债国债": lambda: fetch_bond_treasury(start_date),
        "中债企业债": lambda: fetch_bond_corporate(start_date),
        "中证转债": lambda: fetch_convertible_bond(start_date),
        "南华工业品": lambda: fetch_nanhua_return("NHII", "南华工业品", start_date),
        "南华农产品": lambda: fetch_nanhua_return("NHAI", "南华农产品", start_date),
        "布伦特原油": lambda: fetch_brent_oil(start_date),
        "沪金": lambda: fetch_shfe_gold(start_date),
        "标普500": lambda: fetch_yfinance("标普500", "^GSPC", start_date),
        "美元兑人民币": lambda: fetch_yfinance("美元兑人民币", "USDCNY=X", start_date),
    }


def save_outputs(
    frames: dict[str, pd.DataFrame], failed: dict[str, str], asset_order: list[str]
) -> None:
    raw_dir = OUTPUT_DIR / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    close_panel = None
    for asset in asset_order:
        df = frames.get(asset)
        if df is None:
            cached = raw_dir / f"{asset}.csv"
            if cached.exists():
                df = pd.read_csv(cached, parse_dates=["date"])
            else:
                continue

        df.to_csv(raw_dir / f"{asset}.csv", index=False, encoding="utf-8-sig")

        series = df[["date", "close"]].rename(columns={"close": asset})
        close_panel = series if close_panel is None else close_panel.merge(series, on="date", how="outer")

    if close_panel is not None:
        close_panel = close_panel.sort_values("date").reset_index(drop=True)
        close_panel.to_csv(OUTPUT_DIR / "combined_close.csv", index=False, encoding="utf-8-sig")
        close_panel.to_excel(OUTPUT_DIR / "combined_close.xlsx", index=False)

    summary_rows = []
    for asset in asset_order:
        df = frames.get(asset)
        if df is None:
            cached = raw_dir / f"{asset}.csv"
            if cached.exists():
                df = pd.read_csv(cached, parse_dates=["date"])
        if df is not None:
            summary_rows.append(
                {
                    "资产": asset,
                    "状态": "成功",
                    "起始日": df["date"].min().date(),
                    "结束日": df["date"].max().date(),
                    "样本数": len(df),
                    "数据源": df["source"].iloc[0],
                    "备注": "本地缓存" if asset not in frames else "",
                }
            )
        else:
            summary_rows.append(
                {
                    "资产": asset,
                    "状态": "失败",
                    "起始日": None,
                    "结束日": None,
                    "样本数": 0,
                    "数据源": "",
                    "备注": failed.get(asset, ""),
                }
            )

    pd.DataFrame(summary_rows).to_csv(OUTPUT_DIR / "summary.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser(description="拉取资产日度时间序列")
    parser.add_argument("--start", default=DEFAULT_START_DATE, help="起始日期，默认 2021-01-01")
    args = parser.parse_args()

    fetchers = build_fetchers(args.start)
    frames: dict[str, pd.DataFrame] = {}
    failed: dict[str, str] = {}

    for asset, fn in fetchers.items():
        print(f"正在拉取: {asset} ...", flush=True)
        try:
            frames[asset] = fn()
            print(f"  ✓ {len(frames[asset])} 条", flush=True)
        except Exception as exc:
            failed[asset] = str(exc)
            print(f"  ✗ {exc}", flush=True)

    save_outputs(frames, failed, list(fetchers.keys()))

    print(f"\n完成: 成功 {len(frames)}/{len(fetchers)}")
    print(f"输出目录: {OUTPUT_DIR}")
    if failed:
        print("未成功资产:")
        for asset, msg in failed.items():
            print(f"  - {asset}: {msg}")


if __name__ == "__main__":
    main()
