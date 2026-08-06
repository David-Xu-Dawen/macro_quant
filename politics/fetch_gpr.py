#!/usr/bin/env python3
"""
下载日频地缘政治风险指数 (GPR) 时间序列。

依赖: pip install pandas openpyxl xlrd
数据源: Caldara & Iacoviello (2022) https://www.matteoiacoviello.com/gpr.htm

用法:
  python fetch_gpr.py
  python fetch_gpr.py --start 2018-01-01 --output data/gpr_daily.csv
"""

from __future__ import annotations

import argparse
import io
import sys
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

import pandas as pd

GPR_SOURCE_URL = "https://www.matteoiacoviello.com/gpr.htm"
DAILY_URL = "https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls"
DEFAULT_COLUMNS = ["GPRD", "GPRD_ACT", "GPRD_THREAT", "GPRD_MA7", "GPRD_MA30"]
DEFAULT_START = "2018-01-01"
DEFAULT_OUTPUT = Path("data/gpr_daily.csv")


def download_bytes(url: str, timeout: int = 60) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "gpr-fetch-script/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def fetch_gpr_daily(
    start: str = DEFAULT_START,
    end: str | None = None,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """从官网下载并清洗日频 GPR 数据。"""
    raw = download_bytes(DAILY_URL)
    df = pd.read_excel(io.BytesIO(raw))

    # Excel 前几行是变量说明，过滤掉
    df = df[df["var_name"].isna()].copy()
    df["date"] = pd.to_datetime(df["date"])

    selected = ["date"]
    cols = columns or DEFAULT_COLUMNS
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise ValueError(f"未知字段: {', '.join(missing)}")
    selected.extend(cols)

    out = df[selected].sort_values("date").reset_index(drop=True)

    start_ts = pd.Timestamp(start)
    out = out[out["date"] >= start_ts]
    if end:
        out = out[out["date"] <= pd.Timestamp(end)]

    return out.reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="下载日频 GPR 地缘风险指数")
    parser.add_argument("--start", default=DEFAULT_START, help=f"起始日期 (默认: {DEFAULT_START})")
    parser.add_argument("--end", help="结束日期，如 2025-12-31")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="输出路径")
    parser.add_argument(
        "--format", choices=["csv", "parquet", "json"], default="csv", help="输出格式"
    )
    parser.add_argument("--no-save", action="store_true", help="只打印摘要，不保存文件")
    args = parser.parse_args(argv)

    try:
        df = fetch_gpr_daily(start=args.start, end=args.end)
    except (urllib.error.URLError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"rows: {len(df)}")
    print(f"range: {df['date'].min().date()} -> {df['date'].max().date()}")
    print(df.tail(3).to_string(index=False))

    if not args.no_save:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.format == "csv":
            df.to_csv(args.output, index=False)
        elif args.format == "parquet":
            df.to_parquet(args.output, index=False)
        else:
            df.to_json(args.output, orient="records", date_format="iso", indent=2)
        print(f"saved: {args.output.resolve()}")

    print(f"\nSource: {GPR_SOURCE_URL}")
    print(f"Downloaded on: {datetime.now().date().isoformat()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
