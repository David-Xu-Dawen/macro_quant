#!/usr/bin/env python3
"""Fetch US Dollar Index (DXY) time series from Yahoo Finance."""

import argparse
from pathlib import Path

import pandas as pd
import yfinance as yf

# Yahoo Finance ticker for ICE US Dollar Index
DXY_TICKER = "DX-Y.NYB"


def fetch_dxy(start: str | None = None, end: str | None = None) -> pd.DataFrame:
    ticker = yf.Ticker(DXY_TICKER)
    df = ticker.history(start=start, end=end, auto_adjust=False)

    if df.empty:
        raise RuntimeError(
            f"No data returned for {DXY_TICKER}. "
            "Check your network connection or date range."
        )

    df.index.name = "Date"
    df = df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    return df[["open", "high", "low", "close", "volume"]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Download DXY time series from Yahoo Finance")
    parser.add_argument("--start", default="2000-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="End date (YYYY-MM-DD), exclusive")
    parser.add_argument(
        "--output",
        default="dxy_yahoo.csv",
        help="Output CSV path (default: dxy_yahoo.csv)",
    )
    args = parser.parse_args()

    df = fetch_dxy(start=args.start, end=args.end)
    output = Path(args.output)
    df.to_csv(output)

    print(f"Ticker:   {DXY_TICKER}")
    print(f"Rows:     {len(df)}")
    print(f"From:     {df.index.min().date()}")
    print(f"To:       {df.index.max().date()}")
    print(f"Saved to: {output.resolve()}")
    print()
    print(df.tail(5).to_string())


if __name__ == "__main__":
    main()
