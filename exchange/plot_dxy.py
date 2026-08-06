#!/usr/bin/env python3
"""Plot US Dollar Index (DXY): low-frequency monthly vs high-frequency weekly levels."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["font.sans-serif"] = [
    "PingFang SC",
    "Heiti SC",
    "Arial Unicode MS",
    "SimHei",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

WEEK_FREQ = "W-FRI"
DEFAULT_START = "2021-01-01"


def load_dxy_close(csv_path: Path) -> pd.Series:
    df = pd.read_csv(csv_path, parse_dates=["Date"])
    df["date"] = pd.to_datetime(df["Date"], utc=True).dt.tz_convert(None)
    return df.sort_values("date").set_index("date")["close"].astype(float)


def plot_dxy_lf_hf(close: pd.Series, output: Path, title: str | None = None) -> None:
    monthly = close.resample("ME").last().dropna()
    weekly = close.resample(WEEK_FREQ).last().dropna()

    fig, ax = plt.subplots(figsize=(12, 5.5), dpi=120)
    ax.plot(monthly.index, monthly.values, color="#2C3E50", linewidth=2.0, label="低频：月末指数")
    ax.plot(weekly.index, weekly.values, color="#E74C3C", linewidth=1.0, alpha=0.85, label="高频：周末指数")

    ax.set_title(title or "美元指数因子：低频 vs 高频（绝对水平）", fontsize=14, pad=12)
    ax.set_xlabel("日期")
    ax.set_ylabel("DXY 指数点位")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="upper left", framealpha=0.9)

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.autofmt_xdate(rotation=0, ha="center")
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot DXY low-frequency vs high-frequency levels")
    parser.add_argument("--input", default="dxy_yahoo.csv", help="Input CSV path")
    parser.add_argument("--output", default="dxy_chart.png", help="Output image path")
    parser.add_argument("--start", default=DEFAULT_START, help="Filter start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="Filter end date (YYYY-MM-DD)")
    args = parser.parse_args()

    close = load_dxy_close(Path(args.input))
    if args.start:
        close = close[close.index >= args.start]
    if args.end:
        close = close[close.index <= args.end]

    output = Path(args.output)
    title = "美元指数因子：低频 vs 高频（绝对水平）"
    if args.start or args.end:
        title += f"\n{args.start or close.index.min().date()} ~ {args.end or close.index.max().date()}"

    plot_dxy_lf_hf(close, output, title=title)
    print(f"Saved chart to: {output.resolve()}")


if __name__ == "__main__":
    main()
