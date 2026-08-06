#!/usr/bin/env python3
"""绘制地缘政治风险指数 (GPR)：低频月度 vs 高频周度绝对水平。"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).parent
DEFAULT_INPUT = ROOT / "data" / "gpr_daily.csv"
DEFAULT_OUTPUT = ROOT / "gpr_chart.png"
WEEK_FREQ = "W-FRI"
DEFAULT_START = "2021-01-01"

plt.rcParams["font.sans-serif"] = [
    "PingFang SC",
    "Heiti SC",
    "Arial Unicode MS",
    "SimHei",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


def load_gprd(csv_path: Path) -> pd.Series:
    df = pd.read_csv(csv_path, parse_dates=["date"])
    return df.sort_values("date").set_index("date")["GPRD"].astype(float)


def plot_gpr_lf_hf(gprd: pd.Series, output: Path, title: str | None = None) -> None:
    monthly = gprd.resample("ME").mean().dropna()
    weekly = gprd.resample(WEEK_FREQ).mean().dropna()

    fig, ax = plt.subplots(figsize=(12, 5.5), dpi=120)
    ax.plot(monthly.index, monthly.values, color="#2C3E50", linewidth=2.0, label="低频：月均 GPR")
    ax.plot(weekly.index, weekly.values, color="#E74C3C", linewidth=1.0, alpha=0.85, label="高频：周均 GPR")

    ax.set_title(title or "地缘因子：低频 vs 高频（GPR 绝对水平）", fontsize=14, pad=12)
    ax.set_xlabel("日期")
    ax.set_ylabel("GPR 指数")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="upper left", framealpha=0.9)

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.autofmt_xdate(rotation=0, ha="center")
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="绘制 GPR 低频 vs 高频绝对水平图")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="输入 CSV")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="输出图片")
    parser.add_argument("--start", default=DEFAULT_START, help="起始日期 YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="结束日期 YYYY-MM-DD")
    args = parser.parse_args()

    gprd = load_gprd(args.input)
    if args.start:
        gprd = gprd[gprd.index >= args.start]
    if args.end:
        gprd = gprd[gprd.index <= args.end]
    if gprd.empty:
        raise SystemExit("筛选后无数据，请检查日期范围")

    title = "地缘因子：低频 vs 高频（GPR 绝对水平）— Caldara & Iacoviello"
    if args.start or args.end:
        title += f"\n{args.start or gprd.index.min().date()} ~ {args.end or gprd.index.max().date()}"

    plot_gpr_lf_hf(gprd, args.output, title=title)
    print(f"已保存: {args.output.resolve()}")
    print(f"样本: {len(gprd)} 日, {gprd.index.min().date()} ~ {gprd.index.max().date()}")


if __name__ == "__main__":
    main()
