"""高频拟合宏观因子相关性：读取周频面板并按区间计算 Pearson 矩阵。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

FACTOR_LABELS = [
    "增长因子",
    "通胀因子",
    "利率因子",
    "信用因子",
    "汇率因子",
    "地缘因子",
]

ROOT = Path(__file__).resolve().parent.parent
PANEL_CSV = ROOT / "macro_hf_factor_weekly.csv"
CORR_JSON = ROOT / "macro_hf_factor_corr.json"


def load_panel() -> pd.DataFrame:
    if not PANEL_CSV.exists():
        raise FileNotFoundError("macro_hf_factor_weekly.csv not found，请先运行 plot_macro_hf_corr.py")
    df = pd.read_csv(PANEL_CSV, parse_dates=["week"])
    return df.set_index("week")[FACTOR_LABELS]


def available_weeks(panel: pd.DataFrame | None = None) -> list[str]:
    panel = panel if panel is not None else load_panel()
    return panel.dropna(how="any").index.strftime("%Y-%m-%d").tolist()


def compute_hf_corr(start: str, end: str, panel: pd.DataFrame | None = None) -> dict:
    panel = panel if panel is not None else load_panel()
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if start_ts > end_ts:
        raise ValueError("开始周不能晚于结束周")

    subset = panel.loc[start_ts:end_ts].dropna(how="any")
    if len(subset) < 12:
        raise ValueError(f"有效样本不足，至少需要 12 周（当前 {len(subset)} 周）")

    corr = subset.corr(method="pearson")
    weeks = available_weeks(panel)
    return {
        "labels": FACTOR_LABELS,
        "periods": weeks,
        "weeks": weeks,
        "start": subset.index.min().strftime("%Y-%m-%d"),
        "end": subset.index.max().strftime("%Y-%m-%d"),
        "n_periods": len(subset),
        "n_weeks": len(subset),
        "freq": "W-FRI",
        "corr": [[round(v, 4) for v in row] for row in corr.values.tolist()],
    }
