"""宏观因子相关性：读取月度面板并按区间计算 Pearson 矩阵。"""

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
PANEL_CSV = ROOT / "macro_factor_monthly.csv"
CORR_JSON = ROOT / "macro_factor_corr.json"


def load_panel() -> pd.DataFrame:
    if not PANEL_CSV.exists():
        raise FileNotFoundError("macro_factor_monthly.csv not found，请先运行 plot_macro_factor_corr.py")
    df = pd.read_csv(PANEL_CSV)
    return df.set_index("ym")[FACTOR_LABELS]


def available_months(panel: pd.DataFrame | None = None) -> list[str]:
    panel = panel if panel is not None else load_panel()
    return panel.dropna(how="any").index.astype(str).tolist()


def compute_corr(start: str, end: str, panel: pd.DataFrame | None = None) -> dict:
    panel = panel if panel is not None else load_panel()
    if start > end:
        raise ValueError("开始月份不能晚于结束月份")

    subset = panel.loc[start:end].dropna(how="any")
    if len(subset) < 3:
        raise ValueError(f"有效样本不足，至少需要 3 个月（当前 {len(subset)} 个月）")

    corr = subset.corr(method="pearson")
    return {
        "labels": FACTOR_LABELS,
        "months": available_months(panel),
        "start": str(subset.index.min()),
        "end": str(subset.index.max()),
        "n_months": len(subset),
        "corr": [[round(v, 4) for v in row] for row in corr.values.tolist()],
    }
