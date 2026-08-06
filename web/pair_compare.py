"""高低频因子对对比：解释月频矩阵与周频矩阵相关为何不同。"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from factor_corr import FACTOR_LABELS, load_panel as load_lf
from hf_factor_corr import load_panel as load_hf


def _zscore(s: pd.Series) -> pd.Series:
    std = float(s.std(ddof=0))
    if not math.isfinite(std) or std < 1e-12:
        return s * 0.0
    return (s - s.mean()) / std


def _corr(a: pd.Series, b: pd.Series) -> float | None:
    aligned = pd.concat([a, b], axis=1).dropna()
    if len(aligned) < 3:
        return None
    v = float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))
    return None if not math.isfinite(v) else round(v, 4)


def _rolling_corr(a: pd.Series, b: pd.Series, window: int) -> list[dict[str, Any]]:
    df = pd.concat({"a": a, "b": b}, axis=1).dropna()
    if len(df) < window:
        return []
    rc = df["a"].rolling(window).corr(df["b"])
    out = []
    for idx, val in rc.dropna().items():
        if not math.isfinite(float(val)):
            continue
        if isinstance(idx, pd.Timestamp):
            key = idx.strftime("%Y-%m-%d")
        else:
            key = str(idx)
        out.append({"t": key, "corr": round(float(val), 4)})
    return out


def _series_payload(dates: list[str], a: pd.Series, b: pd.Series) -> dict[str, Any]:
    a_z = _zscore(a)
    b_z = _zscore(b)
    return {
        "dates": dates,
        "a": [None if pd.isna(x) else round(float(x), 6) for x in a.tolist()],
        "b": [None if pd.isna(x) else round(float(x), 6) for x in b.tolist()],
        "a_z": [None if pd.isna(x) else round(float(x), 4) for x in a_z.tolist()],
        "b_z": [None if pd.isna(x) else round(float(x), 4) for x in b_z.tolist()],
    }


def _diagnose(lf_corr: float | None, hf_corr: float | None, hf_m_corr: float | None) -> str:
    if lf_corr is None or hf_corr is None:
        return "样本不足，无法比较高低频相关性。"
    delta = hf_corr - lf_corr
    lines = [
        f"月频相关 {lf_corr:+.2f}，周频相关 {hf_corr:+.2f}，差值 Δ(周-月)={delta:+.2f}。"
    ]
    if hf_m_corr is not None:
        lines.append(f"把高频因子按月末抽样后再算相关为 {hf_m_corr:+.2f}。")
        d_freq = abs(hf_m_corr - lf_corr)
        d_build = abs(hf_m_corr - hf_corr)
        if d_freq < 0.15 and abs(delta) >= 0.2:
            lines.append("月末抽样后接近月频 → 差异主要来自采样频率/周内波动，而非因子定义大不相同。")
        elif d_build < 0.15 and abs(delta) >= 0.2:
            lines.append("月末抽样后仍接近周频 → 差异主要来自低频与高频因子构造不同（代理变量或拟合方式差异）。")
        elif abs(delta) < 0.15:
            lines.append("高低频相关接近，矩阵差异不大。")
        else:
            lines.append("频率效应与构造差异都有贡献：月末抽样落在两者之间，需同时看周内噪声与因子定义。")
    else:
        lines.append("高频月末抽样样本不足，仅能对比原始月频与周频相关。")
    return " ".join(lines)


def compare_pair(
    factor_a: str,
    factor_b: str,
    *,
    start: str | None = None,
    end: str | None = None,
) -> dict[str, Any]:
    if factor_a not in FACTOR_LABELS or factor_b not in FACTOR_LABELS:
        raise ValueError(f"因子须为: {', '.join(FACTOR_LABELS)}")
    if factor_a == factor_b:
        raise ValueError("请选择两个不同的因子")

    lf = load_lf()[[factor_a, factor_b]].dropna(how="any")
    hf = load_hf()[[factor_a, factor_b]].dropna(how="any")

    # 统一用日历区间；未指定则取两者重叠区间
    lf.index = lf.index.astype(str)
    if start is None or end is None:
        lf_start, lf_end = lf.index.min(), lf.index.max()
        hf_start = hf.index.min().strftime("%Y-%m")
        hf_end = hf.index.max().strftime("%Y-%m")
        start = max(lf_start, hf_start)
        end = min(lf_end, hf_end)

    # 月频：YYYY-MM；周频：落在 [start-01, end月末]
    start_m = str(start)[:7]
    end_m = str(end)[:7]
    lf_sub = lf.loc[start_m:end_m]
    if len(lf_sub) < 3:
        raise ValueError(f"月频有效样本不足（当前 {len(lf_sub)}）")

    start_ts = pd.Timestamp(f"{start_m}-01")
    end_ts = pd.Timestamp(f"{end_m}-01") + pd.offsets.MonthEnd(0)
    hf_sub = hf.loc[start_ts:end_ts]
    if len(hf_sub) < 12:
        raise ValueError(f"周频有效样本不足（当前 {len(hf_sub)}）")

    # 高频 → 月末抽样（每月最后一个有效周）
    hf_m = hf_sub.copy()
    hf_m["ym"] = hf_m.index.to_period("M").astype(str)
    hf_month = hf_m.groupby("ym").tail(1).set_index("ym")[[factor_a, factor_b]]
    # 与月频共同月份对齐
    common_m = lf_sub.index.intersection(hf_month.index)
    lf_common = lf_sub.loc[common_m]
    hf_month_common = hf_month.loc[common_m]

    lf_corr = _corr(lf_sub[factor_a], lf_sub[factor_b])
    hf_corr = _corr(hf_sub[factor_a], hf_sub[factor_b])
    hf_m_corr = _corr(hf_month_common[factor_a], hf_month_common[factor_b]) if len(common_m) >= 3 else None
    # 同月份上低频相关（公平对比）
    lf_common_corr = _corr(lf_common[factor_a], lf_common[factor_b]) if len(common_m) >= 3 else lf_corr

    lf_dates = lf_sub.index.astype(str).tolist()
    hf_dates = hf_sub.index.strftime("%Y-%m-%d").tolist()

    return {
        "factor_a": factor_a,
        "factor_b": factor_b,
        "start": start_m,
        "end": end_m,
        "lf": {
            "n": len(lf_sub),
            "corr": lf_corr,
            "rolling_corr": _rolling_corr(lf_sub[factor_a], lf_sub[factor_b], 12),
            **_series_payload(lf_dates, lf_sub[factor_a], lf_sub[factor_b]),
        },
        "hf": {
            "n": len(hf_sub),
            "corr": hf_corr,
            "rolling_corr": _rolling_corr(hf_sub[factor_a], hf_sub[factor_b], 52),
            **_series_payload(hf_dates, hf_sub[factor_a], hf_sub[factor_b]),
        },
        "hf_month_end": {
            "n": int(len(common_m)),
            "corr": hf_m_corr,
            "lf_corr_same_months": lf_common_corr,
            "dates": common_m.astype(str).tolist(),
            "lf_a_z": _zscore(lf_common[factor_a]).round(4).tolist() if len(common_m) else [],
            "lf_b_z": _zscore(lf_common[factor_b]).round(4).tolist() if len(common_m) else [],
            "hf_a_z": _zscore(hf_month_common[factor_a]).round(4).tolist() if len(common_m) else [],
            "hf_b_z": _zscore(hf_month_common[factor_b]).round(4).tolist() if len(common_m) else [],
        },
        "delta": {
            "hf_minus_lf": None if lf_corr is None or hf_corr is None else round(hf_corr - lf_corr, 4),
            "hf_month_minus_lf": None
            if lf_common_corr is None or hf_m_corr is None
            else round(hf_m_corr - lf_common_corr, 4),
        },
        "diagnosis": _diagnose(lf_corr, hf_corr, hf_m_corr),
    }
