"""周频因子波动监控：滚动波动分位 + 本周冲击标记。

冲击与波动均基于与因子暴露相同的周度环比/变化面板（MoM），
避免对 YoY/水平热力图序列再差分后与暴露 β 量纲错配。
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from factor_exposure import load_latest_json as load_exposure_latest
from factor_exposure import load_mom_panel
from hf_factor_corr import FACTOR_LABELS

DEFAULT_WINDOW = 13
SHOCK_Z = 2.0


def _percentile_rank(history: pd.Series, value: float) -> float | None:
    hist = history.dropna()
    if hist.empty or not math.isfinite(value):
        return None
    # 当前值在历史中的分位（含当前）
    return round(float((hist <= value).mean() * 100.0), 1)


def _level_label(pct: float | None) -> str:
    if pct is None:
        return "未知"
    if pct >= 90:
        return "很高"
    if pct >= 75:
        return "偏高"
    if pct >= 40:
        return "中性"
    return "偏低"


def load_mom_changes(end: str | None = None) -> pd.DataFrame:
    """加载与暴露一致的周度 MoM 面板（热力图 6 因子）。"""
    mom = load_mom_panel()
    missing = [c for c in FACTOR_LABELS if c not in mom.columns]
    if missing:
        raise ValueError(f"MoM 面板缺少因子列: {', '.join(missing)}")
    panel = mom[FACTOR_LABELS].dropna(how="any")
    if end is not None:
        panel = panel.loc[: pd.Timestamp(end)]
    return panel


def compute_vol_monitor(
    *,
    window: int = DEFAULT_WINDOW,
    shock_z: float = SHOCK_Z,
    end: str | None = None,
) -> dict[str, Any]:
    # 已是周度环比/变化，不再对水平或同比做 diff
    chg = load_mom_changes(end=end)
    if len(chg) < window + 5:
        raise ValueError(f"周频样本不足，至少需要 {window + 5} 周（当前 {len(chg)}）")

    roll_vol = chg.rolling(window).std(ddof=0)

    as_of = chg.index[-1]
    prev = chg.index[-2] if len(chg) >= 2 else None
    factors_out: list[dict[str, Any]] = []
    shocks: list[dict[str, Any]] = []

    for name in FACTOR_LABELS:
        vol_hist = roll_vol[name].dropna()
        if vol_hist.empty:
            continue
        cur_vol = float(vol_hist.iloc[-1])
        pct = _percentile_rank(vol_hist, cur_vol)
        last_chg = float(chg[name].iloc[-1])
        # 用同期滚动波动做 z；波动为 0 则跳过
        z = last_chg / cur_vol if cur_vol > 1e-12 else 0.0
        is_shock = abs(z) >= shock_z
        item = {
            "factor": name,
            "vol": round(cur_vol, 6),
            "vol_percentile": pct,
            "vol_level": _level_label(pct),
            "week_change": round(last_chg, 6),
            "shock_z": round(float(z), 2),
            "is_shock": bool(is_shock),
        }
        factors_out.append(item)
        if is_shock:
            shocks.append(
                {
                    "factor": name,
                    "week_change": item["week_change"],
                    "shock_z": item["shock_z"],
                    "direction": "上行" if last_chg > 0 else "下行",
                }
            )

    factors_out.sort(key=lambda x: (x["vol_percentile"] is not None, x["vol_percentile"] or -1), reverse=True)
    shocks.sort(key=lambda x: abs(x["shock_z"]), reverse=True)

    max_pct = max((f["vol_percentile"] for f in factors_out if f["vol_percentile"] is not None), default=None)
    n_high = sum(1 for f in factors_out if (f["vol_percentile"] or 0) >= 75)
    if shocks or (max_pct is not None and max_pct >= 90) or n_high >= 3:
        status = "偏高"
        status_note = "周频因子波动/冲击偏高，组合风险环境更躁，宜收紧风险预算。"
    elif max_pct is not None and max_pct >= 75:
        status = "留意"
        status_note = "部分因子波动已到偏高分位，建议结合暴露表关注敏感资产。"
    else:
        status = "平稳"
        status_note = "当前周频因子波动大致处于历史常态。"

    # 暴露压力：|β| × 因子当前滚动波动（与暴露同为 MoM 口径）
    asset_pressure: list[dict[str, Any]] = []
    try:
        exp = load_exposure_latest()
        matrix = exp.get("matrix") or {}
        vols = {f["factor"]: f["vol"] for f in factors_out}
        # 用各因子波动的相对强度（除以截面中位数）避免量纲差过大
        vol_vals = np.array([vols[k] for k in FACTOR_LABELS if k in vols], dtype=float)
        scale = float(np.nanmedian(vol_vals)) if len(vol_vals) else 1.0
        if not math.isfinite(scale) or scale < 1e-12:
            scale = 1.0
        rows = []
        for asset, betas in matrix.items():
            score = 0.0
            drivers = []
            for fac in FACTOR_LABELS:
                if fac not in betas or fac not in vols:
                    continue
                b = abs(float(betas[fac]))
                v = float(vols[fac]) / scale
                contrib = b * v
                score += contrib
                if contrib > 0:
                    drivers.append((contrib, fac, float(betas[fac])))
            drivers.sort(reverse=True)
            top = drivers[0] if drivers else None
            rows.append(
                {
                    "asset": asset,
                    "pressure": round(score, 4),
                    "top_factor": top[1] if top else None,
                    "top_beta": round(top[2], 3) if top else None,
                }
            )
        rows.sort(key=lambda x: x["pressure"], reverse=True)
        asset_pressure = rows[:8]
    except FileNotFoundError:
        asset_pressure = []

    forecast = None
    try:
        from vol_forecast import predict_latest

        forecast = predict_latest(end=as_of.strftime("%Y-%m-%d"))
    except Exception as exc:
        forecast = {"error": str(exc), "note": "树模型暂不可用"}

    return {
        "as_of": as_of.strftime("%Y-%m-%d"),
        "prev_week": prev.strftime("%Y-%m-%d") if prev is not None else None,
        "window_weeks": window,
        "shock_z": shock_z,
        "status": status,
        "status_note": status_note,
        "factors": factors_out,
        "shocks": shocks,
        "asset_pressure": asset_pressure,
        "forecast": forecast,
        "method_note": (
            "波动基于与因子暴露相同的周度环比/变化（MoM）滚动标准差；"
            "分位对比该滚动波动的历史分布；冲击=本周 MoM/当前滚动波动。"
            "树模型为未来4周高波动概率的第二意见（需用 MoM 口径重训）。"
        ),
    }
