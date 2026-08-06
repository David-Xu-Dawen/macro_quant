#!/usr/bin/env python3
"""
流动性（mobility）因子全流程：
1. 低频 mobility_factor.csv：M2 同比 - 社融存量同比
2. 高频日频 mobility_high_freq_daily.csv：市盈率代理（默认同联网更新）
3. 多变量滞后回归（月频对数同比 ~ 低频流动性因子）
4. 高频化流动性因子（先日环比再加权合成，再滚动同比）
5. 导出图表

联网更新（UPDATE_DATA=True）见 fetch_online_data.py：
- M2 同比：东方财富
- 高频 PE：乐咕乐股 沪深300 / 中证1000（映射为申万大盘/小盘列名）
- 社融存量同比：仍读本地 Wind 导出（公开免费源不稳定）
"""

from __future__ import annotations

import itertools
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from fetch_online_data import (
    PE_LARGE_SYMBOL,
    PE_SMALL_SYMBOL,
    build_mobility_monthly,
    fetch_pe_daily,
    load_sf_yoy_from_wind,
)

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
WIND_CSV = ROOT / "中国_M2_同比.csv"

UPDATE_DATA = True
USE_ONLINE = True  # False 时回退仅用本地 Wind 导出
RUN_REGRESSION = True
BUILD_HF = True
MAKE_PLOTS = True

ASSETS = ["申万大盘市盈率", "申万小盘市盈率"]
ASSET_COLORS = {
    "申万大盘市盈率": "#1e40af",
    "申万小盘市盈率": "#dc2626",
}

LAG_RANGE = range(0, 4)
TRADING_DAYS_YEAR = 252
TRADING_DAYS_MONTH = 21
NAV_BASE = 100.0

MOBILITY_FACTOR_CSV = ROOT / "mobility_factor.csv"
HIGH_FREQ_CSV = ROOT / "mobility_high_freq_daily.csv"


def setup_chinese_font() -> None:
    plt.rcParams["font.sans-serif"] = [
        "PingFang SC", "Heiti SC", "STHeiti", "SimHei", "Arial Unicode MS", "DejaVu Sans"
    ]
    plt.rcParams["axes.unicode_minus"] = False


def load_wind_export() -> pd.DataFrame:
    raw = pd.read_csv(WIND_CSV, encoding="gbk").iloc[4:].copy()
    raw = raw[raw["指标名称"] != "数据来源：Wind"]
    raw["date"] = pd.to_datetime(raw["指标名称"], errors="coerce")
    raw = raw.dropna(subset=["date"]).sort_values("date")
    rename = {
        "中国:M2:同比": "m2_yoy",
        "中国:社会融资规模存量:同比": "sf_yoy",
        "市盈率:申万小盘指数": "申万小盘市盈率",
        "市盈率:申万大盘指数": "申万大盘市盈率",
    }
    out = raw.rename(columns=rename)
    for col in ["m2_yoy", "sf_yoy", *ASSETS]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out[["date", "m2_yoy", "sf_yoy", *ASSETS]]


def update_mobility_factor() -> pd.DataFrame:
    print("\n" + "=" * 65)
    print("【1/5】更新低频流动性因子 mobility_factor.csv")
    print("=" * 65)

    if USE_ONLINE:
        try:
            out = build_mobility_monthly()
            sf = load_sf_yoy_from_wind()
            print(
                f"联网: M2=东方财富, 社融存量同比=Wind "
                f"(截至 {sf.index.max().date()})"
            )
        except Exception as exc:
            print(f"联网更新低频失败，回退 Wind：{exc}")
            wind = load_wind_export()
            m2 = wind.dropna(subset=["m2_yoy"]).set_index("date")["m2_yoy"].resample("ME").last()
            sf = wind.dropna(subset=["sf_yoy"]).set_index("date")["sf_yoy"].resample("ME").last()
            merged = pd.concat([m2.rename("m2_yoy"), sf.rename("sf_yoy")], axis=1).dropna()
            merged["mobility_factor"] = merged["m2_yoy"] - merged["sf_yoy"]
            out = merged.reset_index(names="date")
    else:
        wind = load_wind_export()
        m2 = wind.dropna(subset=["m2_yoy"]).set_index("date")["m2_yoy"].resample("ME").last()
        sf = wind.dropna(subset=["sf_yoy"]).set_index("date")["sf_yoy"].resample("ME").last()
        merged = pd.concat([m2.rename("m2_yoy"), sf.rename("sf_yoy")], axis=1).dropna()
        merged["mobility_factor"] = merged["m2_yoy"] - merged["sf_yoy"]
        out = merged.reset_index(names="date")

    out.to_csv(MOBILITY_FACTOR_CSV, index=False, encoding="utf-8-sig", float_format="%.6f")
    print(
        f"已保存: {MOBILITY_FACTOR_CSV} ({len(out)} 月, "
        f"{out['date'].min().date()} ~ {out['date'].max().date()})"
    )
    return out


def update_high_freq_daily() -> pd.DataFrame:
    print("\n" + "=" * 65)
    print("【2/5】更新高频日频 mobility_high_freq_daily.csv")
    print("=" * 65)

    if not UPDATE_DATA and HIGH_FREQ_CSV.exists():
        daily = pd.read_csv(HIGH_FREQ_CSV, parse_dates=["date"]).set_index("date")
        if all(c in daily.columns for c in ASSETS):
            return daily[ASSETS].astype(float).reset_index()

    out: pd.DataFrame | None = None
    if USE_ONLINE:
        try:
            pe = fetch_pe_daily()
            out = pe.reset_index()
            print(
                f"联网 PE 代理: {PE_LARGE_SYMBOL}/{PE_SMALL_SYMBOL} → "
                f"{ASSETS[0]}/{ASSETS[1]}"
            )
        except Exception as exc:
            print(f"联网更新高频 PE 失败，回退 Wind：{exc}")

    if out is None:
        wind = load_wind_export()
        pe = wind.dropna(subset=ASSETS, how="all").set_index("date")[ASSETS].sort_index()
        daily = pe.ffill().dropna(how="any")
        out = daily.reset_index()

    out.to_csv(HIGH_FREQ_CSV, index=False, encoding="utf-8-sig", float_format="%.6f")
    print(
        f"已保存: {HIGH_FREQ_CSV} ({len(out)} 行, "
        f"{out['date'].min().date()} ~ {out['date'].max().date()})"
    )
    return out


def load_daily_prices() -> pd.DataFrame:
    daily = pd.read_csv(HIGH_FREQ_CSV, parse_dates=["date"])
    return daily.sort_values("date").set_index("date")[ASSETS].astype(float)


def load_monthly_log_yoy(daily: pd.DataFrame) -> pd.DataFrame:
    monthly_price = daily.resample("ME").last()
    log_yoy = np.log(monthly_price / monthly_price.shift(12)) * 100
    log_yoy.index = log_yoy.index.to_period("M")
    return log_yoy


def load_macro_y() -> pd.Series:
    macro = pd.read_csv(MOBILITY_FACTOR_CSV, parse_dates=["date"])
    return macro.set_index(macro["date"].dt.to_period("M"))["mobility_factor"].dropna()


def fit_ols(y: pd.Series, X: pd.DataFrame) -> dict | None:
    common = pd.concat([y, X], axis=1).dropna()
    if len(common) < 24:
        return None
    model = sm.OLS(common.iloc[:, 0].values, sm.add_constant(common.iloc[:, 1:].values)).fit()
    return {
        "model": model,
        "n": len(common),
        "r2": model.rsquared,
        "adj_r2": model.rsquared_adj,
        "aic": model.aic,
        "bic": model.bic,
    }


def search_joint_lags(y: pd.Series, X_monthly: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    records = []
    best = None
    best_lags = None
    for lags in itertools.product(LAG_RANGE, repeat=len(ASSETS)):
        X_lagged = pd.DataFrame(
            {col: X_monthly[col].shift(lag) for col, lag in zip(ASSETS, lags)},
            index=X_monthly.index,
        )
        res = fit_ols(y, X_lagged)
        if res is None:
            continue
        records.append({
            **{f"{c}_shift": lag for c, lag in zip(ASSETS, lags)},
            "R²": res["r2"],
            "Adj_R²": res["adj_r2"],
            "AIC": res["aic"],
            "BIC": res["bic"],
            "n": res["n"],
        })
        if best is None or res["aic"] < best["aic"]:
            best = res
            best_lags = lags
    if best is None:
        raise RuntimeError("回归样本不足")
    return {"result": best, "lags": dict(zip(ASSETS, best_lags))}, pd.DataFrame(records)


def build_result_table(model, lags: dict) -> pd.DataFrame:
    rows = []
    betas = []
    for i, col in enumerate(ASSETS):
        beta = model.params[i + 1]
        betas.append(beta)
        rows.append({
            "资产": col,
            "领先期(shift)": lags[col],
            "回归系数beta": beta,
        })
    df = pd.DataFrame(rows)
    abs_sum = np.abs(betas).sum()
    df["归一化权重"] = df["回归系数beta"] / abs_sum if abs_sum > 0 else 0
    df.loc[len(df)] = {
        "资产": "截距α",
        "领先期(shift)": "-",
        "回归系数beta": model.params[0],
        "归一化权重": np.nan,
    }
    return df


def normalized_weights(betas: dict[str, float]) -> dict[str, float]:
    abs_sum = sum(abs(v) for v in betas.values())
    return {k: v / abs_sum for k, v in betas.items()} if abs_sum > 0 else betas


def run_regression() -> tuple[dict[str, int], sm.regression.linear_model.RegressionResultsWrapper]:
    print("\n" + "=" * 65)
    print("【3/5】多变量滞后回归")
    print("=" * 65)

    daily = load_daily_prices()
    X_monthly = load_monthly_log_yoy(daily)
    y = load_macro_y()
    common_idx = X_monthly.index.intersection(y.index)
    X = X_monthly.loc[common_idx]
    y = y.loc[common_idx]
    print(f"样本区间: {common_idx.min()} ~ {common_idx.max()}, 共 {len(common_idx)} 个月")

    joint, _ = search_joint_lags(y, X)
    best_lags = joint["lags"]
    best_res = joint["result"]
    coef_joint = build_result_table(best_res["model"], best_lags)
    coef_joint.to_csv(ROOT / "regression_multivar_lags_coef.csv", index=False, encoding="utf-8-sig")

    meta = {
        "lags_months": best_lags,
        "weights": normalized_weights(
            {col: float(best_res["model"].params[i + 1]) for i, col in enumerate(ASSETS)}
        ),
        "betas": {col: float(best_res["model"].params[i + 1]) for i, col in enumerate(ASSETS)},
        "intercept": float(best_res["model"].params[0]),
        "r_squared": float(best_res["r2"]),
        "adj_r_squared": float(best_res["adj_r2"]),
        "bic": float(best_res["bic"]),
        "n_obs": int(best_res["n"]),
        "assets": ASSETS,
        "y_definition": "M2同比 - 社融存量同比",
    }
    (ROOT / "hf_regression_results.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"最优 R²={best_res['r2']:.4f}, Adj-R²={best_res['adj_r2']:.4f}, AIC={best_res['aic']:.2f}")
    print(coef_joint.to_string(index=False))
    return best_lags, best_res["model"]


def build_hf_mobility(
    lags: dict[str, int], model: sm.regression.linear_model.RegressionResultsWrapper
) -> pd.DataFrame:
    print("\n" + "=" * 65)
    print("【4/5】高频化宏观流动性因子拟合")
    print("=" * 65)

    weights = normalized_weights({col: model.params[i + 1] for i, col in enumerate(ASSETS)})
    for col in ASSETS:
        print(f"  {col}: shift={lags[col]:+d}, weight={weights[col]:+.4f}")

    daily = load_daily_prices()
    daily_ret = np.log(daily / daily.shift(1)) * 100
    parts = [weights[col] * daily_ret[col].shift(lags[col] * TRADING_DAYS_MONTH) for col in ASSETS]
    hf_mom = pd.concat(parts, axis=1).sum(axis=1)

    valid = hf_mom.dropna()
    hf_nav = NAV_BASE * (1.0 + valid / 100.0).cumprod()
    hf_yoy = np.log(hf_nav / hf_nav.shift(TRADING_DAYS_YEAR)) * 100

    out = pd.DataFrame(index=daily.index)
    out["hf_mom_pct"] = hf_mom
    out["hf_mobility_factor"] = hf_nav
    out["hf_yoy"] = hf_yoy
    out = out.reset_index().rename(columns={"index": "date"})
    out.to_csv(ROOT / "hf_mobility_factor_synthetic.csv", index=False, encoding="utf-8-sig", float_format="%.6f")

    macro = pd.read_csv(MOBILITY_FACTOR_CSV, parse_dates=["date"])
    macro_m = macro.set_index(macro["date"].dt.to_period("M"))["mobility_factor"].dropna()
    hf_yoy_m = hf_yoy.resample("ME").last().dropna()
    hf_yoy_m.index = hf_yoy_m.index.to_period("M")
    cmp_idx = hf_yoy_m.index.intersection(macro_m.index)
    corr = hf_yoy_m.loc[cmp_idx].corr(macro_m.loc[cmp_idx])
    pd.DataFrame({
        "ym": cmp_idx.astype(str),
        "mobility_factor": macro_m.loc[cmp_idx].values,
        "hf_yoy": hf_yoy_m.loc[cmp_idx].values,
    }).to_csv(ROOT / "hf_yoy_vs_macro.csv", index=False, encoding="utf-8-sig", float_format="%.6f")

    print(f"高频同比 vs 原始流动性因子（月频）相关系数: {corr:.4f}")
    print("已保存: hf_mobility_factor_synthetic.csv, hf_yoy_vs_macro.csv")
    return out


def plot_all() -> None:
    print("\n" + "=" * 65)
    print("【5/5】生成图表")
    print("=" * 65)
    setup_chinese_font()

    macro = pd.read_csv(MOBILITY_FACTOR_CSV, parse_dates=["date"]).dropna(subset=["mobility_factor"])
    macro_m = macro.set_index("date")["mobility_factor"]
    macro_m.index = macro_m.index.to_period("M")

    hf = pd.read_csv(ROOT / "hf_mobility_factor_synthetic.csv", parse_dates=["date"])
    hf = hf.dropna(subset=["hf_yoy"]).set_index("date")
    hf_m = hf["hf_yoy"].resample("ME").last()
    hf_m.index = hf_m.index.to_period("M")
    idx = hf_m.index.intersection(macro_m.index)
    corr = hf_m.loc[idx].corr(macro_m.loc[idx])
    hf_w = hf["hf_yoy"].resample("W-FRI").last().dropna()

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={"height_ratios": [2, 1]})
    fig.patch.set_facecolor("#fafafa")
    fig.suptitle("高频流动性因子同比 vs M2-社融同比", fontsize=15, fontweight="bold", y=1.01)
    axes[0].plot(idx.to_timestamp(), macro_m.loc[idx], color="#94a3b8", lw=1.5, alpha=0.85, label="低频流动性因子（月频）")
    axes[0].plot(hf_w.index, hf_w.values, color="#0ea5e9", lw=1.5, label="高频化因子同比（周频采样）")
    axes[0].set_title(f"月频对比（相关系数 r = {corr:.3f}）")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(hf.index, hf["hf_yoy"], color="#0284c7", lw=0.8, alpha=0.9)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlabel("日期")
    plt.tight_layout()
    plt.savefig(ROOT / "hf_yoy_vs_macro.png", dpi=150, bbox_inches="tight", facecolor="#fafafa")
    plt.close()
    print("已保存: hf_yoy_vs_macro.png")


def main() -> None:
    setup_chinese_font()
    print("宏观流动性（mobility）因子全流程")
    lags = model = None

    if UPDATE_DATA:
        update_mobility_factor()
    update_high_freq_daily()

    if RUN_REGRESSION:
        lags, model = run_regression()

    if BUILD_HF:
        if lags is None or model is None:
            coef = pd.read_csv(ROOT / "regression_multivar_lags_coef.csv")
            lags = {row["资产"]: int(row["领先期(shift)"]) for _, row in coef.iterrows() if row["资产"] != "截距α"}
            daily = load_daily_prices()
            X = load_monthly_log_yoy(daily)
            y = load_macro_y()
            idx = X.index.intersection(y.index)
            X_lagged = pd.DataFrame({c: X.loc[idx, c].shift(lags[c]) for c in ASSETS})
            common = pd.concat([y.loc[idx], X_lagged], axis=1).dropna()
            model = sm.OLS(common.iloc[:, 0], sm.add_constant(common.iloc[:, 1:])).fit()
        build_hf_mobility(lags, model)

    if MAKE_PLOTS:
        plot_all()

    print("\n全流程完成。")


if __name__ == "__main__":
    main()
