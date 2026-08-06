#!/usr/bin/env python3
"""
地缘（GPR）因子：用黄金 + 布伦特原油绝对价格线性拟合。

流程：
1. 低频目标：GPR 月均绝对水平
2. 高频资产：沪金、布伦特原油日频收盘价（绝对水平）
3. 月频多元滞后回归：GPR ~ 黄金/原油月末绝对价格
4. 高频化：日频绝对价格按回归系数线性合成
5. 输出原始 GPR vs 拟合因子对比图
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

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
PROJECT_ROOT = ROOT.parent
GPR_CSV = ROOT / "data" / "gpr_daily.csv"
GOLD_CSV = PROJECT_ROOT / "factor exposure" / "data" / "raw" / "沪金.csv"
OIL_CSV = PROJECT_ROOT / "factor exposure" / "data" / "raw" / "布伦特原油.csv"

ASSETS = ["沪金", "布伦特原油"]
ASSET_COLORS = {
    "沪金": "#d97706",
    "布伦特原油": "#0f766e",
}

LAG_RANGE = range(0, 4)
TRADING_DAYS_MONTH = 21
WEEK_FREQ = "W-FRI"

HIGH_FREQ_CSV = ROOT / "geo_high_freq_daily.csv"
HF_SYNTHETIC_CSV = ROOT / "hf_geo_factor_synthetic.csv"
MONTHLY_FIT_CSV = ROOT / "geo_fit_monthly.csv"
COMPARE_PNG = ROOT / "geo_compare_chart.png"


def setup_chinese_font() -> None:
    plt.rcParams["font.sans-serif"] = [
        "PingFang SC",
        "Heiti SC",
        "STHeiti",
        "SimHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def load_gpr_daily() -> pd.Series:
    gpr = pd.read_csv(GPR_CSV, parse_dates=["date"])
    return gpr.sort_values("date").set_index("date")["GPRD"].astype(float)


def load_asset_daily(path: Path, col_name: str) -> pd.Series:
    df = pd.read_csv(path, parse_dates=["date"])
    return df.sort_values("date").set_index("date")["close"].astype(float).rename(col_name)


def update_high_freq_daily() -> pd.DataFrame:
    print("\n" + "=" * 65)
    print("【1/4】更新高频日频 geo_high_freq_daily.csv（沪金 + 布伦特原油）")
    print("=" * 65)

    gold = load_asset_daily(GOLD_CSV, "沪金")
    oil = load_asset_daily(OIL_CSV, "布伦特原油")
    daily = pd.concat([gold, oil], axis=1).sort_index().ffill().dropna(how="any")
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


def load_monthly_levels(daily: pd.DataFrame) -> pd.DataFrame:
    monthly_price = daily.resample("ME").last()
    monthly_price.index = monthly_price.index.to_period("M")
    return monthly_price


def load_macro_y() -> pd.Series:
    gpr = load_gpr_daily()
    monthly = gpr.resample("ME").mean()
    monthly.index = monthly.index.to_period("M")
    return monthly.dropna()


def apply_shift(s: pd.Series, shift: int) -> pd.Series:
    return s.shift(shift)


def fit_ols(y: pd.Series, X: pd.DataFrame) -> dict | None:
    common = pd.concat([y, X], axis=1).dropna()
    if len(common) < 24:
        return None
    model = sm.OLS(common.iloc[:, 0].values, sm.add_constant(common.iloc[:, 1:].values)).fit()
    fitted = pd.Series(
        model.predict(sm.add_constant(common.iloc[:, 1:].values)),
        index=common.index,
    )
    return {
        "model": model,
        "n": len(common),
        "r2": model.rsquared,
        "adj_r2": model.rsquared_adj,
        "aic": model.aic,
        "bic": model.bic,
        "fitted": fitted,
    }


def search_joint_lags(y: pd.Series, X_monthly: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    records = []
    best = None
    best_lags = None
    for lags in itertools.product(LAG_RANGE, repeat=len(ASSETS)):
        X_lagged = pd.DataFrame(
            {col: apply_shift(X_monthly[col], lag) for col, lag in zip(ASSETS, lags)},
            index=X_monthly.index,
        )
        res = fit_ols(y, X_lagged)
        if res is None:
            continue
        records.append(
            {
                **{f"{c}_shift": lag for c, lag in zip(ASSETS, lags)},
                "R²": res["r2"],
                "Adj_R²": res["adj_r2"],
                "AIC": res["aic"],
                "BIC": res["bic"],
                "n": res["n"],
            }
        )
        if best is None or res["aic"] < best["aic"]:
            best = res
            best_lags = lags
    if best is None or best_lags is None:
        raise RuntimeError("回归样本不足，无法拟合地缘因子")
    return {"result": best, "lags": dict(zip(ASSETS, best_lags))}, pd.DataFrame(records)


def build_result_table(model, lags: dict) -> pd.DataFrame:
    rows = []
    betas = []
    for i, col in enumerate(ASSETS):
        beta = model.params[i + 1]
        betas.append(beta)
        rows.append(
            {
                "资产": col,
                "领先期(shift)": lags[col],
                "回归系数beta": beta,
            }
        )
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


def run_regression() -> tuple[dict[str, int], sm.regression.linear_model.RegressionResultsWrapper, pd.Series]:
    print("\n" + "=" * 65)
    print("【2/4】月频多元滞后回归：GPR ~ 沪金/原油绝对价格")
    print("=" * 65)

    daily = load_daily_prices()
    X_monthly = load_monthly_levels(daily)
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
        "y_definition": "GPR 月均绝对水平",
        "x_definition": "沪金、布伦特原油月末绝对价格",
    }
    (ROOT / "hf_regression_results.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    monthly_fit = pd.DataFrame(
        {
            "ym": best_res["fitted"].index.astype(str),
            "gpr_actual": y.loc[best_res["fitted"].index].values,
            "gpr_fitted": best_res["fitted"].values,
        }
    )
    monthly_fit.to_csv(MONTHLY_FIT_CSV, index=False, encoding="utf-8-sig", float_format="%.6f")

    print(f"最优 R²={best_res['r2']:.4f}, Adj-R²={best_res['adj_r2']:.4f}, AIC={best_res['aic']:.2f}")
    print(coef_joint.to_string(index=False))
    return best_lags, best_res["model"], best_res["fitted"]


def build_hf_geo(
    lags: dict[str, int], model: sm.regression.linear_model.RegressionResultsWrapper
) -> pd.DataFrame:
    print("\n" + "=" * 65)
    print("【3/4】高频化地缘因子拟合（绝对价格线性合成）")
    print("=" * 65)

    daily = load_daily_prices()
    intercept = float(model.params[0])
    betas = {col: float(model.params[i + 1]) for i, col in enumerate(ASSETS)}

    level_parts = []
    for col in ASSETS:
        lag_days = int(lags[col]) * TRADING_DAYS_MONTH
        level_parts.append(betas[col] * daily[col].shift(lag_days))
    hf_level = intercept + pd.concat(level_parts, axis=1).sum(axis=1, min_count=1)

    out = pd.DataFrame(
        {
            "date": daily.index,
            "hf_geo_factor": hf_level.values,
        }
    )
    out.to_csv(HF_SYNTHETIC_CSV, index=False, encoding="utf-8-sig", float_format="%.6f")

    gpr_daily = load_gpr_daily()
    gpr_weekly = gpr_daily.resample(WEEK_FREQ).mean()
    hf_weekly = hf_level.resample(WEEK_FREQ).last()
    cmp_idx = gpr_weekly.dropna().index.intersection(hf_weekly.dropna().index)
    corr = gpr_weekly.loc[cmp_idx].corr(hf_weekly.loc[cmp_idx])
    print(f"高频拟合水平 vs 原始 GPR 周均相关系数: {corr:.4f}")
    print(f"已保存: {HF_SYNTHETIC_CSV}")
    return out


def make_compare_chart(monthly_fitted: pd.Series) -> None:
    print("\n" + "=" * 65)
    print("【4/4】生成对比图")
    print("=" * 65)

    setup_chinese_font()
    gpr_daily = load_gpr_daily()
    gpr_monthly = gpr_daily.resample("ME").mean()
    gpr_weekly = gpr_daily.resample(WEEK_FREQ).mean()

    hf = pd.read_csv(HF_SYNTHETIC_CSV, parse_dates=["date"]).set_index("date")
    hf_weekly = hf["hf_geo_factor"].resample(WEEK_FREQ).last().dropna()

    monthly_fit = monthly_fitted.copy()
    monthly_fit.index = monthly_fit.index.to_timestamp()

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), dpi=130)

    ax0 = axes[0]
    ax0.plot(
        gpr_monthly.index,
        gpr_monthly.values,
        color="#2C3E50",
        linewidth=2.0,
        label="原始 GPR（月均）",
    )
    ax0.plot(
        monthly_fit.index,
        monthly_fit.values,
        color="#E74C3C",
        linewidth=1.8,
        linestyle="--",
        label="黄金+原油绝对价格线性拟合（月频）",
    )
    ax0.set_title("低频：GPR 原始值 vs 黄金/原油绝对价格拟合值", fontsize=13, pad=10)
    ax0.set_ylabel("GPR 指数")
    ax0.grid(True, linestyle="--", alpha=0.35)
    ax0.legend(loc="upper left", framealpha=0.9)

    ax1 = axes[1]
    ax1.plot(
        gpr_weekly.index,
        gpr_weekly.values,
        color="#2C3E50",
        linewidth=1.5,
        alpha=0.9,
        label="原始 GPR（周均）",
    )
    ax1.plot(
        hf_weekly.index,
        hf_weekly.values,
        color="#E74C3C",
        linewidth=1.3,
        alpha=0.9,
        label="高频拟合地缘因子（绝对价格合成）",
    )
    ax1.set_title("高频：GPR 周均 vs 黄金/原油绝对价格合成因子", fontsize=13, pad=10)
    ax1.set_xlabel("日期")
    ax1.set_ylabel("GPR 指数")
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax1.legend(loc="upper left", framealpha=0.9)

    fig.tight_layout()
    fig.savefig(COMPARE_PNG, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"已保存: {COMPARE_PNG}")


def main() -> None:
    print("地缘因子（黄金 + 布伦特原油绝对价格）拟合全流程")
    update_high_freq_daily()
    lags, model, monthly_fitted = run_regression()
    build_hf_geo(lags, model)
    make_compare_chart(monthly_fitted)
    print("\n完成。")


if __name__ == "__main__":
    main()
