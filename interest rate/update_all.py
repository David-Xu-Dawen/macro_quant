#!/usr/bin/env python3
"""
利率因子项目 — 一键更新全部数据。

宏观因子：十年国债收益率绝对水平（月度，%）
高频因子：中债国债总净价指数相反数（日频水平）；环比为 log 环比相反数

运行: python update_all.py
"""

from __future__ import annotations

import itertools
import json
import warnings
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from fetch_cn10y import fetch_cn10y_daily
from fetch_cn_gov_bond_index import fetch_gov_bond_index_daily

warnings.filterwarnings("ignore")

OUTPUT_DIR = Path(__file__).parent
YIELD_FILE = OUTPUT_DIR / "cn10y_yield_daily.csv"
INDEX_FILE = OUTPUT_DIR / "cn_gov_bond_index_daily.csv"
RATE_FACTOR_FILE = OUTPUT_DIR / "rate_factor.csv"

UPDATE_DATA = True

ASSETS = ["bond_index_neg"]
ASSET_LABELS = {"bond_index_neg": "国债总净价指数（相反数）"}

LAG_RANGE = range(0, 4)  # 0~3 月，X.shift(lag) 仅使用当期及历史数据
TRADING_DAYS_MONTH = 21
PLOT_START_DATE = "2016-10-01"


def setup_chinese_font() -> None:
    import matplotlib.font_manager as fm

    for font in ["Songti SC", "STHeiti", "Kaiti SC", "PingFang HK", "SimHei"]:
        if font in {f.name for f in fm.fontManager.ttflist}:
            plt.rcParams["font.sans-serif"] = [font, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return


def update_market_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    if UPDATE_DATA:
        fetch_cn10y_daily().to_csv(YIELD_FILE, index=False, encoding="utf-8-sig")
        fetch_gov_bond_index_daily().to_csv(INDEX_FILE, index=False, encoding="utf-8-sig")

    yield_df = pd.read_csv(YIELD_FILE, parse_dates=["日期"]).sort_values("日期")
    index_df = pd.read_csv(INDEX_FILE, parse_dates=["日期"]).sort_values("日期")
    print(f"  十年国债: {len(yield_df)} 日, {yield_df['日期'].min().date()} ~ {yield_df['日期'].max().date()}")
    print(f"  国债净价指数: {len(index_df)} 日, {index_df['日期'].min().date()} ~ {index_df['日期'].max().date()}")
    return yield_df, index_df


def monthly_last_asof(series: pd.Series) -> pd.Series:
    """按月取最后一个有效观测，日期戳用真实交易日（避免未完结月写成月末）。"""
    s = series.dropna().sort_index()
    if s.empty:
        return s
    df = s.rename("v").to_frame()
    df["ym"] = df.index.to_period("M")
    return df.groupby("ym", sort=True).tail(1)["v"]


def update_rate_factor(yield_df: pd.DataFrame) -> pd.DataFrame:
    """宏观因子：十年国债收益率月度绝对水平（%）。"""
    setup_chinese_font()

    daily = yield_df.set_index("日期")["yield_10y"].where(lambda s: s > 0)
    monthly_yield = monthly_last_asof(daily)
    rate_factor = monthly_yield

    out = pd.DataFrame({
        "date": monthly_yield.index,
        "yield_10y": monthly_yield.values,
        "rate_factor": rate_factor.values,
    }).dropna(subset=["rate_factor"])

    out.to_csv(RATE_FACTOR_FILE, index=False, float_format="%.6f")

    fig, ax = plt.subplots(figsize=(14, 5), dpi=120)
    ax.plot(out["date"], out["rate_factor"], color="#1f4e79", linewidth=1.2, label="十年国债利率因子（绝对水平）")
    ax.set_title("宏观利率因子：十年国债收益率（月度，%）", fontsize=14, pad=12)
    ax.set_xlabel("日期")
    ax.set_ylabel("收益率 (%)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "cn10y_rate_factor.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  rate_factor.csv: {len(out)} 月, 最新 {out['rate_factor'].iloc[-1]:.4f}")
    return out


def load_index_daily(index_df: pd.DataFrame) -> pd.Series:
    return index_df.set_index("日期")["index_net"].astype(float)


def load_macro_y() -> pd.Series:
    macro = pd.read_csv(RATE_FACTOR_FILE, parse_dates=["date"])
    return macro.set_index(macro["date"].dt.to_period("M"))["rate_factor"].dropna()


def monthly_bond_proxy(index: pd.Series) -> pd.DataFrame:
    """国债净价指数对数水平的相反数（月频），作为收益率水平代理。"""
    monthly = index.resample("ME").last()
    log_level = -np.log(monthly)
    log_level.index = log_level.index.to_period("M")
    return log_level.to_frame("bond_index_neg")


def fit_ols(y: pd.Series, X: pd.DataFrame) -> dict | None:
    common = pd.concat([y, X], axis=1).dropna()
    if len(common) < 24:
        return None
    model = sm.OLS(common.iloc[:, 0], sm.add_constant(common.iloc[:, 1:])).fit()
    return {"model": model, "bic": model.bic, "n": len(common)}


def search_joint_lags(y: pd.Series, X: pd.DataFrame) -> tuple[dict, dict]:
    best, best_lags = None, None
    for lags in itertools.product(LAG_RANGE, repeat=len(ASSETS)):
        X_lag = pd.DataFrame({c: X[c].shift(l) for c, l in zip(ASSETS, lags)}, index=X.index)
        res = fit_ols(y, X_lag)
        if res and (best is None or res["bic"] < best["bic"]):
            best, best_lags = res, dict(zip(ASSETS, lags))
    if best is None:
        raise RuntimeError("回归样本不足")
    return best, best_lags


def normalized_weights(betas: dict[str, float]) -> dict[str, float]:
    s = sum(abs(v) for v in betas.values())
    return {k: v / s for k, v in betas.items()} if s else betas


def macro_yield_daily(yield_df: pd.DataFrame) -> pd.Series:
    """日频十年国债收益率，用于与高频因子对比。"""
    return yield_df.set_index("日期")["yield_10y"].where(lambda s: s > 0)


def update_hf_rate_factor(index_df: pd.DataFrame, yield_df: pd.DataFrame) -> None:
    setup_chinese_font()

    index = load_index_daily(index_df)
    X_m = monthly_bond_proxy(index)
    y = load_macro_y()
    idx = X_m.index.intersection(y.index)
    best, lags = search_joint_lags(y.loc[idx], X_m.loc[idx])
    model = best["model"]
    betas = {c: float(model.params[c]) for c in ASSETS}
    weights = normalized_weights(betas)
    meta = {
        "lags_months": lags,
        "weights": weights,
        "betas": betas,
        "intercept": float(model.params["const"]),
        "r_squared": float(model.rsquared),
        "adj_r_squared": float(model.rsquared_adj),
        "bic": float(model.bic),
        "n_obs": int(best["n"]),
        "assets": ASSETS,
        "hf_proxy": "neg_bond_net_index",
    }
    with open(OUTPUT_DIR / "hf_regression_results.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # 月频拟合
    X_lag = pd.DataFrame({c: X_m.loc[idx, c].shift(lags[c]) for c in ASSETS}, index=idx)
    common = pd.concat([y.loc[idx].rename("rate_factor"), X_lag], axis=1).dropna()
    fitted = np.asarray(model.predict(sm.add_constant(common[ASSETS])))
    monthly_fitted = pd.DataFrame({
        "date": [p.to_timestamp("M") for p in common.index],
        "rate_factor": common["rate_factor"].values,
        "macro_fitted": fitted,
        "neg_log_bond_index": common["bond_index_neg"].values,
        "residual": common["rate_factor"].values - fitted,
    })
    monthly_fitted.to_csv(OUTPUT_DIR / "hf_monthly_fitted.csv", index=False, float_format="%.6f")

    # 日频：国债总净价指数相反数（水平）；环比为 log 环比相反数
    yield_daily = macro_yield_daily(yield_df).reindex(index.index, method="ffill")
    index_neg = -index
    neg_log_mom = -np.log(index / index.shift(1)) * 100
    neg_log_level = -np.log(index)
    hf_fitted = float(model.params["const"]) + float(model.params["bond_index_neg"]) * neg_log_level.shift(
        lags[ASSETS[0]] * TRADING_DAYS_MONTH
    )

    out = pd.DataFrame(index=index.index)
    out["index_net"] = index
    out["index_neg"] = index_neg
    out["neg_log_mom_pct"] = neg_log_mom
    out["hf_level"] = index_neg
    out["hf_fitted"] = hf_fitted
    out["hf_mom_pct"] = neg_log_mom
    out["yield_10y"] = yield_daily
    out = out.reset_index().rename(columns={"index": "日期"})
    out.to_csv(OUTPUT_DIR / "hf_rate_factor_daily.csv", index=False, float_format="%.6f")

    # 对比图：净价指数相反数 vs 十年国债收益率（日频）
    macro_d = yield_daily.dropna()
    hf_plot = index_neg.dropna()
    macro_on_d = macro_d.reindex(hf_plot.index, method="ffill")
    compare = pd.DataFrame({
        "date": hf_plot.index,
        "index_neg": hf_plot.values,
        "rate_factor_level": macro_on_d.values,
    })
    compare["hf_fitted"] = hf_fitted.reindex(hf_plot.index).values
    plot_start = pd.Timestamp(PLOT_START_DATE)
    compare = compare[compare["date"] >= plot_start]
    macro_plot = macro_d[macro_d.index >= plot_start].dropna()
    hf_plot = hf_plot[hf_plot.index >= plot_start]

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    fig.suptitle("高频利率因子 vs 十年国债宏观利率因子（绝对水平）", fontsize=15, fontweight="bold")
    axes[0].plot(macro_plot.index, macro_plot.values, label="宏观：十年国债收益率", color="#2C3E50", linewidth=2)
    axes[0].set_ylabel("收益率 (%)")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title("日频水平对比")
    ax1b = axes[0].twinx()
    ax1b.plot(hf_plot.index, hf_plot.values, label="净价指数相反数", color="#27AE60", linewidth=0.8, alpha=0.7)
    ax1b.set_ylabel("净价指数相反数")
    lines0, labels0 = axes[0].get_legend_handles_labels()
    lines1, labels1 = ax1b.get_legend_handles_labels()
    axes[0].legend(lines0 + lines1, labels0 + labels1, loc="upper right")
    axes[1].plot(compare["date"], compare["hf_fitted"] - compare["rate_factor_level"],
                   color="#3498DB", linewidth=0.9, label="拟合收益率 - 实际收益率")
    axes[1].axhline(0, color="gray", linestyle="--", alpha=0.6)
    axes[1].set_ylabel("偏离 (百分点)")
    axes[1].set_xlabel("日期")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "rate_factor_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    compare.to_csv(OUTPUT_DIR / "hf_yoy_vs_macro.csv", index=False, float_format="%.6f")

    corr = pd.Series(compare["hf_fitted"]).corr(pd.Series(compare["rate_factor_level"]))
    print(f"  hf 回归 R²={meta['r_squared']:.4f}, 拟合收益率与实际收益率相关系数={corr:.4f}")
    print(f"  hf_level 最新={index_neg.dropna().iloc[-1]:.4f}（净价指数相反数）")
    for c in ASSETS:
        print(f"    {ASSET_LABELS[c]}: 滞后{lags[c]}月, 权重={weights[c]:+.4f}")


def main() -> None:
    print("=" * 60)
    print("开始更新利率因子数据")
    print("=" * 60)

    print("\n[1/3] 市场数据")
    yield_df, index_df = update_market_data()

    print("\n[2/3] 宏观利率因子 → rate_factor.csv")
    update_rate_factor(yield_df)

    print("\n[3/3] 高频利率因子（国债总净价指数相反数）")
    update_hf_rate_factor(index_df, yield_df)

    print("\n" + "=" * 60)
    print("全部完成。保留文件：")
    for name in [
        "cn10y_yield_daily.csv",
        "cn_gov_bond_index_daily.csv",
        "rate_factor.csv",
        "cn10y_rate_factor.png",
        "hf_regression_results.json",
        "hf_monthly_fitted.csv",
        "hf_rate_factor_daily.csv",
        "hf_yoy_vs_macro.csv",
        "rate_factor_comparison.png",
    ]:
        print(f"  · {name}")


if __name__ == "__main__":
    main()
