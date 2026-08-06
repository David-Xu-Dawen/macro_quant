#!/usr/bin/env python3
"""
宏观增长因子全流程（单文件）：
1. 更新低频增长因子 growth_factor.csv
2. 更新高频日频 growth_high_freq_daily.csv（恒生指数 + CAD + 申万房地产）
3. 多变量滞后回归（仅同期/滞后，不使用未来数据）
4. 高频化增长因子拟合
5. 导出图表
"""

from __future__ import annotations

import itertools
import re
import warnings
from pathlib import Path

import akshare as ak
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.filters.hp_filter import hpfilter

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent

# ── 配置 ─────────────────────────────────────────────────────────────
UPDATE_DATA = True
RUN_REGRESSION = True
BUILD_HF = True
MAKE_PLOTS = True

ASSETS = [
    "恒生指数",
    "CAD",
    "申万房地产",
]
ASSET_COLORS = {
    "恒生指数": "#1e40af",
    "CAD": "#dc2626",
    "申万房地产": "#059669",
}

LAG_RANGE = range(0, 4)  # 0~3 月，X.shift(lag) 仅使用当期及历史数据
TRADING_DAYS_YEAR = 252
TRADING_DAYS_MONTH = 21
NAV_BASE = 100.0
HP_LAMBDA_MONTHLY = 129600
HP_LAMBDA_DAILY = HP_LAMBDA_MONTHLY / (21**4)

RAW_GROWTH_WEIGHTS = {
    "pmi_yoy_diff_filled": 0.579829835,
    "fai_yoy_filled": 0.069358574,
    "retail_yoy_filled": 0.246186048,
    "trade_yoy_weighted_filled": 0.104625543,
}

GROWTH_FACTOR_CSV = ROOT / "growth_factor.csv"
HIGH_FREQ_CSV = ROOT / "growth_high_freq_daily.csv"
NANHUA_COPPER_CSV = ROOT / "南华沪铜指数.csv"
NH_COPPER_ASSET = "南华沪铜"


def setup_chinese_font() -> None:
    plt.rcParams["font.sans-serif"] = [
        "PingFang SC", "Heiti SC", "STHeiti", "SimHei", "Arial Unicode MS", "DejaVu Sans"
    ]
    plt.rcParams["axes.unicode_minus"] = False


def parse_chinese_month(s: str) -> pd.Timestamp:
    m = re.match(r"(\d{4})年(\d{1,2})月份", str(s).strip())
    if not m:
        raise ValueError(f"无法解析月份: {s}")
    return pd.Timestamp(int(m.group(1)), int(m.group(2)), 1)


def fill_jan_feb(series: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """1月/2月缺失时：1月=(上年12月+当年3月)/2，2月=(当年1月填充值+3月)/2。"""
    out = series.copy()
    for year in sorted(index.year.unique()):
        dec = pd.Timestamp(year - 1, 12, 1)
        jan = pd.Timestamp(year, 1, 1)
        feb = pd.Timestamp(year, 2, 1)
        mar = pd.Timestamp(year, 3, 1)
        if mar not in index or pd.isna(series.get(mar)):
            continue
        v_mar = series.loc[mar]
        if jan in index and pd.isna(out.loc[jan]):
            v_dec = out.loc[dec] if dec in index and pd.notna(out.loc[dec]) else np.nan
            if pd.notna(v_dec):
                out.loc[jan] = (v_dec + v_mar) / 2
        if feb in index and pd.isna(out.loc[feb]) and jan in index and pd.notna(out.loc[jan]):
            out.loc[feb] = (out.loc[jan] + v_mar) / 2
    return out


def update_growth_factor() -> pd.DataFrame:
    print("\n" + "=" * 65)
    print("【1/5】更新低频增长因子 growth_factor.csv")
    print("=" * 65)

    pmi = ak.macro_china_pmi()
    pmi["date"] = pmi["月份"].map(parse_chinese_month)
    pmi = pmi.sort_values("date").set_index("date")
    pmi_yoy_diff = pd.to_numeric(pmi["制造业-同比增长"], errors="coerce")

    fai = ak.macro_china_gdzctz()
    fai["date"] = fai["月份"].map(parse_chinese_month)
    fai = fai.sort_values("date").set_index("date")
    fai_yoy = pd.to_numeric(fai["同比增长"], errors="coerce")

    retail = ak.macro_china_consumer_goods_retail()
    retail["date"] = retail["月份"].map(parse_chinese_month)
    retail = retail.sort_values("date").set_index("date")
    retail_yoy = pd.to_numeric(retail["同比增长"], errors="coerce")

    trade = ak.macro_china_hgjck()
    trade["date"] = trade["月份"].map(parse_chinese_month)
    trade = trade.sort_values("date").set_index("date")
    exp_yoy = pd.to_numeric(trade["当月出口额-同比增长"], errors="coerce")
    imp_yoy = pd.to_numeric(trade["当月进口额-同比增长"], errors="coerce")
    exp_amt = pd.to_numeric(trade["当月出口额-金额"], errors="coerce")
    imp_amt = pd.to_numeric(trade["当月进口额-金额"], errors="coerce")
    w_exp = exp_amt / (exp_amt + imp_amt)
    trade_yoy_weighted = w_exp * exp_yoy + (1 - w_exp) * imp_yoy

    idx = (
        pmi_yoy_diff.dropna().index
        .union(fai_yoy.dropna().index)
        .union(retail_yoy.dropna().index)
        .union(trade_yoy_weighted.dropna().index)
    )
    idx = pd.DatetimeIndex(sorted(idx))

    df = pd.DataFrame(index=idx)
    df["pmi_yoy_diff"] = pmi_yoy_diff.reindex(idx)
    df["fai_yoy"] = fai_yoy.reindex(idx)
    df["retail_yoy"] = retail_yoy.reindex(idx)
    df["trade_yoy_weighted"] = trade_yoy_weighted.reindex(idx)

    df["pmi_yoy_diff_filled"] = df["pmi_yoy_diff"]
    df["trade_yoy_weighted_filled"] = df["trade_yoy_weighted"]
    df["fai_yoy_filled"] = fill_jan_feb(df["fai_yoy"], idx)
    df["retail_yoy_filled"] = fill_jan_feb(df["retail_yoy"], idx)

    df["raw_growth_factor"] = sum(
        df[col] * w for col, w in RAW_GROWTH_WEIGHTS.items()
    )
    valid_raw = df["raw_growth_factor"].dropna()
    _, hp_trend = hpfilter(valid_raw.values, lamb=HP_LAMBDA_MONTHLY)
    df["growth_factor_hp"] = pd.Series(hp_trend, index=valid_raw.index)

    out = df.reset_index().rename(columns={"index": "date"})
    out.to_csv(GROWTH_FACTOR_CSV, index=False, encoding="utf-8-sig", float_format="%.6f")
    print(f"已保存: {GROWTH_FACTOR_CSV} ({len(out)} 行, {out['date'].min().date()} ~ {out['date'].max().date()})")
    return out


def load_nanhua_copper_daily() -> pd.Series:
    """读取本地 Wind 导出的南华沪铜指数日频数据。"""
    raw = pd.read_csv(NANHUA_COPPER_CSV, header=None, encoding="gbk")
    data = raw.iloc[5:].copy()
    data.columns = ["date", NH_COPPER_ASSET]
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data[NH_COPPER_ASSET] = pd.to_numeric(data[NH_COPPER_ASSET], errors="coerce")
    series = data.dropna().set_index("date")[NH_COPPER_ASSET].astype(float).sort_index()
    if series.empty:
        raise ValueError(f"未能从 {NANHUA_COPPER_CSV} 解析南华沪铜指数")
    return series


def load_other_hf_assets() -> pd.DataFrame:
    """恒生指数、CAD、申万房地产（801180）。"""
    if not UPDATE_DATA and HIGH_FREQ_CSV.exists():
        daily = pd.read_csv(HIGH_FREQ_CSV, parse_dates=["date"]).set_index("date")
        rename = {"申万801180": "申万房地产", "房地产": "申万房地产"}
        daily = daily.rename(columns=rename)
        cols = [c for c in ASSETS if c in daily.columns]
        if len(cols) == len(ASSETS):
            return daily[cols].astype(float)

    hs = ak.stock_hk_index_daily_sina(symbol="HSI")
    hs["date"] = pd.to_datetime(hs["date"])
    hs = hs.set_index("date")["close"].astype(float).rename("恒生指数")

    cad = ak.futures_foreign_hist(symbol="CAD")
    cad["date"] = pd.to_datetime(cad["date"])
    cad = cad.set_index("date")["close"].astype(float).rename("CAD")

    sw = ak.index_hist_sw(symbol="801180", period="day")
    sw["date"] = pd.to_datetime(sw["日期"])
    sw = sw.set_index("date")["收盘"].astype(float).rename("申万房地产")

    return pd.concat([hs, cad, sw], axis=1).sort_index()


def update_high_freq_daily() -> pd.DataFrame:
    print("\n" + "=" * 65)
    print("【2/5】更新高频日频 growth_high_freq_daily.csv（恒生指数 + CAD + 申万房地产）")
    print("=" * 65)

    other = load_other_hf_assets()
    daily = other[ASSETS].sort_index().ffill().dropna(how="any")
    out = daily.reset_index()
    out.to_csv(HIGH_FREQ_CSV, index=False, encoding="utf-8-sig", float_format="%.6f")
    print(
        f"已保存: {HIGH_FREQ_CSV} ({len(out)} 行, {out['date'].min().date()} ~ {out['date'].max().date()})"
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
    macro = pd.read_csv(GROWTH_FACTOR_CSV, parse_dates=["date"])
    y = macro.set_index(macro["date"].dt.to_period("M"))["raw_growth_factor"]
    return y.dropna()


def apply_shift(s: pd.Series, shift: int) -> pd.Series:
    return s.shift(shift)


def lag_label(shift: int) -> str:
    if shift > 0:
        return f"滞后{shift}期"
    if shift < 0:
        raise ValueError(f"不允许负滞后（会使用未来数据）: {shift}")
    return "同期"


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
            {col: apply_shift(X_monthly[col], lag) for col, lag in zip(ASSETS, lags)},
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
            "领先/滞后说明": lag_label(lags[col]),
            "回归系数beta": beta,
        })
    df = pd.DataFrame(rows)
    abs_sum = np.abs(betas).sum()
    df["归一化权重"] = df["回归系数beta"] / abs_sum if abs_sum > 0 else 0
    df.loc[len(df)] = {
        "资产": "截距α",
        "领先期(shift)": "-",
        "领先/滞后说明": "-",
        "回归系数beta": model.params[0],
        "归一化权重": np.nan,
    }
    return df


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

    print(f"最优 R²={best_res['r2']:.4f}, Adj-R²={best_res['adj_r2']:.4f}, AIC={best_res['aic']:.2f}")
    print(coef_joint.to_string(index=False))
    return best_lags, best_res["model"]


def normalized_weights(betas: dict[str, float]) -> dict[str, float]:
    abs_sum = sum(abs(v) for v in betas.values())
    return {k: v / abs_sum for k, v in betas.items()} if abs_sum > 0 else betas


def build_hf_growth(lags: dict[str, int], model: sm.regression.linear_model.RegressionResultsWrapper) -> pd.DataFrame:
    print("\n" + "=" * 65)
    print("【4/5】高频化宏观增长因子拟合")
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
    out["hf_growth_factor"] = hf_nav
    out["hf_yoy"] = hf_yoy
    out = out.reset_index().rename(columns={"index": "date"})
    out.to_csv(ROOT / "hf_growth_factor_synthetic.csv", index=False, encoding="utf-8-sig", float_format="%.6f")

    macro = pd.read_csv(GROWTH_FACTOR_CSV, parse_dates=["date"])
    macro_m = macro.set_index(macro["date"].dt.to_period("M"))["raw_growth_factor"].dropna()
    hf_yoy_m = hf_yoy.resample("ME").last().dropna()
    hf_yoy_m.index = hf_yoy_m.index.to_period("M")
    cmp_idx = hf_yoy_m.index.intersection(macro_m.index)
    corr = hf_yoy_m.loc[cmp_idx].corr(macro_m.loc[cmp_idx])

    pd.DataFrame({
        "ym": cmp_idx.astype(str),
        "raw_growth_factor": macro_m.loc[cmp_idx].values,
        "hf_yoy": hf_yoy_m.loc[cmp_idx].values,
    }).to_csv(ROOT / "hf_yoy_vs_macro.csv", index=False, encoding="utf-8-sig", float_format="%.6f")

    print(f"高频同比 vs 原始增长因子（月频）相关系数: {corr:.4f}")
    print(f"已保存: hf_growth_factor_synthetic.csv, hf_yoy_vs_macro.csv")
    return out


def export_asset_log_yoy_hp(daily: pd.DataFrame) -> None:
    hp_cols = {}
    for col in ASSETS:
        log_yoy = np.log(daily[col] / daily[col].shift(TRADING_DAYS_YEAR)) * 100
        valid = log_yoy.dropna()
        _, trend = hpfilter(valid.values, lamb=HP_LAMBDA_DAILY)
        hp_cols[col] = pd.Series(trend, index=valid.index)
    pd.DataFrame(hp_cols).reset_index().rename(columns={"index": "date"}).to_csv(
        ROOT / "asset_log_yoy_hp.csv", index=False, encoding="utf-8-sig", float_format="%.6f"
    )


def plot_all() -> None:
    print("\n" + "=" * 65)
    print("【5/5】生成图表")
    print("=" * 65)
    setup_chinese_font()

    # 图1：高频同比 vs 原始增长因子
    df = pd.read_csv(GROWTH_FACTOR_CSV, parse_dates=["date"])
    macro = df.dropna(subset=["raw_growth_factor"]).set_index("date")
    macro_m = macro["raw_growth_factor"].copy()
    macro_m.index = macro_m.index.to_period("M")
    hf = pd.read_csv(ROOT / "hf_growth_factor_synthetic.csv", parse_dates=["date"])
    hf = hf.dropna(subset=["hf_yoy"]).set_index("date")
    hf_m = hf["hf_yoy"].resample("ME").last()
    hf_m.index = hf_m.index.to_period("M")
    idx2 = hf_m.index.intersection(macro_m.index)
    corr = hf_m.loc[idx2].corr(macro_m.loc[idx2])
    hf_w = hf["hf_yoy"].resample("W-FRI").last().dropna()
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={"height_ratios": [2, 1]})
    fig.patch.set_facecolor("#fafafa")
    fig.suptitle("高频宏观因子同比 vs 原始增长因子", fontsize=15, fontweight="bold", y=1.01)
    ax1 = axes[0]
    ax1.set_facecolor("#fafafa")
    ax1.plot(idx2.to_timestamp(), macro_m.loc[idx2], color="#94a3b8", lw=1.5, alpha=0.85, label="原始增长因子（月频）")
    ax1.plot(hf_w.index, hf_w.values, color="#8b5cf6", lw=1.5, label="高频化因子同比（周频采样）")
    ax1.set_title(f"月频对比（相关系数 r = {corr:.3f}）")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)
    ax2 = axes[1]
    ax2.set_facecolor("#fafafa")
    ax2.plot(hf.index, hf["hf_yoy"], color="#059669", lw=0.8, alpha=0.9, label="高频化因子同比（日频）")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel("日期")
    plt.tight_layout()
    plt.savefig(ROOT / "hf_yoy_vs_macro.png", dpi=150, bbox_inches="tight", facecolor="#fafafa")
    plt.close()

    # 图2：拟合资产对数同比 + HP
    daily = load_daily_prices()
    export_asset_log_yoy_hp(daily)
    n = len(ASSETS)
    ncol = min(n, 3)
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.5 * ncol, 4 * nrow))
    fig.patch.set_facecolor("#fafafa")
    fig.suptitle("拟合资产对数同比及 HP 滤波趋势", fontsize=15, fontweight="bold", y=1.02)
    axes_flat = np.atleast_1d(axes).flat
    for ax, col in zip(axes_flat, ASSETS):
        ax.set_facecolor("#fafafa")
        color = ASSET_COLORS[col]
        log_yoy = np.log(daily[col] / daily[col].shift(TRADING_DAYS_YEAR)) * 100
        valid = log_yoy.dropna()
        _, trend = hpfilter(valid.values, lamb=HP_LAMBDA_DAILY)
        ax.plot(valid.index, valid, color=color, alpha=0.25, lw=0.8, label="对数同比")
        ax.plot(valid.index, trend, color=color, lw=1.8, label="HP滤波")
        ax.set_title(col, fontsize=12, fontweight="bold")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)
    for ax in axes_flat[len(ASSETS):]:
        ax.set_visible(False)
    plt.tight_layout()
    plt.savefig(ROOT / "asset_log_yoy_hp.png", dpi=150, bbox_inches="tight", facecolor="#fafafa")
    plt.close()

    print("已保存:")
    for name in [
        "hf_yoy_vs_macro.png",
        "asset_log_yoy_hp.png",
    ]:
        print(f"  {name}")


def main() -> None:
    setup_chinese_font()
    print("宏观增长因子全流程")
    lags = model = None

    if UPDATE_DATA:
        update_growth_factor()
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
            X_lagged = pd.DataFrame({c: apply_shift(X.loc[idx, c], lags[c]) for c in ASSETS})
            common = pd.concat([y.loc[idx], X_lagged], axis=1).dropna()
            model = sm.OLS(common.iloc[:, 0], sm.add_constant(common.iloc[:, 1:])).fit()
        build_hf_growth(lags, model)

    if MAKE_PLOTS:
        plot_all()

    print("\n全流程完成。")


if __name__ == "__main__":
    main()
