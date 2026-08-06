#!/usr/bin/env python3
"""
通胀因子项目 — 一键更新全部数据。

运行: python update_all.py

输出:
  commodities.csv          三商品周度价格（齐全起记）
  inflation_factor.csv     CPI/PPI 合成通胀因子
  hf_*.csv / .json         高频通胀因子
  *.png                    图表
"""

from __future__ import annotations

import itertools
import json
import re
import warnings
from pathlib import Path

import akshare as ak
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

warnings.filterwarnings("ignore")

OUTPUT_DIR = Path(__file__).parent
COMMODITIES_FILE = OUTPUT_DIR / "commodities.csv"
INFLATION_FILE = OUTPUT_DIR / "inflation_factor.csv"

VOL_WINDOW = 12
ASSETS = ["pork", "brent", "rebar"]
ASSET_LABELS = {"pork": "猪肉", "brent": "布伦特原油", "rebar": "螺纹钢"}

LAG_RANGE = range(0, 4)  # 0~3 月，X.shift(lag) 仅使用当期及历史数据
WEEKS_PER_MONTH = 4
WEEKS_YEAR = 52
NAV_BASE = 1.0
HF_PLOT_SCALE = 10
PLOT_START_DATE = "2016-10-01"


# ── 工具 ──────────────────────────────────────────────────────────

def setup_chinese_font():
    import matplotlib.font_manager as fm

    for font in ["Songti SC", "STHeiti", "Kaiti SC", "PingFang HK", "SimHei"]:
        if font in {f.name for f in fm.fontManager.ttflist}:
            plt.rcParams["font.sans-serif"] = [font, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return


def parse_chinese_month(s: str) -> pd.Timestamp:
    m = re.match(r"(\d{4})年(\d{1,2})月份", str(s).strip())
    if not m:
        raise ValueError(f"无法解析日期: {s}")
    year, month = int(m.group(1)), int(m.group(2))
    return pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)


# ── 1. commodities.csv ────────────────────────────────────────────

def fetch_commodities() -> pd.DataFrame:
    """猪肉周度 + 原油/螺纹日频(ffill)，三列齐全起记。"""
    pork_raw = ak.index_hog_spot_price()
    pork = pork_raw.rename(columns={"日期": "date", "成交均价": "pork_price"})
    pork["date"] = pd.to_datetime(pork["date"]).astype("datetime64[ns]")
    pork = pork[["date", "pork_price"]].drop_duplicates("date").sort_values("date")

    brent = ak.futures_foreign_hist(symbol="OIL")
    brent["date"] = pd.to_datetime(brent["date"]).astype("datetime64[ns]")
    brent = brent.set_index("date")["close"].rename("brent_close")

    rebar_raw = ak.futures_main_sina(symbol="RB0", start_date="20100101", end_date="20991231")
    rebar_raw["date"] = pd.to_datetime(rebar_raw["日期"]).astype("datetime64[ns]")
    rebar = rebar_raw.set_index("date")["收盘价"].rename("rebar_close")

    oil_rebar = pd.concat([brent, rebar], axis=1, sort=True).sort_index().ffill().reset_index()

    out = pd.merge_asof(pork, oil_rebar, on="date", direction="backward")
    out = out.dropna(subset=["pork_price", "brent_close", "rebar_close"], how="any")
    return out.reset_index(drop=True)


def update_commodities() -> pd.DataFrame:
    df = fetch_commodities()
    df.to_csv(COMMODITIES_FILE, index=False, float_format="%.4f")
    print(f"  commodities.csv: {len(df)} 周, {df['date'].min().date()} ~ {df['date'].max().date()}")
    return df


# ── 2. inflation_factor.csv ───────────────────────────────────────

def update_inflation_factor() -> pd.DataFrame:
    setup_chinese_font()

    cpi_raw = ak.macro_china_cpi()
    ppi_raw = ak.macro_china_ppi()

    cpi = pd.DataFrame({
        "date": cpi_raw["月份"].map(parse_chinese_month),
        "cpi_index": pd.to_numeric(cpi_raw["全国-当月"], errors="coerce"),
        "cpi_yoy_official": pd.to_numeric(cpi_raw["全国-同比增长"], errors="coerce"),
    }).sort_values("date")

    ppi = pd.DataFrame({
        "date": ppi_raw["月份"].map(parse_chinese_month),
        "ppi_index": pd.to_numeric(ppi_raw["当月"], errors="coerce"),
        "ppi_yoy_official": pd.to_numeric(ppi_raw["当月同比增长"], errors="coerce"),
    }).sort_values("date")

    cpi["cpi_yoy"] = cpi["cpi_index"] - 100
    ppi["ppi_yoy"] = ppi["ppi_index"] - 100
    merged = pd.merge(cpi, ppi, on="date", how="inner")

    vol_cpi = merged["cpi_yoy"].rolling(VOL_WINDOW, min_periods=VOL_WINDOW).std()
    vol_ppi = merged["ppi_yoy"].rolling(VOL_WINDOW, min_periods=VOL_WINDOW).std()
    inv_sum = 1 / vol_cpi + 1 / vol_ppi
    w_cpi = (1 / vol_cpi) / inv_sum
    w_ppi = (1 / vol_ppi) / inv_sum

    result = pd.DataFrame({
        "date": merged["date"],
        "cpi_index": merged["cpi_index"],
        "ppi_index": merged["ppi_index"],
        "cpi_yoy": merged["cpi_yoy"],
        "ppi_yoy": merged["ppi_yoy"],
        "cpi_vol": vol_cpi,
        "ppi_vol": vol_ppi,
        "w_cpi": w_cpi,
        "w_ppi": w_ppi,
        "inflation_factor": w_cpi * merged["cpi_yoy"] + w_ppi * merged["ppi_yoy"],
        "cpi_yoy_official": merged["cpi_yoy_official"],
        "ppi_yoy_official": merged["ppi_yoy_official"],
    })
    result.to_csv(INFLATION_FILE, index=False, float_format="%.4f")

    plot_df = result.dropna(subset=["inflation_factor"])
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    fig.suptitle("中国通胀因子：CPI/PPI 同比 + 波动率倒数加权", fontsize=15, fontweight="bold")
    axes[0].plot(plot_df["date"], plot_df["cpi_yoy"], label="CPI 同比 (%)", color="#E74C3C")
    axes[0].plot(plot_df["date"], plot_df["ppi_yoy"], label="PPI 同比 (%)", color="#3498DB")
    axes[0].legend(); axes[0].grid(True, alpha=0.3); axes[0].set_title("CPI / PPI 同比增速")
    axes[1].plot(plot_df["date"], plot_df["w_cpi"], label="CPI 权重", color="#E74C3C")
    axes[1].plot(plot_df["date"], plot_df["w_ppi"], label="PPI 权重", color="#3498DB")
    axes[1].set_ylim(0, 1); axes[1].legend(); axes[1].grid(True, alpha=0.3)
    axes[2].plot(plot_df["date"], plot_df["inflation_factor"], label="通胀因子", color="#2C3E50", linewidth=2)
    axes[2].axhline(0, color="gray", linestyle="--", alpha=0.6)
    axes[2].legend(); axes[2].grid(True, alpha=0.3); axes[2].set_title("综合通胀因子")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "inflation_factor.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  inflation_factor.csv: {len(result)} 月, 最新 {plot_df['inflation_factor'].iloc[-1]:.2f}%")
    return result


# ── 3. 高频通胀因子（基于 commodities.csv）────────────────────────

def load_commodity_prices(df: pd.DataFrame | None = None) -> pd.DataFrame:
    if df is None:
        df = pd.read_csv(COMMODITIES_FILE, parse_dates=["date"])
    df = df.sort_values("date").set_index("date")
    return df.rename(columns={
        "pork_price": "pork",
        "brent_close": "brent",
        "rebar_close": "rebar",
    }).astype(float)


def load_macro_y() -> pd.Series:
    macro = pd.read_csv(INFLATION_FILE, parse_dates=["date"])
    return macro.set_index(macro["date"].dt.to_period("M"))["inflation_factor"].dropna()


def monthly_log_yoy(weekly: pd.DataFrame) -> pd.DataFrame:
    monthly = weekly.resample("ME").last()
    log_yoy = np.log(monthly / monthly.shift(12)) * 100
    log_yoy.index = log_yoy.index.to_period("M")
    return log_yoy


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


def update_hf_inflation_factor(weekly: pd.DataFrame) -> None:
    setup_chinese_font()

    X_m = monthly_log_yoy(weekly)
    y = load_macro_y()
    idx = X_m.index.intersection(y.index)
    best, lags = search_joint_lags(y.loc[idx], X_m.loc[idx])
    model = best["model"]
    betas = {c: float(model.params[c]) for c in ASSETS}
    weights = normalized_weights(betas)
    meta = {
        "lags_months": lags, "weights": weights, "betas": betas,
        "intercept": float(model.params["const"]),
        "r_squared": float(model.rsquared),
        "adj_r_squared": float(model.rsquared_adj),
        "bic": float(model.bic), "n_obs": int(best["n"]),
        "assets": ASSETS, "price_file": COMMODITIES_FILE.name,
    }

    with open(OUTPUT_DIR / "hf_regression_results.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # 月频拟合
    X_lag = pd.DataFrame({c: X_m.loc[idx, c].shift(lags[c]) for c in ASSETS}, index=idx)
    common = pd.concat([y.loc[idx].rename("inflation_factor"), X_lag], axis=1).dropna()
    fitted = np.asarray(model.predict(sm.add_constant(common[ASSETS])))
    monthly_fitted = pd.DataFrame({
        "date": [p.to_timestamp("M") for p in common.index],
        "inflation_factor": common["inflation_factor"].values,
        "macro_fitted": fitted,
        **{f"log_yoy_{c}": common[c].values for c in ASSETS},
        "residual": common["inflation_factor"].values - fitted,
    })
    monthly_fitted.to_csv(OUTPUT_DIR / "hf_monthly_fitted.csv", index=False, float_format="%.6f")

    # 周频对数环比 → 净值 → 对数同比
    wow = np.log(weekly / weekly.shift(1)) * 100
    parts = [weights[c] * wow[c].shift(lags[c] * WEEKS_PER_MONTH) for c in ASSETS]
    hf_wow = pd.concat(parts, axis=1).sum(axis=1, min_count=1)
    valid = hf_wow.dropna()
    hf_nav = NAV_BASE * (1.0 + valid / 100.0).cumprod()
    hf_nav = hf_nav.reindex(weekly.index)
    hf_yoy = np.log(hf_nav / hf_nav.shift(WEEKS_YEAR)) * 100

    out = weekly.copy()
    out["hf_wow"] = hf_wow
    out["hf_nav"] = hf_nav
    out["hf_yoy_pct"] = hf_yoy
    out = out.reset_index().rename(columns={"index": "date"})
    out.to_csv(OUTPUT_DIR / "hf_inflation_weekly.csv", index=False, float_format="%.6f")

    # 对比图
    macro_m = load_macro_y()
    macro_m.index = [p.to_timestamp("M") for p in macro_m.index]
    hf_plot = (hf_yoy / HF_PLOT_SCALE).dropna()
    macro_on_w = macro_m.reindex(hf_plot.index, method="ffill")
    compare = pd.DataFrame({
        "date": hf_plot.index,
        "hf_yoy_pct": hf_yoy.reindex(hf_plot.index).values,
        "hf_yoy_plot": hf_plot.values,
        "inflation_factor": macro_on_w.values,
    })
    compare["diff_plot"] = compare["hf_yoy_plot"] - compare["inflation_factor"]
    plot_start = pd.Timestamp(PLOT_START_DATE)
    compare = compare[compare["date"] >= plot_start]
    macro_plot = macro_m[macro_m.index >= plot_start]
    hf_plot = hf_plot[hf_plot.index >= plot_start]

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    fig.suptitle("高频宏观通胀因子 vs 原始通胀增长因子", fontsize=15, fontweight="bold")
    axes[0].plot(macro_plot.index, macro_plot.values, label="原始通胀因子（月度）", color="#2C3E50", linewidth=2)
    axes[0].plot(hf_plot.index, hf_plot.values,
                 label=f"高频合成因子同比÷{HF_PLOT_SCALE}（周频）", color="#E74C3C", linewidth=1.5)
    axes[0].axhline(0, color="gray", linestyle="--", alpha=0.6)
    axes[0].set_ylabel("同比增速 (%)"); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[0].set_title("同比序列对比")
    axes[1].bar(compare["date"], compare["diff_plot"], width=4, color="#3498DB", alpha=0.6,
                label=f"高频÷{HF_PLOT_SCALE} - 原始")
    axes[1].axhline(0, color="gray", linestyle="--", alpha=0.6)
    axes[1].set_ylabel("偏离 (%)"); axes[1].set_xlabel("日期"); axes[1].legend(); axes[1].grid(True, alpha=0.3)
    axes[1].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "hf_inflation_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    compare.to_csv(OUTPUT_DIR / "hf_yoy_vs_macro.csv", index=False, float_format="%.4f")

    print(f"  hf 回归 R²={meta['r_squared']:.4f}, 相关系数={compare['hf_yoy_plot'].corr(compare['inflation_factor']):.4f}")
    for c in ASSETS:
        print(f"    {ASSET_LABELS[c]}: 滞后{lags[c]}月, 权重={weights[c]:+.4f}")


# ── 主入口 ────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("开始更新全部数据")
    print("=" * 60)

    print("\n[1/3] 商品数据 → commodities.csv")
    commodities = update_commodities()

    print("\n[2/3] 通胀因子 → inflation_factor.csv")
    update_inflation_factor()

    print("\n[3/3] 高频通胀因子（基于 commodities.csv）")
    update_hf_inflation_factor(load_commodity_prices(commodities))

    print("\n" + "=" * 60)
    print("全部完成。保留文件：")
    for f in [
        "commodities.csv", "inflation_factor.csv", "inflation_factor.png",
        "hf_regression_results.json", "hf_monthly_fitted.csv",
        "hf_inflation_weekly.csv", "hf_yoy_vs_macro.csv", "hf_inflation_comparison.png",
    ]:
        print(f"  · {f}")


if __name__ == "__main__":
    main()
