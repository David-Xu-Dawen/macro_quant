#!/usr/bin/env python3
"""
信用因子项目 — 一键更新全部数据。

数据源：中债 3Y AA中票-国开债利差 + 企业债AA/国开债财富指数（update_bond_yields.py）
宏观因子（原始）：3Y AA中短票收益率 − 3Y 国开债收益率（水平）
高频因子：企业债AA财富 − 国开债总财富 → HP 去趋势 → 取相反数（标准化为指数形态）

运行: python update_all.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.filters.hp_filter import hpfilter

warnings.filterwarnings("ignore")

OUTPUT_DIR = Path(__file__).parent
RAW_PANEL_CSV = OUTPUT_DIR / "中债_3Y_国开债_AA中票_收益率_2020至今.csv"
RAW_PANEL_SCRIPT = OUTPUT_DIR / "update_bond_yields.py"

SPREAD_FILE = OUTPUT_DIR / "aaa_mtn_3y_minus_treasury_3y.csv"
WEALTH_FILE = OUTPUT_DIR / "credit_wealth_daily.csv"
CREDIT_FACTOR_FILE = OUTPUT_DIR / "credit_factor.csv"

UPDATE_DATA = True
PLOT_START_DATE = "2020-01-01"

# 月度 HP λ=129600；日频水平序列去趋势用 λ × 21^4（保留多年度信用周期）
HP_LAMBDA_MONTHLY = 129600
HP_LAMBDA_DAILY = HP_LAMBDA_MONTHLY * (21**4)
HF_INDEX_VOL = 0.05  # 指数形态波动刻度，使左轴约在 0.9~1.1


def setup_chinese_font() -> None:
    import matplotlib.font_manager as fm

    for font in ["Songti SC", "STHeiti", "Kaiti SC", "PingFang HK", "SimHei"]:
        if font in {f.name for f in fm.fontManager.ttflist}:
            plt.rcParams["font.sans-serif"] = [font, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return


def refresh_raw_panel() -> Path:
    """调用 update_bond_yields.py 增量更新原始面板。"""
    if not RAW_PANEL_SCRIPT.exists():
        raise FileNotFoundError(f"找不到原始面板脚本: {RAW_PANEL_SCRIPT}")
    print(f"  运行: {RAW_PANEL_SCRIPT.name}")
    subprocess.check_call([sys.executable, str(RAW_PANEL_SCRIPT)], cwd=OUTPUT_DIR)
    if not RAW_PANEL_CSV.exists():
        raise FileNotFoundError(f"未生成原始面板 CSV: {RAW_PANEL_CSV}")
    return RAW_PANEL_CSV


def load_raw_panel(csv_path: Path | None = None) -> pd.DataFrame:
    path = csv_path or RAW_PANEL_CSV
    if not path.exists():
        raise FileNotFoundError(
            f"原始面板不存在: {path}，请先运行 update_bond_yields.py"
        )
    raw = pd.read_csv(path, parse_dates=["日期"]).sort_values("日期")
    required = [
        "国开债_3Y",
        "中短期票据AA_3Y",
        "企业债AA财富_3-5年",
        "国开债总财富_3-5年",
        "利差_AA中票减国开_bp",
    ]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError(f"原始面板缺少列: {missing}")
    return raw.dropna(
        subset=[
            "利差_AA中票减国开_bp",
            "企业债AA财富_3-5年",
            "国开债总财富_3-5年",
        ]
    ).reset_index(drop=True)


def panel_to_spread_wealth(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """把原始面板转成利差 / 财富表结构。"""
    spread_df = pd.DataFrame(
        {
            "日期": raw["日期"],
            "AA中票3Y": raw["中短期票据AA_3Y"],
            "国开债3Y": raw["国开债_3Y"],
            "利差BP": raw["利差_AA中票减国开_bp"],
        }
    )
    wealth_df = pd.DataFrame(
        {
            "日期": raw["日期"],
            "corp_wealth": raw["企业债AA财富_3-5年"],
            "gov_wealth": raw["国开债总财富_3-5年"],
        }
    )
    wealth_df["wealth_ratio"] = wealth_df["corp_wealth"] / wealth_df["gov_wealth"]
    return spread_df, wealth_df


def update_market_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    if UPDATE_DATA:
        refresh_raw_panel()

    raw = load_raw_panel()
    spread_df, wealth_df = panel_to_spread_wealth(raw)

    spread_df.to_csv(SPREAD_FILE, index=False, encoding="utf-8-sig")
    wealth_df.to_csv(WEALTH_FILE, index=False, encoding="utf-8-sig")

    print(
        f"  信用利差(AA中票-国开): {len(spread_df)} 日, "
        f"{spread_df['日期'].min().date()} ~ {spread_df['日期'].max().date()}"
    )
    print(
        f"  财富指数(AA/国开): {len(wealth_df)} 日, "
        f"{wealth_df['日期'].min().date()} ~ {wealth_df['日期'].max().date()}"
    )
    return spread_df, wealth_df


def update_credit_factor(spread_df: pd.DataFrame) -> pd.DataFrame:
    """宏观/原始信用因子：3Y AA中短票 − 3Y 国开（收益率利差水平，%）。"""
    setup_chinese_font()

    daily = spread_df.set_index("日期").copy()
    # 收益率单位与源数据一致（%）；利差 = AA − 国开
    daily_spread_pct = (daily["AA中票3Y"] - daily["国开债3Y"]).astype(float)
    panel = pd.DataFrame(
        {
            "credit_factor": daily_spread_pct,
            "spread_bp": daily["利差BP"].astype(float),
        }
    ).dropna(subset=["credit_factor"])
    panel["ym"] = panel.index.to_period("M")
    # 按月取最后观测，日期用真实交易日（避免未完结月写成月末）
    monthly = panel.groupby("ym", sort=True).tail(1)

    out = pd.DataFrame(
        {
            "date": monthly.index,
            "spread_bp": monthly["spread_bp"].values,
            "credit_factor": monthly["credit_factor"].values,  # 百分点数，如 0.30 = 30bp
        }
    )

    out.to_csv(CREDIT_FACTOR_FILE, index=False, float_format="%.6f")

    fig, ax = plt.subplots(figsize=(14, 5), dpi=120)
    ax.plot(
        out["date"],
        out["credit_factor"],
        color="#C0392B",
        linewidth=1.2,
        label="原始信用因子：AA中票−国开",
    )
    ax.set_title(
        "原始信用因子：3Y AA中短票收益率 − 3Y 国开债收益率",
        fontsize=14,
        pad=12,
    )
    ax.set_xlabel("日期")
    ax.set_ylabel("利差 (%)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "credit_factor.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(
        f"  credit_factor.csv: {len(out)} 月, "
        f"最新 {out['credit_factor'].iloc[-1]:.4f}%（{out['spread_bp'].iloc[-1]:.2f} bp）"
    )
    return out


def hp_cycle(series: pd.Series, lamb: float = HP_LAMBDA_DAILY) -> pd.Series:
    """HP 去趋势，返回周期项。"""
    valid = series.dropna()
    if len(valid) < 30:
        raise RuntimeError(f"HP 去趋势样本不足: {series.name or 'series'} n={len(valid)}")
    cycle, _trend = hpfilter(valid.values, lamb=lamb)
    return pd.Series(cycle, index=valid.index, name=series.name)


def build_hf_credit(wealth_df: pd.DataFrame) -> pd.DataFrame:
    """
    高频信用因子（截图口径）:
    中债企业债AA财富(3-5年) − 中债国开债总财富(3-5年) → 去趋势 → 相反数
    再标准化为均值 1 附近的指数形态，便于与原始利差双轴对比。
    """
    corp = wealth_df.set_index("日期")["corp_wealth"].astype(float)
    cdb = wealth_df.set_index("日期")["gov_wealth"].astype(float)

    wealth_diff = corp - cdb
    cycle = hp_cycle(wealth_diff.rename("wealth_diff"))
    hf_raw = -cycle  # 相反数

    std = float(hf_raw.std())
    if std < 1e-12:
        raise RuntimeError("高频信用因子去趋势后波动过小")
    hf_index = 1.0 + (hf_raw - hf_raw.mean()) / std * HF_INDEX_VOL
    hf_mom = hf_index.pct_change() * 100

    return pd.DataFrame(
        {
            "corp_wealth": corp,
            "cdb_wealth": cdb,
            "wealth_diff": wealth_diff,
            "wealth_diff_cycle": cycle,
            "hf_raw": hf_raw,
            "hf_credit_factor": hf_index,
            "hf_mom_pct": hf_mom,
        }
    )


def update_hf_credit_factor(wealth_df: pd.DataFrame, spread_df: pd.DataFrame) -> None:
    setup_chinese_font()

    panel = build_hf_credit(wealth_df)
    meta = {
        "method": (
            "企业债AA财富 − 国开债总财富，HP 去趋势后取相反数，"
            "再标准化为均值 1 的指数形态"
        ),
        "corp_index": "企业债AA财富_3-5年",
        "cdb_index": "国开债总财富_3-5年",
        "detrend": "HP on (corp_wealth - cdb_wealth)",
        "hp_lambda_daily": HP_LAMBDA_DAILY,
        "hf_def": "-HP_cycle(corp - cdb), indexed to mean 1",
        "macro_def": "AA中票3Y − 国开3Y yield spread level",
        "n_obs": int(panel["hf_credit_factor"].notna().sum()),
    }
    with open(OUTPUT_DIR / "hf_regression_results.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    out = panel.reset_index().rename(columns={"index": "日期"})
    if "日期" not in out.columns:
        out = out.rename(columns={out.columns[0]: "日期"})
    out["wealth_ratio"] = out["corp_wealth"] / out["cdb_wealth"]
    keep_cols = [
        "日期",
        "corp_wealth",
        "cdb_wealth",
        "wealth_ratio",
        "wealth_diff",
        "wealth_diff_cycle",
        "hf_raw",
        "hf_credit_factor",
        "hf_mom_pct",
    ]
    out[keep_cols].to_csv(
        OUTPUT_DIR / "hf_credit_factor_daily.csv", index=False, float_format="%.6f"
    )

    monthly = (
        panel[["hf_credit_factor", "hf_mom_pct"]]
        .resample("ME")
        .agg({"hf_credit_factor": "last", "hf_mom_pct": "sum"})
        .dropna(how="all")
        .reset_index()
        .rename(columns={"日期": "date", "index": "date"})
    )
    if "date" not in monthly.columns:
        monthly = monthly.rename(columns={monthly.columns[0]: "date"})
    monthly.to_csv(OUTPUT_DIR / "hf_monthly_fitted.csv", index=False, float_format="%.6f")

    # 原始信用因子：利差水平（小数，右轴 0~0.035 风格）
    macro_dec = (
        (spread_df.set_index("日期")["AA中票3Y"] - spread_df.set_index("日期")["国开债3Y"])
        .astype(float)
        / 100.0
    )
    hf = panel["hf_credit_factor"].dropna()
    both = pd.concat(
        [hf.rename("hf_credit_factor"), macro_dec.rename("credit_spread_decimal")],
        axis=1,
    ).dropna()
    plot_start = pd.Timestamp(PLOT_START_DATE)
    both = both[both.index >= plot_start]
    compare = both.reset_index().rename(columns={"日期": "date", "index": "date"})
    if "date" not in compare.columns:
        compare = compare.rename(columns={compare.columns[0]: "date"})

    # 模仿截图：左轴高频化信用因子，右轴原始信用因子
    fig, ax = plt.subplots(figsize=(14, 5.5), dpi=120)
    (line_hf,) = ax.plot(
        both.index,
        both["hf_credit_factor"],
        color="#2471A3",
        linewidth=1.1,
        label="高频化信用因子",
    )
    ax.set_ylabel("高频化信用因子")
    axb = ax.twinx()
    (line_m,) = axb.plot(
        both.index,
        both["credit_spread_decimal"],
        color="#C0392B",
        linewidth=1.1,
        label="原始信用因子（右轴）",
    )
    axb.set_ylabel("原始信用因子（右轴）")
    ax.set_title("信用因子：高频化 vs 原始", fontsize=14, pad=12)
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
    fig.legend(
        [line_hf, line_m],
        ["高频化信用因子", "原始信用因子（右轴）"],
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18)
    fig.savefig(OUTPUT_DIR / "credit_factor_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    compare.to_csv(OUTPUT_DIR / "hf_mom_vs_macro.csv", index=False, float_format="%.6f")
    for obsolete in ("hf_yoy_vs_macro.csv",):
        p = OUTPUT_DIR / obsolete
        if p.exists():
            p.unlink()

    corr_d = both["hf_credit_factor"].corr(both["credit_spread_decimal"])
    corr_m = (
        pd.concat(
            [
                panel["hf_credit_factor"].resample("ME").last(),
                macro_dec.resample("ME").last(),
            ],
            axis=1,
        )
        .dropna()
        .corr()
        .iloc[0, 1]
    )
    print(
        f"  高频构造: 财富差HP去趋势取反；有效样本 {meta['n_obs']} 日；"
        f"与原始利差相关 日={corr_d:.4f} / 月={corr_m:.4f}"
    )
    print(f"  最新 hf_credit_factor={panel['hf_credit_factor'].dropna().iloc[-1]:.4f}")


def main() -> None:
    print("=" * 60)
    print("开始更新信用因子数据")
    print("=" * 60)

    print("\n[1/3] 市场数据（中债 AA中票 / 国开债）")
    spread_df, wealth_df = update_market_data()

    print("\n[2/3] 原始/宏观信用因子 → credit_factor.csv")
    update_credit_factor(spread_df)

    print("\n[3/3] 高频化信用因子（财富差去趋势取反）")
    update_hf_credit_factor(wealth_df, spread_df)

    print("\n" + "=" * 60)
    print("全部完成。保留文件：")
    for name in [
        "中债_3Y_国开债_AA中票_收益率_2020至今.csv",
        "aaa_mtn_3y_minus_treasury_3y.csv",
        "credit_wealth_daily.csv",
        "credit_factor.csv",
        "credit_factor.png",
        "hf_regression_results.json",
        "hf_monthly_fitted.csv",
        "hf_credit_factor_daily.csv",
        "hf_mom_vs_macro.csv",
        "credit_factor_comparison.png",
    ]:
        print(f"  · {name}")


if __name__ == "__main__":
    main()
