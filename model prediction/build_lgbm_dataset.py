"""把原始日线拼成面板，构造特征与未来超额收益标签，供 LightGBM 使用。"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from config import (
    ASSETS,
    BENCHMARK,
    FORWARD_DAYS,
    PANEL_DIR,
    RAW_DIR,
    START_DATE,
    TRADE_UNIVERSE,
)


def load_raw(asset_ids: list[str]) -> pd.DataFrame:
    frames = []
    for asset_id in asset_ids:
        path = RAW_DIR / f"{asset_id}.csv"
        if not path.exists():
            print(f"[warn] missing raw file: {path}")
            continue
        df = pd.read_csv(path, parse_dates=["date"])
        frames.append(df)
    if not frames:
        raise FileNotFoundError("未找到任何 raw 数据，请先运行 fetch_akshare_data.py")
    return pd.concat(frames, ignore_index=True)


def build_wide_close(long_df: pd.DataFrame, assets: list[str]) -> pd.DataFrame:
    close = (
        long_df[long_df["asset"].isin(assets)]
        .pivot_table(index="date", columns="asset", values="close", aggfunc="last")
        .sort_index()
    )
    return close


def add_features_for_asset(asset: str, close: pd.DataFrame, ret: pd.DataFrame) -> pd.DataFrame:
    """为单个资产构造横截面+时序特征。"""
    s = close[asset]
    r = ret[asset]
    available = [a for a in TRADE_UNIVERSE if a in close.columns]

    feat = pd.DataFrame({"date": close.index, "asset": asset})
    feat["close"] = s.to_numpy()
    feat["ret_1d"] = r.to_numpy()
    feat["ret_5d"] = s.pct_change(5).to_numpy()
    feat["ret_10d"] = s.pct_change(10).to_numpy()
    feat["ret_20d"] = s.pct_change(20).to_numpy()
    feat["ret_60d"] = s.pct_change(60).to_numpy()
    feat["vol_20d"] = r.rolling(20).std().to_numpy()
    feat["vol_60d"] = r.rolling(60).std().to_numpy()
    feat["mom_20_60"] = feat["ret_20d"] - feat["ret_60d"]
    feat["ma_ratio_20"] = (s / s.rolling(20).mean() - 1.0).to_numpy()
    feat["ma_ratio_60"] = (s / s.rolling(60).mean() - 1.0).to_numpy()
    feat["drawdown_60"] = (s / s.rolling(60).max() - 1.0).to_numpy()

    # 相对全市场（当日有报价资产）的相对强弱
    mkt = ret[available].mean(axis=1)
    feat["rel_ret_5d"] = feat["ret_5d"] - close[available].pct_change(5).mean(axis=1).to_numpy()
    feat["rel_ret_20d"] = feat["ret_20d"] - close[available].pct_change(20).mean(axis=1).to_numpy()
    feat["beta_proxy_60"] = r.rolling(60).corr(mkt).to_numpy()

    # 与基准的利差/超额（历史）
    if BENCHMARK in ret.columns:
        feat["excess_1d"] = (r - ret[BENCHMARK]).to_numpy()
        feat["excess_5d"] = feat["ret_5d"] - close[BENCHMARK].pct_change(5).to_numpy()
        feat["excess_20d"] = feat["ret_20d"] - close[BENCHMARK].pct_change(20).to_numpy()

    # 跨资产宏观风格：股债商品相对位置
    for other in ["csi300", "bond_gov", "gold_au", "crude_sc", "spx", "hsi"]:
        if other == asset or other not in close.columns:
            continue
        feat[f"spread20_vs_{other}"] = feat["ret_20d"] - close[other].pct_change(20).to_numpy()

    return feat


def build_dataset(
    start: str = START_DATE,
    forward_days: int = FORWARD_DAYS,
    universe: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    universe = universe or TRADE_UNIVERSE
    needed = sorted(set(universe) | {BENCHMARK} | set(ASSETS.keys()))
    long_df = load_raw(needed)

    # 交易日轴：以 A 股主要指数交集为主，避免美股/商品日历污染过多
    anchor_assets = [a for a in ["csi300", "sse50", "bond_gov"] if a in long_df["asset"].unique()]
    if not anchor_assets:
        anchor_assets = universe[:1]
    calendar = (
        long_df[long_df["asset"].isin(anchor_assets)]
        .groupby("date")["asset"]
        .nunique()
        .loc[lambda s: s >= max(1, len(anchor_assets) - 1)]
        .index.sort_values()
    )

    close = build_wide_close(long_df, list(set(universe) | {BENCHMARK}))
    close = close.reindex(calendar).ffill()
    close = close[close.index >= pd.Timestamp(start)]
    ret = close.pct_change()

    frames = []
    for asset in universe:
        if asset not in close.columns:
            continue
        feat = add_features_for_asset(asset, close, ret)
        # 未来超额收益标签：asset future return - benchmark future return
        fwd_asset = close[asset].shift(-forward_days) / close[asset] - 1.0
        fwd_bench = close[BENCHMARK].shift(-forward_days) / close[BENCHMARK] - 1.0
        feat["y_excess"] = (fwd_asset - fwd_bench).to_numpy()
        feat["y_ret"] = fwd_asset.to_numpy()
        feat["forward_days"] = forward_days
        feat["cn_name"] = ASSETS[asset]["cn_name"]
        feat["asset_class"] = ASSETS[asset]["asset_class"]
        frames.append(feat)

    panel = pd.concat(frames, ignore_index=True)
    panel = panel.dropna(subset=["y_excess"]).sort_values(["date", "asset"]).reset_index(drop=True)

    # 当日横截面：谁是未来超额最高（用于分类/排序评估）
    panel["y_rank"] = panel.groupby("date")["y_excess"].rank(ascending=False, method="first")
    panel["y_is_best"] = (panel["y_rank"] == 1).astype(int)

    feature_cols = [
        c
        for c in panel.columns
        if c
        not in {
            "date",
            "asset",
            "cn_name",
            "asset_class",
            "close",
            "y_excess",
            "y_ret",
            "y_rank",
            "y_is_best",
            "forward_days",
        }
    ]
    return panel, pd.DataFrame({"feature": feature_cols})


def main():
    parser = argparse.ArgumentParser(description="构建 LightGBM 超额收益训练集")
    parser.add_argument("--start", default=START_DATE)
    parser.add_argument("--forward-days", type=int, default=FORWARD_DAYS)
    args = parser.parse_args()

    PANEL_DIR.mkdir(parents=True, exist_ok=True)
    panel, feat_meta = build_dataset(start=args.start, forward_days=args.forward_days)

    panel_path = PANEL_DIR / "lgbm_panel.parquet"
    csv_path = PANEL_DIR / "lgbm_panel.csv"
    feat_path = PANEL_DIR / "feature_list.csv"

    try:
        panel.to_parquet(panel_path, index=False)
    except Exception:
        panel_path = None
    panel.to_csv(csv_path, index=False)
    feat_meta.to_csv(feat_path, index=False)

    print(f"panel rows={len(panel)}, dates={panel['date'].nunique()}, assets={panel['asset'].nunique()}")
    print(f"date range: {panel['date'].min().date()} ~ {panel['date'].max().date()}")
    print(f"features ({len(feat_meta)}): {feat_meta['feature'].tolist()}")
    print(f"saved: {csv_path}")
    if panel_path:
        print(f"saved: {panel_path}")
    print(f"saved: {feat_path}")


if __name__ == "__main__":
    main()
