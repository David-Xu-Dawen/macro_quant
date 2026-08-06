"""LightGBM：预测各资产未来超额收益，并按预测选最大超额资产做简单回测。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

from config import BENCHMARK, FORWARD_DAYS, MODEL_DIR, PANEL_DIR


META_COLS = {
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


def load_panel() -> pd.DataFrame:
    parquet = PANEL_DIR / "lgbm_panel.parquet"
    csv = PANEL_DIR / "lgbm_panel.csv"
    if parquet.exists():
        try:
            return pd.read_parquet(parquet)
        except Exception:
            pass
    if not csv.exists():
        raise FileNotFoundError("请先运行 build_lgbm_dataset.py")
    return pd.read_csv(csv, parse_dates=["date"])


def time_split(panel: pd.DataFrame, valid_ratio: float = 0.15, test_ratio: float = 0.15):
    dates = np.array(sorted(panel["date"].unique()))
    n = len(dates)
    n_test = max(1, int(n * test_ratio))
    n_valid = max(1, int(n * valid_ratio))
    test_dates = set(dates[-n_test:])
    valid_dates = set(dates[-(n_test + n_valid) : -n_test])
    train_dates = set(dates[: -(n_test + n_valid)])

    train = panel[panel["date"].isin(train_dates)].copy()
    valid = panel[panel["date"].isin(valid_dates)].copy()
    test = panel[panel["date"].isin(test_dates)].copy()
    return train, valid, test


def feature_columns(panel: pd.DataFrame) -> list[str]:
    return [c for c in panel.columns if c not in META_COLS]


def train_model(train: pd.DataFrame, valid: pd.DataFrame, features: list[str]) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(
        n_estimators=800,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(
        train[features],
        train["y_excess"],
        eval_set=[(valid[features], valid["y_excess"])],
        eval_metric="l2",
        callbacks=[lgb.early_stopping(80, verbose=False)],
    )
    return model


def evaluate(df: pd.DataFrame, pred: np.ndarray, label: str) -> dict:
    y = df["y_excess"].to_numpy()
    rmse = float(np.sqrt(mean_squared_error(y, pred)))
    r2 = float(r2_score(y, pred)) if len(np.unique(y)) > 1 else float("nan")

    tmp = df[["date", "asset", "y_excess", "y_is_best"]].copy()
    tmp["pred"] = pred
    # 每日选预测超额最高的资产，看真实超额与命中率
    picked = tmp.sort_values(["date", "pred"], ascending=[True, False]).groupby("date", as_index=False).head(1)
    avg_excess = float(picked["y_excess"].mean())
    hit = float(picked["y_is_best"].mean())
    # 等权随机基准：全样本均值
    equal_weight = float(tmp.groupby("date")["y_excess"].mean().mean())
    return {
        "split": label,
        "rmse": rmse,
        "r2": r2,
        "pick_avg_excess": avg_excess,
        "pick_best_hit_rate": hit,
        "equal_weight_avg_excess": equal_weight,
        "lift_vs_equal_weight": avg_excess - equal_weight,
        "n_rows": int(len(df)),
        "n_days": int(tmp["date"].nunique()),
    }


def backtest_nav(test: pd.DataFrame, pred: np.ndarray, forward_days: int) -> pd.DataFrame:
    """按预测每日持有最优资产，用真实 forward return 近似叠加净值（非重叠简化）。"""
    tmp = test[["date", "asset", "y_ret", "y_excess"]].copy()
    tmp["pred"] = pred
    picked = tmp.sort_values(["date", "pred"], ascending=[True, False]).groupby("date", as_index=False).head(1)
    # 简化：每隔 forward_days 调仓一次，避免收益重叠重复计
    picked = picked.sort_values("date").iloc[::forward_days].copy()
    picked["strategy_ret"] = picked["y_ret"]
    picked["nav"] = (1 + picked["strategy_ret"]).cumprod()
    return picked[["date", "asset", "y_ret", "y_excess", "pred", "strategy_ret", "nav"]]


def main():
    parser = argparse.ArgumentParser(description="训练 LightGBM 超额收益模型")
    parser.add_argument("--valid-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    args = parser.parse_args()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    panel = load_panel()
    features = feature_columns(panel)
    train, valid, test = time_split(panel, args.valid_ratio, args.test_ratio)

    print(
        f"train={len(train)} valid={len(valid)} test={len(test)} | "
        f"features={len(features)} | benchmark={BENCHMARK} | forward={FORWARD_DAYS}d"
    )

    model = train_model(train, valid, features)

    metrics = []
    preds = {}
    for name, df in [("train", train), ("valid", valid), ("test", test)]:
        pred = model.predict(df[features])
        preds[name] = pred
        metrics.append(evaluate(df, pred, name))

    metrics_df = pd.DataFrame(metrics)
    print("\n=== metrics ===")
    print(metrics_df.to_string(index=False))

    nav = backtest_nav(test, preds["test"], FORWARD_DAYS)
    importance = (
        pd.DataFrame({"feature": features, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    model_path = MODEL_DIR / "lgbm_excess_return.joblib"
    metrics_path = MODEL_DIR / "metrics.csv"
    importance_path = MODEL_DIR / "feature_importance.csv"
    nav_path = MODEL_DIR / "test_nav.csv"
    meta_path = MODEL_DIR / "model_meta.json"

    joblib.dump({"model": model, "features": features}, model_path)
    metrics_df.to_csv(metrics_path, index=False)
    importance.to_csv(importance_path, index=False)
    nav.to_csv(nav_path, index=False)
    meta_path.write_text(
        json.dumps(
            {
                "benchmark": BENCHMARK,
                "forward_days": FORWARD_DAYS,
                "features": features,
                "best_iteration": int(getattr(model, "best_iteration_", model.n_estimators)),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n=== top features ===")
    print(importance.head(15).to_string(index=False))
    print(f"\nsaved model: {model_path}")
    print(f"saved metrics: {metrics_path}")
    print(f"saved nav: {nav_path}")


if __name__ == "__main__":
    main()
