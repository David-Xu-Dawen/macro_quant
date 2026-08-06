"""运行 LightGBM 标签周期、风险调整目标和 asset_id 消融实验。

所有产物写入 output/lgbm_experiments，不覆盖生产面板或模型。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import CORR_THRESHOLD, OUTPUT_DIR, START_DATE, TRADE_UNIVERSE
from macro_features import add_labels, build_panel_features, load_price_panel
from model_lgbm import rank_metrics, train_with_timeseries_cv


EXPERIMENTS = [
    {"name": "return_20_with_asset_id", "forward_days": 20, "target": "return", "include_asset_id": True},
    {"name": "return_20_no_asset_id", "forward_days": 20, "target": "return", "include_asset_id": False},
    {"name": "return_40_no_asset_id", "forward_days": 40, "target": "return", "include_asset_id": False},
    {"name": "return_60_no_asset_id", "forward_days": 60, "target": "return", "include_asset_id": False},
    {"name": "risk_adjusted_40_no_asset_id", "forward_days": 40, "target": "risk_adjusted", "include_asset_id": False},
]


def _ensemble_fold_metrics(
    oof: pd.DataFrame,
    cv_metrics: pd.DataFrame,
    forward_days: int,
) -> pd.DataFrame:
    valid_dates = sorted(oof.loc[oof["oof_pred"].notna(), "date"].unique())
    cursor = 0
    rows: list[dict] = []
    for _, fold in cv_metrics.iterrows():
        n_dates = int(fold["n_valid_days"])
        dates = valid_dates[cursor : cursor + n_dates]
        cursor += n_dates
        sample = oof[oof["date"].isin(dates)].dropna(subset=["oof_pred"])
        metrics = rank_metrics(
            sample,
            sample["oof_pred"].to_numpy(),
            forward_days=forward_days,
        )
        metrics["fold"] = int(fold["fold"])
        rows.append(metrics)
    return pd.DataFrame(rows)


def run_one(base_panel: pd.DataFrame, close: pd.DataFrame, spec: dict, root: Path) -> dict:
    name = str(spec["name"])
    forward_days = int(spec["forward_days"])
    target = str(spec["target"])
    include_asset_id = bool(spec["include_asset_id"])
    excluded = () if include_asset_id else ("asset_id",)
    out_dir = root / name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== {name} ===")
    panel = add_labels(
        base_panel,
        close,
        forward_days=forward_days,
        mode="ranking",
        target=target,
    )
    ranker = train_with_timeseries_cv(
        panel,
        mode="ranking",
        corr_threshold=CORR_THRESHOLD,
        purge_gap=forward_days,
        exclude_features=excluded,
    )
    regression_panel = panel.copy()
    regression_panel["y"] = regression_panel["y_target"]
    regressor = train_with_timeseries_cv(
        regression_panel,
        mode="regression",
        corr_threshold=CORR_THRESHOLD,
        purge_gap=forward_days,
        exclude_features=excluded,
    )

    oof = ranker.oof_pred.copy()
    rank_pct = ranker.oof_pred["oof_pred"].groupby(oof["date"]).rank(pct=True, method="average")
    reg_pct = regressor.oof_pred["oof_pred"].groupby(oof["date"]).rank(pct=True, method="average")
    oof["rank_pred_pct"] = rank_pct
    oof["reg_pred_pct"] = reg_pct
    oof["model_disagreement"] = (rank_pct - reg_pct).abs()
    oof["oof_pred"] = 0.25 * rank_pct + 0.75 * reg_pct
    valid = oof.dropna(subset=["oof_pred"])

    return_metrics = rank_metrics(
        valid,
        valid["oof_pred"].to_numpy(),
        target_col="y_ret",
        forward_days=forward_days,
    )
    target_metrics = rank_metrics(
        valid,
        valid["oof_pred"].to_numpy(),
        target_col="y_target",
        forward_days=forward_days,
    )
    folds = _ensemble_fold_metrics(oof, ranker.cv_metrics, forward_days)

    ranker.cv_metrics.to_csv(out_dir / "cv_metrics_ranker.csv", index=False)
    regressor.cv_metrics.to_csv(out_dir / "cv_metrics_regressor.csv", index=False)
    folds.to_csv(out_dir / "cv_metrics_ensemble.csv", index=False)
    oof.to_csv(out_dir / "oof_predictions.csv", index=False)
    ranker.importance.to_csv(out_dir / "feature_importance_ranker.csv", index=False)
    regressor.importance.to_csv(out_dir / "feature_importance_regressor.csv", index=False)

    row = {
        **spec,
        "n_features": len(ranker.features),
        "rank_ic": return_metrics["rank_ic"],
        "icir": return_metrics["icir"],
        "ndcg_at_3": return_metrics["ndcg_at_3"],
        "top1_hit_rate": return_metrics["top1_hit_rate"],
        "top3_overlap": return_metrics["top3_overlap"],
        "top1_excess": return_metrics["top1_excess"],
        "top3_excess": return_metrics["top3_excess"],
        "target_rank_ic": target_metrics["rank_ic"],
        "target_icir": target_metrics["icir"],
        "min_fold_ic": float(folds["rank_ic"].min()),
        "recent_fold_ic": float(folds.sort_values("fold").iloc[-1]["rank_ic"]),
        "positive_folds": int((folds["rank_ic"] > 0).sum()),
        "fold_ic_std": float(folds["rank_ic"].std(ddof=1)),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(row, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return row


def main() -> None:
    root = OUTPUT_DIR / "lgbm_experiments"
    root.mkdir(parents=True, exist_ok=True)
    close = load_price_panel(TRADE_UNIVERSE, start=START_DATE)
    base_panel = build_panel_features(close, TRADE_UNIVERSE)
    rows = [run_one(base_panel, close, spec, root) for spec in EXPERIMENTS]
    summary = pd.DataFrame(rows).sort_values(
        ["rank_ic", "min_fold_ic", "top3_excess"],
        ascending=False,
    )
    summary.to_csv(root / "summary.csv", index=False)
    (root / "summary.json").write_text(
        json.dumps(summary.to_dict(orient="records"), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print("\n=== Experiment summary ===")
    print(
        summary[
            [
                "name", "rank_ic", "icir", "min_fold_ic", "recent_fold_ic",
                "positive_folds", "top3_excess", "n_features",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
