"""按因子预测未来 4 周是否高波动（仅单因子，不含综合）。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from factor_exposure import load_mom_panel
from hf_factor_corr import FACTOR_LABELS

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "vol_forecast_by_factor.joblib"
META_PATH = MODEL_DIR / "vol_forecast_by_factor_meta.json"

# 兼容旧入口
LEGACY_MODEL_PATH = MODEL_DIR / "vol_forecast_4w.joblib"

HORIZON = 4
SHORT_W = 4
LONG_W = 13
HIGH_PCT = 75.0
MIN_TRAIN = 80
PURGE_GAP = 4
DEFAULT_FACTOR = FACTOR_LABELS[0]


def _short(name: str) -> str:
    return name.replace("因子", "")


def _factor_changes(panel: pd.DataFrame | None = None) -> pd.DataFrame:
    """周度 MoM/环比变化（与因子暴露口径一致）；不再对水平或同比做 diff。"""
    if panel is None:
        mom = load_mom_panel()
        missing = [c for c in FACTOR_LABELS if c not in mom.columns]
        if missing:
            raise ValueError(f"MoM 面板缺少因子列: {', '.join(missing)}")
        panel = mom[FACTOR_LABELS]
    return panel[FACTOR_LABELS].dropna(how="any")


def build_feature_frame(chg: pd.DataFrame) -> pd.DataFrame:
    feats: dict[str, pd.Series] = {}
    abs_chg = chg.abs()
    for name in FACTOR_LABELS:
        s = _short(name)
        vol13 = chg[name].rolling(LONG_W).std(ddof=0)
        feats[f"{s}_vol{LONG_W}"] = vol13
        feats[f"{s}_volpct"] = vol13.expanding(min_periods=26).rank(pct=True) * 100.0
        feats[f"{s}_abs1"] = abs_chg[name]
        feats[f"{s}_realized4"] = abs_chg[name].rolling(SHORT_W).mean()

    vol_cols = [f"{_short(n)}_vol{LONG_W}" for n in FACTOR_LABELS]
    tmp = pd.DataFrame(feats)
    tmp["xs_vol_mean"] = tmp[vol_cols].mean(axis=1)
    tmp["xs_vol_max"] = tmp[vol_cols].max(axis=1)
    return tmp


def _forward_vol_series(series_abs: pd.Series) -> pd.Series:
    return pd.concat([series_abs.shift(-k) for k in range(1, HORIZON + 1)], axis=1).mean(axis=1)


def build_labels_causal(forward: pd.Series) -> tuple[pd.Series, pd.Series]:
    thr = forward.shift(HORIZON).expanding(min_periods=40).quantile(HIGH_PCT / 100.0)
    y = (forward >= thr).astype(float)
    y[forward.isna() | thr.isna()] = np.nan
    return y.rename("y"), thr.rename("thr")


def _features_for_target(target: str) -> list[str]:
    s = _short(target)
    # 单因子：自身持续性 + 截面环境
    return [f"{s}_vol{LONG_W}", f"{s}_volpct", f"{s}_realized4", f"{s}_abs1", "xs_vol_mean", "xs_vol_max"]


def _metrics(y_true: pd.Series, proba: np.ndarray) -> dict[str, Any]:
    if len(y_true) < 8 or y_true.nunique() < 2:
        return {"n": int(len(y_true)), "accuracy": None, "auc": None, "positive_rate": None}
    pred = (proba >= 0.5).astype(int)
    return {
        "n": int(len(y_true)),
        "accuracy": round(float(accuracy_score(y_true, pred)), 3),
        "auc": round(float(roc_auc_score(y_true, proba)), 3),
        "positive_rate": round(float(y_true.mean()), 3),
    }


def _train_one(X: pd.DataFrame, y: pd.Series, cols: list[str]) -> tuple[Pipeline, dict[str, Any]]:
    cols = [c for c in cols if c in X.columns]
    split = max(int(len(X) * 0.7), MIN_TRAIN // 2)
    te_start = min(split + PURGE_GAP, len(X) - 10)
    X_tr, y_tr = X.iloc[:split], y.iloc[:split]
    X_te, y_te = X.iloc[te_start:], y.iloc[te_start:]

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=1.0,
                    max_iter=1000,
                    class_weight="balanced",
                    solver="lbfgs",
                ),
            ),
        ]
    )
    model.fit(X_tr[cols], y_tr)
    clf: LogisticRegression = model.named_steps["clf"]
    coefs = np.abs(clf.coef_[0])
    importances = [
        {"feature": f, "importance": round(float(v), 5)}
        for f, v in sorted(zip(cols, coefs), key=lambda x: x[1], reverse=True)
    ]
    meta = {
        "feature_names": cols,
        "train_end": str(X_tr.index[-1].date()),
        "test_start": str(X_te.index[0].date()) if len(X_te) else None,
        "train_metrics": _metrics(y_tr, model.predict_proba(X_tr[cols])[:, 1]),
        "test_metrics": _metrics(y_te, model.predict_proba(X_te[cols])[:, 1]),
        "top_features": importances[:5],
    }
    # 自身 vol13 基线
    own = next((c for c in cols if c.endswith(f"_vol{LONG_W}")), None)
    if own and own in X_te.columns and y_te.nunique() > 1:
        try:
            meta["baseline_auc_own_vol13"] = round(float(roc_auc_score(y_te, X_te[own])), 3)
        except ValueError:
            pass
    return model, meta


def train_model(*, save: bool = True) -> dict[str, Any]:
    chg = _factor_changes()
    X_all = build_feature_frame(chg)
    abs_chg = chg.abs()

    targets = list(FACTOR_LABELS)
    models: dict[str, Any] = {}
    per_meta: dict[str, Any] = {}

    for target in targets:
        forward = _forward_vol_series(abs_chg[target])
        y, thr = build_labels_causal(forward)
        data = pd.concat([X_all, y, thr], axis=1).dropna()
        if len(data) < MIN_TRAIN or data["y"].nunique() < 2:
            per_meta[target] = {"error": f"样本不足或标签单一（n={len(data)}）"}
            continue
        cols = _features_for_target(target)
        model, m = _train_one(data[X_all.columns], data["y"].astype(int), cols)
        models[target] = {"model": model, "features": m["feature_names"]}
        m["latest_threshold"] = round(float(data["thr"].iloc[-1]), 6)
        per_meta[target] = m

    summary = {
        "model_type": "logistic_regression_per_factor",
        "label_source": "weekly_hf",
        "feature_source": "weekly_hf",
        "horizon_weeks": HORIZON,
        "high_percentile": HIGH_PCT,
        "purge_gap_weeks": PURGE_GAP,
        "targets": targets,
        "as_of_sample_end": str(X_all.dropna().index[-1].date()),
        "n_samples_features": int(len(X_all.dropna())),
        "by_target": {
            k: {
                "test_metrics": v.get("test_metrics"),
                "train_metrics": v.get("train_metrics"),
                "feature_names": v.get("feature_names"),
                "top_features": v.get("top_features"),
                "baseline_auc_own_vol13": v.get("baseline_auc_own_vol13"),
                "error": v.get("error"),
            }
            for k, v in per_meta.items()
        },
        "note": "仅单因子：预测该因子未来4周波动是否高于其历史因果75分位。",
    }

    if save:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(models, MODEL_PATH)
        META_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _load_models() -> dict[str, Any]:
    if not MODEL_PATH.exists():
        train_model(save=True)
    return joblib.load(MODEL_PATH)


def available_targets() -> list[str]:
    return list(FACTOR_LABELS)


def _level(proba: float) -> str:
    if proba >= 0.65:
        return "偏高风险"
    if proba >= 0.5:
        return "中性偏高"
    if proba >= 0.35:
        return "中性偏低"
    return "偏低风险"


def predict_factor(factor: str = DEFAULT_FACTOR, end: str | None = None) -> dict[str, Any]:
    if factor not in available_targets():
        raise ValueError(f"未知目标: {factor}，可选: {', '.join(available_targets())}")

    models = _load_models()
    if factor not in models:
        raise ValueError(f"目标「{factor}」模型未训练成功，请重新运行 train_vol_forecast.py")

    bundle = models[factor]
    model = bundle["model"]
    features: list[str] = bundle["features"]

    chg = _factor_changes()
    if end is not None:
        chg = chg.loc[: pd.Timestamp(end)]
    X = build_feature_frame(chg).dropna()
    if X.empty:
        raise ValueError("特征不足，无法预测")
    row = X.iloc[-1:]
    missing = [c for c in features if c not in row.columns or pd.isna(row.iloc[0][c])]
    if missing:
        raise ValueError(f"最新特征缺失: {', '.join(missing)}")

    proba = float(model.predict_proba(row[features])[0, 1])
    meta_all = json.loads(META_PATH.read_text(encoding="utf-8")) if META_PATH.exists() else {}
    meta = (meta_all.get("by_target") or {}).get(factor) or {}

    return {
        "factor": factor,
        "factor_label": factor,
        "as_of": str(row.index[-1].date()),
        "horizon_weeks": HORIZON,
        "label_source": "weekly_hf",
        "feature_source": "weekly_hf",
        "prob_high_vol": round(proba, 3),
        "pred_high_vol": bool(proba >= 0.5),
        "level": _level(proba),
        "interpretation": (
            f"{factor}未来 {HORIZON} 周进入高波动的概率约 {proba:.0%}。"
            + ("建议结合该因子暴露与分位警报。" if proba >= 0.5 else "暂未显示高波动偏向。")
        ),
        "top_features": meta.get("top_features") or [],
        "test_metrics": meta.get("test_metrics"),
        "train_metrics": meta.get("train_metrics"),
        "baseline_auc_own_vol13": meta.get("baseline_auc_own_vol13"),
        "model_type": "logistic_regression",
        "available_factors": available_targets(),
        "note": meta_all.get("note", "按因子可选的第二意见"),
    }


def predict_all(end: str | None = None) -> dict[str, Any]:
    rows = []
    errors = {}
    for t in available_targets():
        try:
            rows.append(predict_factor(t, end=end))
        except Exception as exc:
            errors[t] = str(exc)
    # 保持因子固定顺序，便于下拉选择
    order = {name: i for i, name in enumerate(FACTOR_LABELS)}
    rows.sort(key=lambda r: order.get(r.get("factor"), 999))
    as_of = rows[0]["as_of"] if rows else None
    return {
        "as_of": as_of,
        "horizon_weeks": HORIZON,
        "label_source": "weekly_hf",
        "items": [
            {
                "factor": r["factor"],
                "factor_label": r["factor_label"],
                "prob_high_vol": r["prob_high_vol"],
                "pred_high_vol": r["pred_high_vol"],
                "level": r["level"],
                "test_auc": (r.get("test_metrics") or {}).get("auc"),
            }
            for r in rows
        ],
        "errors": errors,
        "note": "各因子独立模型；可选查看某一因子的未来4周高波动概率。",
    }


def predict_latest(end: str | None = None) -> dict[str, Any]:
    """默认返回首个单因子预测，并附带全部因子摘要。"""
    out = predict_factor(DEFAULT_FACTOR, end=end)
    try:
        out["all_factors"] = predict_all(end=end).get("items", [])
    except Exception:
        out["all_factors"] = []
    return out


def ensure_model() -> dict[str, Any]:
    if MODEL_PATH.exists() and META_PATH.exists():
        return json.loads(META_PATH.read_text(encoding="utf-8"))
    return train_model(save=True)
