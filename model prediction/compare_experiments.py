"""比较两个独立实验输出，生成可审计的 A/B 摘要。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output"

METRICS = [
    "strategy_ann_return",
    "strategy_ann_vol",
    "strategy_sharpe",
    "strategy_max_drawdown",
    "portfolio_cvar_95",
    "avg_turnover",
    "cvar_optimizer_usage",
    "optimizer_fallback_rate",
]
STRESS_WINDOWS = {
    "2018Q4": ("2018-10-01", "2018-12-31"),
    "2020Q1": ("2020-01-01", "2020-03-31"),
    "2022": ("2022-01-01", "2022-12-31"),
}


def load(name: str) -> dict:
    path = OUTPUT / name / "rolling_metrics.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def stress_metrics(name: str) -> list[dict]:
    nav = pd.read_csv(OUTPUT / name / "rolling_nav.csv", parse_dates=["date"]).set_index("date")
    rows = []
    for window, (start, end) in STRESS_WINDOWS.items():
        series = nav.loc[start:end, "strategy"].dropna()
        if len(series) < 2:
            continue
        drawdown = series / series.cummax() - 1.0
        rows.append(
            {
                "experiment": name,
                "window": window,
                "return": float(series.iloc[-1] / series.iloc[0] - 1.0),
                "max_drawdown": float(drawdown.min()),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default="baseline_ab")
    parser.add_argument("--candidate", default="candidate_ab")
    args = parser.parse_args()

    baseline = load(args.baseline)
    candidate = load(args.candidate)
    rows = []
    for metric in METRICS:
        old = baseline.get(metric)
        new = candidate.get(metric)
        rows.append(
            {
                "metric": metric,
                "baseline": old,
                "candidate": new,
                "delta": (new - old) if old is not None and new is not None else None,
            }
        )
    comparison = pd.DataFrame(rows)
    comparison.to_csv(OUTPUT / "ab_comparison.csv", index=False)
    stress = pd.DataFrame(stress_metrics(args.baseline) + stress_metrics(args.candidate))
    stress.to_csv(OUTPUT / "ab_stress_comparison.csv", index=False)
    payload = {
        "baseline": args.baseline,
        "candidate": args.candidate,
        "metrics": comparison.to_dict(orient="records"),
        "stress_windows": stress.to_dict(orient="records"),
        "candidate_pass": bool(
            candidate.get("strategy_max_drawdown", -1)
            > baseline.get("strategy_max_drawdown", -1)
            and candidate.get("portfolio_cvar_95", 1)
            < baseline.get("portfolio_cvar_95", 1)
            and candidate.get("optimizer_fallback_rate", 1) <= 0.05
        ),
        "note": "A/B 为同一 OOF 信号下的组合层诊断；生产结论仍以严格 walk-forward 为准。",
    }
    (OUTPUT / "ab_comparison.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(comparison.to_string(index=False))
    print(f"candidate_pass={payload['candidate_pass']}")


if __name__ == "__main__":
    main()
