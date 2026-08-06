"""聊天工具：查询相关矩阵 / 因子暴露。"""

from __future__ import annotations

import json
from typing import Any

from factor_corr import CORR_JSON, compute_corr
from factor_exposure import compute_exposure, load_latest_json as load_exposure_latest
from hf_factor_corr import CORR_JSON as HF_CORR_JSON
from hf_factor_corr import compute_hf_corr

TOOL_DEFS = [
    {
        "type": "function",
        "function": {
            "name": "get_corr_matrix",
            "description": "查询月频宏观因子相关矩阵。可指定起止月份 YYYY-MM；不指定则返回最新缓存。",
            "parameters": {
                "type": "object",
                "properties": {
                    "start": {"type": "string", "description": "起始月份 YYYY-MM"},
                    "end": {"type": "string", "description": "结束月份 YYYY-MM"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_hf_corr_matrix",
            "description": "查询周频高频因子相关矩阵。可指定起止周 YYYY-MM-DD；不指定则返回最新缓存。",
            "parameters": {
                "type": "object",
                "properties": {
                    "start": {"type": "string", "description": "起始周 YYYY-MM-DD"},
                    "end": {"type": "string", "description": "结束周 YYYY-MM-DD"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_factor_exposure",
            "description": "查询资产对宏观因子的暴露矩阵。可指定结束周 end(YYYY-MM-DD) 与滚动窗口周数；不指定 end 则返回最新缓存。",
            "parameters": {
                "type": "object",
                "properties": {
                    "end": {"type": "string", "description": "结束周 YYYY-MM-DD"},
                    "rolling_window_weeks": {
                        "type": "integer",
                        "description": "滚动窗口周数，例如 412",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_vol_monitor",
            "description": "查询周频宏观因子波动分位、冲击 Z 值、暴露压力，以及未来4周高波动概率。",
            "parameters": {
                "type": "object",
                "properties": {
                    "window": {"type": "integer", "minimum": 4, "maximum": 52},
                    "shock_z": {"type": "number", "minimum": 1, "maximum": 5},
                    "end": {"type": "string", "description": "可选结束周 YYYY-MM-DD"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_model_backtest_summary",
            "description": (
                "查询 LightGBM+BL 的样本外回测指标、模型对比和历史权重摘要。"
                "注意：其中 OOF/最新展示权重只用于评估，不是实时信号。"
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


def _compact_corr(data: dict) -> dict:
    labels = data.get("labels", [])
    corr = data.get("corr", [])
    pairs = []
    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            if j <= i:
                continue
            try:
                v = round(float(corr[i][j]), 3)
            except Exception:
                continue
            pairs.append({"a": a, "b": b, "corr": v})
    pairs.sort(key=lambda x: abs(x["corr"]), reverse=True)
    return {
        "start": data.get("start"),
        "end": data.get("end"),
        "n": data.get("n_months") or data.get("n_weeks"),
        "labels": labels,
        "top_pairs": pairs[:12],
        "corr": [[round(float(x), 3) for x in row] for row in corr],
    }


def _compact_exposure(data: dict) -> dict:
    matrix = {
        asset: {f: round(float(c), 3) for f, c in row.items()}
        for asset, row in data.get("matrix", {}).items()
    }
    r2 = {a: round(float(v), 3) for a, v in data.get("r_squared", {}).items()}
    return {
        "window_start": data.get("window_start"),
        "window_end": data.get("window_end"),
        "rolling_window_weeks": data.get("rolling_window_weeks"),
        "sample_length_weeks": data.get("sample_length_weeks"),
        "bootstrap_samples": data.get("bootstrap_samples"),
        "alpha_scale": data.get("alpha_scale"),
        "factors": data.get("factors"),
        "assets": data.get("assets"),
        "r_squared": r2,
        "matrix": matrix,
    }


def run_tool(name: str, arguments: dict[str, Any] | str | None) -> str:
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments) if arguments.strip() else {}
        except json.JSONDecodeError:
            arguments = {}
    args = arguments or {}

    try:
        if name == "get_corr_matrix":
            start, end = args.get("start"), args.get("end")
            if start and end:
                data = compute_corr(start, end)
            else:
                if not CORR_JSON.exists():
                    return json.dumps({"error": "macro_factor_corr.json 不存在"}, ensure_ascii=False)
                data = json.loads(CORR_JSON.read_text(encoding="utf-8"))
            return json.dumps(_compact_corr(data), ensure_ascii=False)

        if name == "get_hf_corr_matrix":
            start, end = args.get("start"), args.get("end")
            if start and end:
                data = compute_hf_corr(start, end)
            else:
                if not HF_CORR_JSON.exists():
                    return json.dumps({"error": "macro_hf_factor_corr.json 不存在"}, ensure_ascii=False)
                data = json.loads(HF_CORR_JSON.read_text(encoding="utf-8"))
            return json.dumps(_compact_corr(data), ensure_ascii=False)

        if name == "get_factor_exposure":
            end = args.get("end")
            rolling = args.get("rolling_window_weeks")
            if end:
                data = compute_exposure(end=end, rolling_window_weeks=rolling)
            else:
                data = load_exposure_latest()
            return json.dumps(_compact_exposure(data), ensure_ascii=False)

        if name == "get_vol_monitor":
            from vol_monitor import compute_vol_monitor

            data = compute_vol_monitor(
                window=int(args.get("window", 13)),
                shock_z=float(args.get("shock_z", 2.0)),
                end=args.get("end"),
            )
            forecast = data.get("forecast") or {}
            compact_forecast = None
            if isinstance(forecast, dict) and not forecast.get("error"):
                compact_forecast = {
                    "as_of": forecast.get("as_of"),
                    "prob_high_vol": forecast.get("prob_high_vol"),
                    "level": forecast.get("level"),
                    "all_factors": (forecast.get("all_factors") or [])[:6],
                }
            elif isinstance(forecast, dict):
                compact_forecast = {"error": forecast.get("error")}
            return json.dumps(
                {
                    "as_of": data.get("as_of"),
                    "status": data.get("status"),
                    "status_note": data.get("status_note"),
                    "factors": data.get("factors"),
                    "shocks": data.get("shocks"),
                    "asset_pressure": data.get("asset_pressure"),
                    "forecast": compact_forecast,
                    "method_note": data.get("method_note"),
                },
                ensure_ascii=False,
            )

        if name == "get_model_backtest_summary":
            from model_prediction import summarize as summarize_model_prediction

            data = summarize_model_prediction()
            return json.dumps(
                {
                    "as_of": data.get("as_of"),
                    "metrics": data.get("metrics"),
                    "metrics_display": data.get("metrics_display"),
                    "model_comparison": data.get("model_comparison"),
                    "top_features": (data.get("top_features") or [])[:12],
                    "latest_weights": data.get("latest_weights"),
                    "forward_days": data.get("forward_days"),
                    "label_mode": data.get("label_mode"),
                    "note": data.get("note"),
                    "warning": (
                        "latest_weights 当前来自 bl_weights_latest/OOF 展示路径；"
                        "仅可用于历史评估，不能视为实时可交易仓位。"
                    ),
                },
                ensure_ascii=False,
            )

        return json.dumps({"error": f"未知工具: {name}"}, ensure_ascii=False)
    except Exception as exc:
        return json.dumps({"error": str(exc)}, ensure_ascii=False)
