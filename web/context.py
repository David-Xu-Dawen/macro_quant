"""把当前相关矩阵 / 暴露 / 波动 / 模型预测摘要注入到聊天 system prompt。"""

from __future__ import annotations

import json
from pathlib import Path

from factor_corr import CORR_JSON
from factor_exposure import load_latest_json as load_exposure_latest
from hf_factor_corr import CORR_JSON as HF_CORR_JSON

ROOT = Path(__file__).resolve().parent.parent


def _top_pairs(labels: list[str], corr: list[list[float]], k: int = 6) -> list[str]:
    pairs = []
    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            if j <= i:
                continue
            try:
                v = float(corr[i][j])
            except Exception:
                continue
            pairs.append((abs(v), a, b, v))
    pairs.sort(reverse=True)
    return [f"{a} vs {b}: {v:+.2f}" for _, a, b, v in pairs[:k]]


def summarize_corr(path: Path, title: str) -> str:
    if not path.exists():
        return f"{title}: 文件不存在（{path.name}）"
    data = json.loads(path.read_text(encoding="utf-8"))
    labels = data.get("labels", [])
    corr = data.get("corr", [])
    n = data.get("n_months") or data.get("n_weeks") or "?"
    unit = "月" if data.get("n_months") is not None else "周"
    lines = [
        f"{title}",
        f"- 区间: {data.get('start')} ~ {data.get('end')}（n={n}{unit}）",
        f"- 因子: {', '.join(labels)}",
        "- 最强相关对: " + "; ".join(_top_pairs(labels, corr)),
    ]
    return "\n".join(lines)


def summarize_exposure() -> str:
    try:
        data = load_exposure_latest()
    except FileNotFoundError:
        return "因子暴露: 尚未生成 factor_exposure_latest.json"
    lines = [
        "因子暴露（最新缓存）",
        f"- 窗口: {data.get('window_start')} ~ {data.get('window_end')}",
        f"- 滚动窗口: {data.get('rolling_window_weeks')} 周；样本长度: {data.get('sample_length_weeks')} 周",
        f"- Bootstrap: {data.get('bootstrap_samples')}；alpha_scale: {data.get('alpha_scale')}",
        f"- 因子: {', '.join(data.get('factors', []))}",
    ]
    r2 = data.get("r_squared", {})
    if r2:
        r2_txt = "; ".join(f"{a}={float(v):.2f}" for a, v in r2.items())
        lines.append(f"- R²: {r2_txt}")
    matrix = data.get("matrix", {})
    notable = []
    for asset, row in matrix.items():
        for factor, coef in row.items():
            c = float(coef)
            if abs(c) >= 0.25:
                notable.append((abs(c), asset, factor, c))
    notable.sort(reverse=True)
    if notable:
        top = "; ".join(f"{a}×{f}={c:+.2f}" for _, a, f, c in notable[:10])
        lines.append(f"- 较大暴露(|β|≥0.25): {top}")
    return "\n".join(lines)


def summarize_vol_monitor() -> str:
    try:
        from vol_monitor import compute_vol_monitor

        data = compute_vol_monitor()
    except Exception as exc:
        return f"波动监控: 暂不可用（{exc}）"

    lines = [
        "周频波动监控",
        f"- 截至: {data.get('as_of')}；状态: {data.get('status')}；窗口: {data.get('window_weeks')} 周",
        f"- 说明: {data.get('status_note')}",
    ]
    shocks = data.get("shocks") or []
    if shocks:
        shock_txt = "; ".join(
            f"{s.get('factor')}({s.get('direction')}) Z={float(s.get('shock_z')):.2f}"
            for s in shocks[:6]
            if s.get("shock_z") is not None
        )
        if shock_txt:
            lines.append(f"- 本周冲击: {shock_txt}")
    else:
        lines.append("- 本周冲击: 无明显冲击")

    factors = data.get("factors") or []
    if factors:
        top_vol = sorted(
            factors,
            key=lambda x: float(x.get("vol_percentile") or 0),
            reverse=True,
        )[:4]
        vol_txt = "; ".join(
            f"{f.get('factor')} 分位={float(f.get('vol_percentile')):.0f}%"
            for f in top_vol
            if f.get("vol_percentile") is not None
        )
        if vol_txt:
            lines.append(f"- 波动分位偏高: {vol_txt}")

    pressure = data.get("asset_pressure") or []
    if pressure:
        top_p = sorted(
            pressure,
            key=lambda x: abs(float(x.get("pressure") or 0)),
            reverse=True,
        )[:5]
        p_txt = "; ".join(
            f"{p.get('asset')}={float(p.get('pressure')):+.2f}"
            for p in top_p
            if p.get("pressure") is not None
        )
        if p_txt:
            lines.append(f"- 暴露压力偏高: {p_txt}")

    forecast = data.get("forecast") or {}
    if isinstance(forecast, dict) and not forecast.get("error"):
        if forecast.get("prob_high_vol") is not None:
            label = forecast.get("factor_label") or forecast.get("factor") or "因子"
            lines.append(
                f"- 未来4周高波动概率({label}): {100 * float(forecast['prob_high_vol']):.0f}%"
                f"（{forecast.get('level') or ''}）"
            )
        rows = forecast.get("all_factors") or []
        if isinstance(rows, list) and rows:
            high = sorted(
                [
                    r
                    for r in rows
                    if r.get("prob_high_vol") is not None or r.get("prob") is not None
                ],
                key=lambda x: float(x.get("prob_high_vol", x.get("prob"))),
                reverse=True,
            )[:3]
            if high:
                f_txt = "; ".join(
                    f"{r.get('factor') or r.get('factor_label')}="
                    f"{100 * float(r.get('prob_high_vol', r.get('prob'))):.0f}%"
                    for r in high
                )
                lines.append(f"- 高波动概率靠前因子: {f_txt}")
    return "\n".join(lines)


def summarize_model_prediction() -> str:
    try:
        from model_prediction import summarize

        data = summarize()
    except Exception as exc:
        return f"模型预测: 暂不可用（{exc}）"

    display = data.get("metrics_display") or {}
    metrics = data.get("metrics") or {}
    lines = [
        "模型预测（LightGBM + Black-Litterman）",
        f"- 截至: {data.get('as_of')}；标签={data.get('label_mode')}；前瞻 {data.get('forward_days')} 交易日",
        (
            f"- 样本外表现: 年化={display.get('年化')}；波动={display.get('波动')}；"
            f"Sharpe={display.get('Sharpe')}；最大回撤={display.get('最大回撤')}；"
            f"相对等权超额年化={display.get('相对等权超额年化')}；平均换手={display.get('平均换手')}"
        ),
        (
            f"- 回测区间: {metrics.get('start')} ~ {metrics.get('end')}；"
            f"再平衡每 {metrics.get('rebalance_every')} 日；成本 {metrics.get('cost_bps')}bps"
        ),
        (
            "- 注意: latest_weights / OOF 结果主要用于历史评估，"
            "不能直接当作实时可交易仓位。"
        ),
    ]

    weights = data.get("latest_weights") or []
    if weights:
        top_w = "; ".join(
            f"{w.get('asset_cn') or w.get('asset')}={100 * float(w['weight']):.1f}%"
            for w in weights[:6]
            if w.get("weight") is not None
        )
        if top_w:
            lines.append(f"- 最新展示权重(评估用): {top_w}")

    model_cmp = data.get("model_comparison") or []
    if model_cmp:
        cmp_txt = "; ".join(
            f"{m.get('model')} RankIC={m.get('rank_ic')} ICIR={m.get('icir')}"
            for m in model_cmp[:4]
        )
        lines.append(f"- 模型对比: {cmp_txt}")

    feats = data.get("top_features") or []
    if feats:
        feat_txt = "; ".join(
            f"{f.get('feature')}={f.get('importance')}" for f in feats[:6]
        )
        lines.append(f"- 重要特征: {feat_txt}")
    return "\n".join(lines)


def summarize_debate_roles() -> str:
    return """
多 Agent 辩论角色（LangGraph，页面「Agent辩论」）
- 宏观基本面专家：增长、通胀、利率、信用、汇率、地缘与流动性；给出基准宏观情景，核对月/周频信号是否一致，提示政策与数据时点风险。
- 技术量价专家：跨资产动量、趋势、波动、回撤、相关结构与模型 Rank IC/NDCG；强调样本外稳定性，不以训练集拟合优度代替可交易证据。
- 情绪风险专家：地缘政治、黄金/原油、美元、波动冲击、风险偏好与拥挤交易；识别尾部情景与叙事反转，避免把价格代理当成真实情绪。
- 风控对冲专家：因子暴露、集中度、回撤、换手、流动性与情景压力；把观点转成风险预算与约束，不把同期暴露当稳定因果，也不把 OOF 权重当实时仓位。
- 总协调人（投研负责人）：综合四位专家多轮辩论，保留分歧、不以多数票代替证据；输出大类资产配置策略报告（执行摘要、数据口径、证据、共识分歧、情景、配置框架、对冲触发、模型可信度与更新清单）。
""".strip()


def build_live_context() -> str:
    parts = [
        "【当前项目数据摘要】",
        summarize_corr(CORR_JSON, "月频因子相关矩阵"),
        "",
        summarize_corr(HF_CORR_JSON, "周频高频因子相关矩阵"),
        "",
        summarize_exposure(),
        "",
        summarize_vol_monitor(),
        "",
        summarize_model_prediction(),
        "",
        summarize_debate_roles(),
    ]
    return "\n".join(parts)
