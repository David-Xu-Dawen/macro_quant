"""宏观量化工作台 · 纯 Streamlit 版（无需 Render / FastAPI）。

Streamlit Community Cloud:
  Main file path = streamlit_app.py

本地:
  streamlit run streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web"
if str(WEB) not in sys.path:
    sys.path.insert(0, str(WEB))

from factor_corr import available_months, compute_corr  # noqa: E402
from factor_exposure import available_weeks, compute_exposure, load_latest_json  # noqa: E402
from hf_factor_corr import available_weeks as hf_available_weeks  # noqa: E402
from hf_factor_corr import compute_hf_corr  # noqa: E402
from model_prediction import figure_path, list_profiles, summarize  # noqa: E402
from vol_forecast import DEFAULT_FACTOR, FACTOR_LABELS as VOL_FACTORS  # noqa: E402
from vol_forecast import predict_factor  # noqa: E402
from vol_monitor import compute_vol_monitor  # noqa: E402


st.set_page_config(
    page_title="宏观量化工作台",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .stApp {
        background:
          radial-gradient(circle at 12% 0%, rgba(143,45,45,0.08), transparent 28%),
          linear-gradient(145deg, #f8f3ed 0%, #f2f5f1 55%, #f7f1eb 100%);
      }
      h1, h2, h3 { color: #1d252d !important; letter-spacing: 0.02em; }
      div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.78);
        border: 1px solid rgba(51,42,35,0.10);
        border-radius: 14px;
        padding: 10px 12px;
      }
      .mq-note {
        border-left: 3px solid #8f2d2d;
        background: rgba(143,45,45,0.06);
        border-radius: 0 12px 12px 0;
        padding: 10px 14px;
        color: #514348;
        margin: 0 0 14px;
        font-size: 0.92rem;
        line-height: 1.55;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


def _corr_heatmap(matrix: list[list[float]], labels: list[str], title: str):
    df = pd.DataFrame(matrix, index=labels, columns=labels)
    fig = px.imshow(
        df,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        aspect="auto",
        title=title,
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=48, b=10),
        coloraxis_colorbar=dict(title="ρ"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="PingFang SC, Microsoft YaHei, sans-serif"),
    )
    return fig


def page_matrix():
    st.header("因子相关性矩阵")
    st.markdown(
        '<div class="mq-note">左侧月频偏配置叙事；右侧周频同口径。数值为 Pearson 相关。</div>',
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("月频低频矩阵")
        months = available_months()
        if len(months) < 3:
            st.error("月频样本不足")
            return
        start_m, end_m = st.select_slider(
            "月频区间",
            options=months,
            value=(months[max(0, len(months) - 36)], months[-1]),
            key="mf_range",
        )
        try:
            data = compute_corr(start_m, end_m)
            st.caption(f"样本 {data['n_months']} 个月 · {data['start']} ~ {data['end']}")
            st.plotly_chart(
                _corr_heatmap(data["corr"], data["labels"], "月频相关"),
                use_container_width=True,
            )
        except Exception as exc:
            st.error(str(exc))

    with c2:
        st.subheader("周频高频矩阵")
        weeks = hf_available_weeks()
        if len(weeks) < 12:
            st.error("周频样本不足")
            return
        start_w, end_w = st.select_slider(
            "周频区间",
            options=weeks,
            value=(weeks[max(0, len(weeks) - 156)], weeks[-1]),
            key="hf_range",
        )
        try:
            data = compute_hf_corr(start_w, end_w)
            st.caption(f"样本 {data['n_weeks']} 周 · {data['start']} ~ {data['end']}")
            st.plotly_chart(
                _corr_heatmap(data["corr"], data["labels"], "周频相关"),
                use_container_width=True,
            )
        except Exception as exc:
            st.error(str(exc))


def page_vol():
    st.header("周频波动警报")
    st.markdown(
        '<div class="mq-note">基于与暴露一致的周度 MoM：13 周波动分位 + 本周 2σ 冲击。</div>',
        unsafe_allow_html=True,
    )
    try:
        mon = compute_vol_monitor(window=13, shock_z=2.0)
    except Exception as exc:
        st.error(f"波动监控加载失败：{exc}")
        return

    st.metric("状态", mon.get("status", "—"))
    st.caption(mon.get("status_note", ""))
    st.caption(f"截至 {mon.get('as_of', '—')}")

    left, right = st.columns(2)
    with left:
        st.subheader("因子波动风险")
        factors = pd.DataFrame(mon.get("factors") or [])
        if factors.empty:
            st.info("暂无因子波动数据")
        else:
            show = factors[["factor", "vol_percentile", "vol_level", "week_change", "shock_z", "is_shock"]].copy()
            show.columns = ["因子", "波动分位", "水平", "本周变化", "冲击Z", "是否冲击"]
            st.dataframe(show, use_container_width=True, hide_index=True)

    with right:
        st.subheader("本周冲击")
        shocks = pd.DataFrame(mon.get("shocks") or [])
        if shocks.empty:
            st.success("本周无明显冲击")
        else:
            st.dataframe(shocks, use_container_width=True, hide_index=True)

        st.subheader("暴露压力 Top")
        pressure = pd.DataFrame(mon.get("asset_pressure") or [])
        if pressure.empty:
            st.info("暂无暴露压力")
        else:
            st.dataframe(pressure.head(10), use_container_width=True, hide_index=True)

    st.subheader("波动风险预测（4 周）")
    factor = st.selectbox(
        "查看因子",
        VOL_FACTORS,
        index=VOL_FACTORS.index(DEFAULT_FACTOR) if DEFAULT_FACTOR in VOL_FACTORS else 0,
    )
    try:
        pred = predict_factor(factor)
        m1, m2, m3 = st.columns(3)
        m1.metric("高波动概率", f"{100 * float(pred.get('prob_high_vol', 0)):.1f}%")
        m2.metric("风险等级", pred.get("level") or "—")
        m3.metric("预测高波动", "是" if pred.get("pred_high_vol") else "否")
        st.caption(pred.get("interpretation") or pred.get("note") or "")
    except Exception as exc:
        st.warning(f"波动预测暂不可用：{exc}")


def page_exposure():
    st.header("资产宏观因子暴露")
    st.markdown(
        '<div class="mq-note">默认展示最新预计算结果；也可按结束周重算（Bootstrap 较慢，云端建议用预计算）。</div>',
        unsafe_allow_html=True,
    )

    mode = st.radio("数据来源", ["最新预计算", "按结束周重算"], horizontal=True)
    try:
        if mode == "最新预计算":
            payload = load_latest_json()
        else:
            info = available_weeks()
            weeks = info.get("weeks") or []
            if not weeks:
                st.error("没有可用结束周")
                return
            end = st.selectbox("结束周", weeks[::-1], index=0)
            rolling = st.selectbox("滚动窗口（周）", [104, 156, 260, 315], index=2)
            with st.spinner("正在重算暴露（可能需要 1–3 分钟）..."):
                payload = compute_exposure(
                    end=end,
                    rolling_window_weeks=rolling,
                    bootstrap=500,  # 云端加速；本地预计算仍是 3000
                )
    except Exception as exc:
        st.error(str(exc))
        return

    st.caption(
        f"窗口 {payload.get('window_start', '—')} ~ {payload.get('window_end', '—')} · "
        f"滚动 {payload.get('rolling_window_weeks', '—')} 周 · "
        f"Bootstrap {payload.get('bootstrap_samples', '—')}"
    )

    matrix = payload.get("matrix") or {}
    if not matrix:
        st.warning("暴露矩阵为空")
        return
    df = pd.DataFrame(matrix).T
    # 常见列顺序
    cols = [c for c in ["增长因子", "通胀因子", "利率因子", "信用因子", "汇率因子", "地缘因子", "流动性因子"] if c in df.columns]
    df = df[cols + [c for c in df.columns if c not in cols]]
    st.dataframe(df.style.format("{:.3f}").background_gradient(cmap="RdBu_r", axis=None), use_container_width=True)

    r2 = payload.get("r_squared") or payload.get("r2") or {}
    if r2:
        st.subheader("R²")
        r2_df = pd.Series(r2, name="R2").sort_values(ascending=False).to_frame()
        st.bar_chart(r2_df)


def page_model():
    st.header("模型预测 · 大类配置")
    st.markdown(
        '<div class="mq-note">滚动回测净值 / 回撤 / 最新配置权重。档位只改组合层，不改 LightGBM 信号。</div>',
        unsafe_allow_html=True,
    )

    profiles = list_profiles()
    ready = [p for p in profiles if p.get("ready")]
    if not ready:
        st.error("尚未生成模型预测结果，请先在本地跑 model prediction。")
        return

    labels = {p["key"]: p["label"] for p in ready}
    key = st.selectbox(
        "激进程度",
        options=list(labels.keys()),
        format_func=lambda k: labels[k],
        index=list(labels.keys()).index("balanced") if "balanced" in labels else 0,
    )
    try:
        data = summarize(aggression=key)
    except Exception as exc:
        st.error(str(exc))
        return

    profile = data.get("aggression_profile") or {}
    st.caption(
        f"{data.get('aggression_label')} · 上限 {100 * float(profile.get('weight_max') or 0):.0f}% · "
        f"CVaR厌恶 {profile.get('cvar_risk_aversion')} · {data.get('as_of') or ''}"
    )

    metrics = data.get("metrics") or {}
    mcols = st.columns(4)
    mcols[0].metric("年化", f"{100 * float(metrics.get('ann_return') or 0):.1f}%")
    mcols[1].metric("Sharpe", f"{float(metrics.get('sharpe') or 0):.2f}")
    mcols[2].metric("最大回撤", f"{100 * float(metrics.get('max_drawdown') or 0):.1f}%")
    mcols[3].metric("相对等权", f"{100 * float(metrics.get('excess_ann_vs_ew') or 0):.1f}%")

    img_cols = st.columns(2)
    for i, name in enumerate(["01_nav_curve.png", "02_drawdown.png"]):
        try:
            path = figure_path(name, aggression=key)
            if path.exists():
                with img_cols[i]:
                    st.image(str(path), use_container_width=True)
        except Exception:
            pass

    st.subheader("最新配置权重")
    weights = data.get("latest_weights") or []
    if not weights:
        st.info("暂无权重")
        return
    wdf = pd.DataFrame(weights)
    show = pd.DataFrame(
        {
            "资产": wdf.get("asset_cn", wdf.get("asset")),
            "权重": (wdf["weight"] * 100).round(1).astype(str) + "%",
            "得分": wdf.get("score"),
            "μ_BL": (wdf["mu_bl"] * 100).round(2).astype(str) + "%" if "mu_bl" in wdf else None,
        }
    )
    st.dataframe(show, use_container_width=True, hide_index=True)
    chart_df = pd.DataFrame(
        {
            "资产": wdf.get("asset_cn", wdf.get("asset")),
            "权重": wdf["weight"],
        }
    ).set_index("资产")
    st.bar_chart(chart_df)


def main():
    st.title("宏观量化工作台")
    st.caption("Streamlit 版 · 数据与原 FastAPI 工作台同源 · AI 助手/辩论需本机 Ollama，云端暂不开放")

    tab1, tab2, tab3, tab4 = st.tabs(["因子矩阵", "波动警报", "因子暴露", "模型预测"])
    with tab1:
        page_matrix()
    with tab2:
        page_vol()
    with tab3:
        page_exposure()
    with tab4:
        page_model()


if __name__ == "__main__":
    main()
