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
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web"
if str(WEB) not in sys.path:
    sys.path.insert(0, str(WEB))

from factor_corr import FACTOR_LABELS as CORR_FACTORS  # noqa: E402
from factor_corr import available_months, compute_corr  # noqa: E402
from factor_exposure import available_weeks, compute_exposure, load_latest_json  # noqa: E402
from hf_factor_corr import available_weeks as hf_available_weeks  # noqa: E402
from hf_factor_corr import compute_hf_corr  # noqa: E402
from model_prediction import figure_path, list_profiles, summarize  # noqa: E402
from pair_compare import compare_pair  # noqa: E402
from rag import retrieve  # noqa: E402
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
      .block-container {
        max-width: 1420px;
        padding-top: 1.6rem;
        padding-bottom: 5rem;
      }
      h1, h2, h3 { color: #1d252d !important; letter-spacing: 0.02em; }
      div[data-testid="stTabs"] button {
        border-radius: 10px 10px 0 0;
        font-weight: 650;
      }
      div[data-testid="stTabs"] button[aria-selected="true"] {
        color: #8f2d2d;
        background: rgba(143,45,45,0.06);
      }
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
      .mq-hero {
        position: relative;
        overflow: hidden;
        padding: 22px 24px;
        margin: 0 0 18px;
        border: 1px solid rgba(51,42,35,0.10);
        border-left: 4px solid #8f2d2d;
        border-radius: 18px;
        background: linear-gradient(112deg, rgba(255,255,255,0.96), rgba(255,250,245,0.86));
        box-shadow: 0 12px 32px rgba(67,50,37,0.08);
      }
      .mq-hero h1 { margin: 0; font-size: 1.75rem; }
      .mq-hero p { margin: 7px 0 0; color: #6f747c; line-height: 1.6; }
      .mq-section-title {
        margin: 18px 0 8px;
        padding-bottom: 8px;
        border-bottom: 1px solid rgba(51,42,35,0.10);
        color: #262b31;
        font-weight: 720;
        font-size: 1.08rem;
      }
      .mq-chip-row { display:flex; flex-wrap:wrap; gap:8px; margin:10px 0 12px; }
      .mq-chip {
        display:inline-block;
        padding:6px 10px;
        border:1px solid rgba(143,45,45,0.13);
        border-radius:999px;
        background:#fffaf7;
        color:#694043;
        font-size:.82rem;
      }
      div[data-testid="stDataFrame"], div[data-testid="stPlotlyChart"] {
        border-radius: 14px;
        overflow: hidden;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


def _hero(title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="mq-hero"><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def _cached_lf_corr(start: str, end: str) -> dict:
    return compute_corr(start, end)


@st.cache_data(show_spinner=False)
def _cached_hf_corr(start: str, end: str) -> dict:
    return compute_hf_corr(start, end)


@st.cache_data(show_spinner=False)
def _cached_pair(a: str, b: str, start: str, end: str) -> dict:
    return compare_pair(a, b, start=start, end=end)


@st.cache_data(show_spinner=False, ttl=600)
def _cached_vol() -> dict:
    return compute_vol_monitor(window=13, shock_z=2.0)


@st.cache_data(show_spinner=False)
def _cached_latest_exposure() -> dict:
    return load_latest_json()


@st.cache_data(show_spinner=False)
def _cached_model(key: str) -> dict:
    return summarize(aggression=key)


def _corr_heatmap(matrix: list[list[float]], labels: list[str], title: str) -> go.Figure:
    df = pd.DataFrame(matrix, index=labels, columns=labels)
    text = np.vectorize(lambda x: f"{float(x):+.2f}")(df.to_numpy())
    fig = go.Figure(
        go.Heatmap(
            x=labels,
            y=labels,
            z=df.to_numpy(),
            zmin=-1,
            zmax=1,
            colorscale="RdBu",
            reversescale=True,
            text=text,
            texttemplate="%{text}",
            textfont={"size": 13},
            colorbar={"title": "ρ", "thickness": 12},
            hovertemplate="%{y} × %{x}<br>ρ=%{z:.3f}<extra></extra>",
        )
    )
    # Streamlit 暂不返回 Heatmap 单元格点击。叠加几乎透明的 Scatter，
    # 让每个非对角格成为可选择点，并把 x/y 因子名送回 Python。
    click_x: list[str] = []
    click_y: list[str] = []
    click_corr: list[float] = []
    for i, row_name in enumerate(labels):
        for j, col_name in enumerate(labels):
            if i == j:
                continue
            click_x.append(col_name)
            click_y.append(row_name)
            click_corr.append(float(df.iloc[i, j]))
    fig.add_trace(
        go.Scatter(
            x=click_x,
            y=click_y,
            mode="markers",
            customdata=np.asarray(click_corr)[:, None],
            marker={"size": 34, "opacity": 0.01, "color": "#8f2d2d"},
            hovertemplate="%{y} × %{x}<br>ρ=%{customdata[0]:.3f}<extra>点击拆解</extra>",
            showlegend=False,
            name="点击拆解",
        )
    )
    fig.update_layout(
        title={"text": title, "x": 0.02, "xanchor": "left"},
        height=520,
        margin=dict(l=12, r=12, t=52, b=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="PingFang SC, Microsoft YaHei, sans-serif"),
        dragmode="select",
        clickmode="event+select",
        xaxis={"side": "top", "tickangle": -28},
        yaxis={"autorange": "reversed"},
    )
    return fig


def _selected_pair(event: object) -> tuple[str, str] | None:
    """兼容 Streamlit Plotly 事件的 dict / 属性两种返回形式。"""
    if event is None:
        return None
    try:
        selection = event.selection
    except AttributeError:
        selection = event.get("selection", {}) if isinstance(event, dict) else {}
    try:
        points = selection.points
    except AttributeError:
        points = selection.get("points", []) if isinstance(selection, dict) else []
    if not points:
        return None
    point = points[-1]
    if not isinstance(point, dict):
        return None
    a, b = point.get("y"), point.get("x")
    if a and b and a != b:
        return str(a), str(b)
    return None


def _line_figure(
    dates: list[str],
    a_values: list[float | None],
    b_values: list[float | None],
    factor_a: str,
    factor_b: str,
    title: str,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=dates, y=a_values, name=factor_a, mode="lines", line={"color": "#8f2d2d", "width": 2})
    )
    fig.add_trace(
        go.Scatter(x=dates, y=b_values, name=factor_b, mode="lines", line={"color": "#2563eb", "width": 2})
    )
    fig.add_hline(y=0, line={"color": "rgba(100,116,139,.35)", "dash": "dot"})
    fig.update_layout(
        title=title,
        height=340,
        margin=dict(l=20, r=12, t=48, b=24),
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.12},
        yaxis_title="窗口内 Z-score",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,.55)",
    )
    return fig


def _rolling_figure(pair: dict) -> go.Figure:
    lf = pd.DataFrame(pair["lf"].get("rolling_corr") or [])
    hf = pd.DataFrame(pair["hf"].get("rolling_corr") or [])
    fig = go.Figure()
    if not lf.empty:
        fig.add_trace(
            go.Scatter(
                x=lf["t"], y=lf["corr"], name="月频 12月滚动相关",
                mode="lines", line={"color": "#8f2d2d", "width": 2},
            )
        )
    if not hf.empty:
        fig.add_trace(
            go.Scatter(
                x=hf["t"], y=hf["corr"], name="周频 52周滚动相关",
                mode="lines", line={"color": "#2563eb", "width": 2},
            )
        )
    fig.add_hline(y=0, line={"color": "rgba(100,116,139,.45)", "dash": "dot"})
    fig.update_yaxes(range=[-1.05, 1.05], title="相关系数")
    fig.update_layout(
        title="滚动相关",
        height=360,
        margin=dict(l=20, r=12, t=48, b=24),
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.12},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,.55)",
    )
    return fig


def _pair_drilldown(factor_a: str, factor_b: str, start: str, end: str) -> None:
    st.markdown(
        f'<div class="mq-section-title">{factor_a} × {factor_b}：高低频差异拆解</div>',
        unsafe_allow_html=True,
    )
    try:
        pair = _cached_pair(factor_a, factor_b, start, end)
    except Exception as exc:
        st.error(f"因子对拆解失败：{exc}")
        return

    chips = [
        ("月频相关", pair["lf"].get("corr")),
        ("周频相关", pair["hf"].get("corr")),
        ("Δ(周-月)", pair["delta"].get("hf_minus_lf")),
        ("高频月末抽样", pair["hf_month_end"].get("corr")),
        ("同月低频相关", pair["hf_month_end"].get("lf_corr_same_months")),
    ]
    chip_html = "".join(
        f'<span class="mq-chip">{name} <strong>{"—" if value is None else f"{float(value):+.2f}"}</strong></span>'
        for name, value in chips
    )
    st.markdown(f'<div class="mq-chip-row">{chip_html}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="mq-note">{pair.get("diagnosis", "")}</div>', unsafe_allow_html=True)
    st.caption(
        f"区间 {pair['start']} ~ {pair['end']} · 月频 {pair['lf']['n']} 个月 · "
        f"周频 {pair['hf']['n']} 周 · 月末共同样本 {pair['hf_month_end']['n']} 个月"
    )

    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            _line_figure(
                pair["lf"]["dates"], pair["lf"]["a_z"], pair["lf"]["b_z"],
                factor_a, factor_b, "月频序列（窗口内标准化）",
            ),
            width="stretch",
            key=f"lf_pair_{factor_a}_{factor_b}_{start}_{end}",
        )
    with right:
        st.plotly_chart(
            _line_figure(
                pair["hf"]["dates"], pair["hf"]["a_z"], pair["hf"]["b_z"],
                factor_a, factor_b, "周频序列（窗口内标准化）",
            ),
            width="stretch",
            key=f"hf_pair_{factor_a}_{factor_b}_{start}_{end}",
        )
    st.plotly_chart(
        _rolling_figure(pair),
        width="stretch",
        key=f"rolling_pair_{factor_a}_{factor_b}_{start}_{end}",
    )


def _compact_vol_monitor() -> None:
    st.markdown('<div class="mq-section-title">周频波动警报</div>', unsafe_allow_html=True)
    try:
        mon = _cached_vol()
    except Exception as exc:
        st.warning(f"波动监控暂不可用：{exc}")
        return
    factors = mon.get("factors") or []
    high = [f for f in factors if float(f.get("vol_percentile") or 0) >= 75]
    shocks = mon.get("shocks") or []
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("风险状态", mon.get("status", "—"))
    c2.metric("高波动因子", len(high))
    c3.metric("本周冲击", len(shocks))
    c4.metric("截至", mon.get("as_of", "—"))
    st.caption(mon.get("status_note", ""))


def page_matrix():
    _hero(
        "宏观因子相关性对比矩阵",
        "月频偏配置叙事，周频观察高频变化。点击任意非对角格，查看标准化序列、滚动相关与差异诊断。",
    )
    _compact_vol_monitor()
    st.markdown('<div class="mq-section-title">月频 vs 周频矩阵</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    selected: tuple[str, str] | None = None
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
            data = _cached_lf_corr(start_m, end_m)
            st.caption(f"样本 {data['n_months']} 个月 · {data['start']} ~ {data['end']}")
            event_lf = st.plotly_chart(
                _corr_heatmap(data["corr"], data["labels"], "月频相关"),
                width="stretch",
                key=f"lf_matrix_{start_m}_{end_m}",
                on_select="rerun",
                selection_mode="points",
            )
            selected = _selected_pair(event_lf) or selected
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
            data = _cached_hf_corr(start_w, end_w)
            st.caption(f"样本 {data['n_weeks']} 周 · {data['start']} ~ {data['end']}")
            event_hf = st.plotly_chart(
                _corr_heatmap(data["corr"], data["labels"], "周频相关"),
                width="stretch",
                key=f"hf_matrix_{start_w}_{end_w}",
                on_select="rerun",
                selection_mode="points",
            )
            selected = _selected_pair(event_hf) or selected
        except Exception as exc:
            st.error(str(exc))

    if selected:
        st.session_state["selected_pair"] = selected

    with st.expander("若设备上点击格子不方便，可手动选择因子对", expanded=False):
        m1, m2 = st.columns(2)
        manual_a = m1.selectbox("因子 A", CORR_FACTORS, index=0)
        manual_b = m2.selectbox("因子 B", CORR_FACTORS, index=1)
        if st.button("查看因子对拆解", type="primary"):
            st.session_state["selected_pair"] = (manual_a, manual_b)

    pair = st.session_state.get("selected_pair")
    if pair and pair[0] != pair[1]:
        common_start = max(str(start_m)[:7], str(start_w)[:7])
        common_end = min(str(end_m)[:7], str(end_w)[:7])
        _pair_drilldown(pair[0], pair[1], common_start, common_end)


def page_vol():
    _hero("周频波动警报", "13 周波动分位、本周 2σ 冲击、资产暴露压力与未来 4 周高波动概率。")
    st.markdown(
        '<div class="mq-note">基于与暴露一致的周度 MoM：13 周波动分位 + 本周 2σ 冲击。</div>',
        unsafe_allow_html=True,
    )
    try:
        mon = _cached_vol()
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
            st.dataframe(show, width="stretch", hide_index=True)

    with right:
        st.subheader("本周冲击")
        shocks = pd.DataFrame(mon.get("shocks") or [])
        if shocks.empty:
            st.success("本周无明显冲击")
        else:
            st.dataframe(shocks, width="stretch", hide_index=True)

        st.subheader("暴露压力 Top")
        pressure = pd.DataFrame(mon.get("asset_pressure") or [])
        if pressure.empty:
            st.info("暂无暴露压力")
        else:
            st.dataframe(pressure.head(10), width="stretch", hide_index=True)

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
    _hero("资产宏观因子暴露矩阵", "资产周度收益对宏观高频因子变化的 Bootstrap + Lasso 暴露。")
    st.markdown(
        '<div class="mq-note">默认展示最新预计算结果；也可按结束周重算（Bootstrap 较慢，云端建议用预计算）。</div>',
        unsafe_allow_html=True,
    )

    mode = st.radio("数据来源", ["最新预计算", "按结束周重算"], horizontal=True)
    try:
        if mode == "最新预计算":
            payload = _cached_latest_exposure()
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
    st.dataframe(
        df.style.format("{:.3f}").background_gradient(cmap="RdBu_r", axis=None),
        width="stretch",
    )

    r2 = payload.get("r_squared") or payload.get("r2") or {}
    if r2:
        st.subheader("R²")
        r2_df = pd.Series(r2, name="R2").sort_values(ascending=False).to_frame()
        st.bar_chart(r2_df)


def page_model():
    _hero("模型预测 · 大类配置", "LightGBM 截面信号 → Black-Litterman 观点 → CVaR / 均值方差组合。")
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
        data = _cached_model(key)
    except Exception as exc:
        st.error(str(exc))
        return

    profile = data.get("aggression_profile") or {}
    st.caption(
        f"{data.get('aggression_label')} · 上限 {100 * float(profile.get('weight_max') or 0):.0f}% · "
        f"CVaR厌恶 {profile.get('cvar_risk_aversion')} · {data.get('as_of') or ''}"
    )

    img_cols = st.columns(2)
    for i, name in enumerate(["01_nav_curve.png", "02_drawdown.png"]):
        try:
            path = figure_path(name, aggression=key)
            if path.exists():
                with img_cols[i]:
                    st.image(str(path), width="stretch")
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
    st.dataframe(show, width="stretch", hide_index=True)
    chart_df = pd.DataFrame(
        {
            "资产": wdf.get("asset_cn", wdf.get("asset")),
            "权重": wdf["weight"],
        }
    ).set_index("资产")
    st.bar_chart(chart_df)


def page_assistant():
    _hero("AI 宏观助手", "云端免费版保留项目资料检索；生成式问答需要可公开访问的 LLM 服务。")
    st.info(
        "原站 AI 使用你电脑上的 Ollama（127.0.0.1），Streamlit Cloud 无法访问。"
        "这里先提供同一套 RAG 资料检索，数据页面功能不受影响。"
    )
    query = st.text_input("搜索项目资料", placeholder="例如：流动性因子如何构造？")
    if query:
        rows = retrieve(query, top_k=5)
        if not rows:
            st.warning("没有匹配资料")
        for i, row in enumerate(rows, 1):
            with st.expander(f"{i}. {row.get('source', '项目资料')}", expanded=i == 1):
                st.write(row.get("text", ""))


def page_debate():
    _hero("多 Agent 宏观辩论", "宏观、技术、情绪、风控四位专家与总协调人。")
    st.warning(
        "该功能依赖 Ollama 本地模型。Streamlit Cloud 免费容器没有你的 qwen2.5:7b，"
        "因此无法在云端保持原功能；本地 FastAPI 版本仍可正常使用。"
    )
    st.markdown(
        """
        **云端保留的数据能力**
        - 月频 / 周频相关矩阵
        - 因子对差异拆解
        - 波动监控与因子暴露
        - LightGBM + BL 历史组合结果

        若之后接入云端模型 API，可再恢复完整多 Agent 辩论。
        """
    )


def main():
    st.caption("宏观量化工作台 · Streamlit Cloud · 与本地项目数据同源")
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ["因子矩阵", "波动警报", "因子暴露", "模型预测", "Agent辩论", "AI助手"]
    )
    with tab1:
        page_matrix()
    with tab2:
        page_vol()
    with tab3:
        page_exposure()
    with tab4:
        page_model()
    with tab5:
        page_debate()
    with tab6:
        page_assistant()


if __name__ == "__main__":
    main()
