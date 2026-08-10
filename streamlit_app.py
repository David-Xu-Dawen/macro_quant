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
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
      :root {
        --mq-ink: #17212b;
        --mq-muted: #66717d;
        --mq-red: #9d3535;
        --mq-red-dark: #752626;
        --mq-gold: #c89954;
        --mq-line: rgba(37, 45, 54, .09);
        --mq-surface: rgba(255, 255, 255, .88);
        --mq-shadow: 0 14px 38px rgba(37, 28, 20, .07);
      }
      html { scroll-behavior: smooth; }
      .stApp {
        background:
          radial-gradient(circle at 8% -5%, rgba(157,53,53,.10), transparent 27rem),
          radial-gradient(circle at 92% 8%, rgba(200,153,84,.09), transparent 25rem),
          linear-gradient(145deg, #f8f5f0 0%, #f2f5f3 52%, #f8f4ef 100%);
        color: var(--mq-ink);
      }
      [data-testid="stHeader"] { background: transparent; }
      [data-testid="stDecoration"] { display: none; }
      #MainMenu, footer { visibility: hidden; }
      .block-container {
        max-width: 1380px;
        padding-top: 1.1rem;
        padding-bottom: 4rem;
      }
      h1, h2, h3 { color: var(--mq-ink) !important; letter-spacing: -.01em; }
      p { color: #3f4953; }
      .mq-topbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 18px;
        padding: 9px 4px 16px;
      }
      .mq-brand { display: flex; align-items: center; gap: 11px; }
      .mq-brand-mark {
        display: grid;
        place-items: center;
        width: 38px;
        height: 38px;
        border-radius: 12px;
        color: white;
        font: 800 .78rem/1 ui-sans-serif, system-ui;
        letter-spacing: .08em;
        background: linear-gradient(145deg, var(--mq-red), var(--mq-red-dark));
        box-shadow: 0 8px 20px rgba(117,38,38,.20);
      }
      .mq-brand-name { color: var(--mq-ink); font-size: 1rem; font-weight: 780; }
      .mq-brand-sub { color: var(--mq-muted); font-size: .72rem; margin-top: 1px; }
      .mq-live {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        padding: 6px 10px;
        border: 1px solid rgba(37,45,54,.08);
        border-radius: 999px;
        background: rgba(255,255,255,.64);
        color: #5f6973;
        font-size: .76rem;
      }
      .mq-live::before {
        content: "";
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: #2f9b68;
        box-shadow: 0 0 0 4px rgba(47,155,104,.10);
      }
      div[data-testid="stRadio"] > div[role="radiogroup"] {
        display: flex;
        gap: 6px;
        padding: 5px;
        margin-bottom: 16px;
        border: 1px solid var(--mq-line);
        border-radius: 15px;
        background: rgba(255,255,255,.72);
        box-shadow: 0 7px 24px rgba(37,28,20,.04);
      }
      div[data-testid="stRadio"] > div[role="radiogroup"] label {
        flex: 1;
        justify-content: center;
        min-height: 38px;
        padding: 7px 12px;
        border-radius: 10px;
        transition: background .18s ease, color .18s ease, box-shadow .18s ease;
      }
      div[data-testid="stRadio"] > div[role="radiogroup"] label:has(input:checked) {
        background: linear-gradient(135deg, #a13b3b, #7f2d2d);
        box-shadow: 0 7px 16px rgba(127,45,45,.18);
      }
      div[data-testid="stRadio"] > div[role="radiogroup"] label:has(input:checked) p {
        color: white !important;
        font-weight: 700;
      }
      div[data-testid="stRadio"] input { display: none; }
      div[data-testid="stMetric"] {
        min-height: 108px;
        padding: 16px 17px;
        border: 1px solid var(--mq-line);
        border-radius: 16px;
        background: var(--mq-surface);
        box-shadow: 0 7px 25px rgba(37,28,20,.045);
        transition: transform .18s ease, box-shadow .18s ease;
      }
      div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 28px rgba(37,28,20,.075);
      }
      div[data-testid="stMetricLabel"] p {
        color: var(--mq-muted);
        font-size: .78rem;
        font-weight: 650;
        letter-spacing: .05em;
        text-transform: uppercase;
      }
      div[data-testid="stMetricValue"] {
        color: var(--mq-ink);
        font-size: 1.48rem;
        font-weight: 760;
      }
      .mq-note {
        border: 1px solid rgba(157,53,53,.10);
        border-left: 3px solid var(--mq-red);
        background: rgba(157,53,53,.045);
        border-radius: 4px 12px 12px 4px;
        padding: 11px 14px;
        color: #5d4b4d;
        margin: 0 0 16px;
        font-size: .88rem;
        line-height: 1.6;
      }
      .mq-hero {
        position: relative;
        overflow: hidden;
        padding: 27px 30px 25px;
        margin: 0 0 22px;
        border: 1px solid var(--mq-line);
        border-radius: 21px;
        background:
          linear-gradient(112deg, rgba(255,255,255,.97), rgba(255,249,243,.88));
        box-shadow: var(--mq-shadow);
      }
      .mq-hero::after {
        content: "";
        position: absolute;
        width: 210px;
        height: 210px;
        right: -65px;
        top: -95px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(157,53,53,.12), transparent 68%);
      }
      .mq-eyebrow {
        margin-bottom: 7px;
        color: var(--mq-red);
        font-size: .71rem;
        font-weight: 800;
        letter-spacing: .14em;
        text-transform: uppercase;
      }
      .mq-hero h1 { margin: 0; font-size: clamp(1.55rem, 2.2vw, 2.05rem); font-weight: 790; }
      .mq-hero p { max-width: 800px; margin: 8px 0 0; color: var(--mq-muted); line-height: 1.7; }
      .mq-section-title {
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 28px 0 13px;
        color: var(--mq-ink);
        font-weight: 760;
        font-size: 1.05rem;
      }
      .mq-section-title::before {
        content: "";
        width: 4px;
        height: 19px;
        border-radius: 9px;
        background: linear-gradient(var(--mq-red), var(--mq-gold));
      }
      .mq-chip-row { display:flex; flex-wrap:wrap; gap:8px; margin:10px 0 12px; }
      .mq-chip {
        display:inline-block;
        padding:7px 11px;
        border:1px solid rgba(157,53,53,.13);
        border-radius:999px;
        background:rgba(255,250,247,.92);
        color:#694043;
        font-size:.82rem;
      }
      div[data-testid="stDataFrame"], div[data-testid="stPlotlyChart"],
      div[data-testid="stImage"] {
        border: 1px solid var(--mq-line);
        border-radius: 17px;
        overflow: hidden;
        background: rgba(255,255,255,.67);
        box-shadow: 0 8px 25px rgba(37,28,20,.045);
      }
      div[data-testid="stExpander"] {
        border: 1px solid var(--mq-line);
        border-radius: 15px;
        background: rgba(255,255,255,.62);
        overflow: hidden;
      }
      div[data-testid="stExpander"] summary { font-weight: 680; }
      div[data-baseweb="select"] > div,
      div[data-baseweb="input"] > div,
      div[data-testid="stTextInput"] input {
        border-color: var(--mq-line) !important;
        border-radius: 11px !important;
        background: rgba(255,255,255,.82) !important;
      }
      .stButton > button {
        min-height: 42px;
        padding: 0 18px;
        border-radius: 11px;
        border-color: rgba(157,53,53,.16);
        font-weight: 700;
        transition: transform .16s ease, box-shadow .16s ease;
      }
      .stButton > button[kind="primary"] {
        border: none;
        background: linear-gradient(135deg, #a13b3b, #7f2d2d);
        box-shadow: 0 8px 18px rgba(127,45,45,.18);
      }
      .stButton > button:hover { transform: translateY(-1px); }
      .mq-footer {
        margin-top: 42px;
        padding: 18px 4px 4px;
        border-top: 1px solid var(--mq-line);
        color: #7a838b;
        font-size: .75rem;
        text-align: center;
      }
      @media (max-width: 760px) {
        .block-container { padding: .7rem .8rem 3rem; }
        .mq-topbar { padding-bottom: 10px; }
        .mq-brand-sub, .mq-live { display: none; }
        .mq-hero { padding: 21px 19px; border-radius: 17px; }
        .mq-hero p { font-size: .88rem; }
        div[data-testid="stRadio"] > div[role="radiogroup"] {
          overflow-x: auto;
          justify-content: flex-start;
        }
        div[data-testid="stRadio"] > div[role="radiogroup"] label {
          flex: 0 0 auto;
          white-space: nowrap;
        }
        div[data-testid="stMetric"] { min-height: 94px; }
      }
    </style>
    """,
    unsafe_allow_html=True,
)


def _app_header() -> None:
    st.markdown(
        """
        <div class="mq-topbar">
          <div class="mq-brand">
            <div class="mq-brand-mark">MQ</div>
            <div>
              <div class="mq-brand-name">宏观量化工作台</div>
              <div class="mq-brand-sub">Macro Intelligence & Asset Allocation</div>
            </div>
          </div>
          <div class="mq-live">数据工作台在线</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _hero(title: str, subtitle: str, eyebrow: str = "MACRO QUANT") -> None:
    st.markdown(
        f'<div class="mq-hero"><div class="mq-eyebrow">{eyebrow}</div>'
        f'<h1>{title}</h1><p>{subtitle}</p></div>',
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
    cell_x: list[str] = []
    cell_y: list[str] = []
    cell_corr: list[float] = []
    cell_text: list[str] = []
    for row_name in labels:
        for col_name in labels:
            value = float(df.loc[row_name, col_name])
            cell_x.append(col_name)
            cell_y.append(row_name)
            cell_corr.append(value)
            cell_text.append(f"{value:+.2f}")

    # Streamlit 不返回 Heatmap 单元格点击事件，因此用方形 Scatter
    # 直接构成热力图。每个可见格子本身就是可选择点，无需透明覆盖层。
    fig = go.Figure(
        go.Scatter(
            x=cell_x,
            y=cell_y,
            mode="markers+text",
            customdata=np.asarray(cell_corr)[:, None],
            text=cell_text,
            textposition="middle center",
            textfont={"size": 13, "color": "#17202a"},
            marker={
                "symbol": "square",
                "size": 62,
                "color": cell_corr,
                "cmin": -1,
                "cmax": 1,
                # 柔和的发散色，保证深色文字在高相关格子上仍清晰可读。
                "colorscale": [
                    [0.00, "#79a5c7"],
                    [0.25, "#b7d0e1"],
                    [0.50, "#f7f4ef"],
                    [0.75, "#e8b9b3"],
                    [1.00, "#d7837f"],
                ],
                "colorbar": {"title": "ρ", "thickness": 12},
                "line": {"color": "rgba(255,255,255,.72)", "width": 2},
            },
            selected={"marker": {"opacity": 1}},
            unselected={"marker": {"opacity": 1}},
            hovertemplate="%{y} × %{x}<br>ρ=%{customdata[0]:.3f}<extra>点击拆解</extra>",
            showlegend=False,
            name="相关系数",
        )
    )
    fig.update_xaxes(
        side="top",
        tickangle=-28,
        categoryorder="array",
        categoryarray=labels,
        fixedrange=True,
    )
    fig.update_yaxes(
        autorange="reversed",
        categoryorder="array",
        categoryarray=labels,
        fixedrange=True,
    )
    fig.update_layout(
        title={"text": title, "x": 0.02, "xanchor": "left"},
        height=520,
        margin=dict(l=12, r=12, t=72, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="PingFang SC, Microsoft YaHei, sans-serif"),
        clickmode="event+select",
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


def page_matrix():
    _hero(
        "宏观因子矩阵与风险警报",
        "在一个页面查看周频风险警报、月频/周频相关矩阵与因子对拆解。点击任意非对角格即可生成下方图表。",
        "RISK RADAR · CORRELATION",
    )
    page_vol(embedded=True)
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

    with st.expander("手动选择因子对", expanded=False):
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


def page_vol(*, embedded: bool = False):
    if embedded:
        st.markdown('<div class="mq-section-title">周频因子风险警报</div>', unsafe_allow_html=True)
    else:
        _hero("周频波动警报", "13 周波动分位、本周 2σ 冲击、资产暴露压力与未来 4 周高波动概率。")
    try:
        mon = _cached_vol()
    except Exception as exc:
        st.error(f"波动监控加载失败：{exc}")
        return

    factors = pd.DataFrame(mon.get("factors") or [])
    shocks = pd.DataFrame(mon.get("shocks") or [])
    pressure = pd.DataFrame(mon.get("asset_pressure") or [])
    high_count = 0
    if not factors.empty and "vol_percentile" in factors:
        high_count = int((pd.to_numeric(factors["vol_percentile"], errors="coerce") >= 75).sum())

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("市场状态", mon.get("status", "—"))
    m2.metric("高波动因子", high_count, help="13 周波动分位处于 75% 以上")
    m3.metric("本周 2σ 冲击", len(shocks))
    m4.metric("数据截至", mon.get("as_of", "—"))
    st.caption(mon.get("status_note", ""))

    with st.expander("查看警报明细与资产压力", expanded=False):
        left, right = st.columns([1.18, 0.82])
        with left:
            st.markdown("**因子波动风险**")
            if factors.empty:
                st.info("暂无因子波动数据")
            else:
                show = factors[
                    ["factor", "vol_percentile", "vol_level", "week_change", "shock_z", "is_shock"]
                ].copy()
                show.columns = ["因子", "波动分位", "水平", "本周变化", "冲击 Z", "是否冲击"]
                st.dataframe(show, width="stretch", hide_index=True)
        with right:
            st.markdown("**本周冲击**")
            if shocks.empty:
                st.success("本周无明显冲击")
            else:
                st.dataframe(shocks, width="stretch", hide_index=True)
            st.markdown("**资产暴露压力 Top 10**")
            if pressure.empty:
                st.info("暂无暴露压力")
            else:
                st.dataframe(pressure.head(10), width="stretch", hide_index=True)

    pred_left, pred_right = st.columns([0.34, 0.66], vertical_alignment="bottom")
    with pred_left:
        factor = st.selectbox(
            "未来 4 周波动预测",
            VOL_FACTORS,
            index=VOL_FACTORS.index(DEFAULT_FACTOR) if DEFAULT_FACTOR in VOL_FACTORS else 0,
        )
    try:
        pred = predict_factor(factor)
        with pred_right:
            p1, p2, p3 = st.columns(3)
            p1.metric("高波动概率", f"{100 * float(pred.get('prob_high_vol', 0)):.1f}%")
            p2.metric("风险等级", pred.get("level") or "—")
            p3.metric("预测高波动", "是" if pred.get("pred_high_vol") else "否")
        st.caption(pred.get("interpretation") or pred.get("note") or "")
    except Exception as exc:
        st.warning(f"波动预测暂不可用：{exc}")


def page_exposure():
    _hero(
        "资产宏观因子暴露矩阵",
        "资产周度收益对宏观高频因子变化的 Bootstrap + Lasso 暴露。",
        "ASSET EXPOSURE",
    )
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
    _hero(
        "模型预测 · 大类配置",
        "LightGBM 截面信号 → Black-Litterman 观点 → CVaR / 均值方差组合。",
        "MODEL PORTFOLIO",
    )
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
    _hero(
        "AI 宏观助手",
        "云端免费版保留项目资料检索；生成式问答需要可公开访问的 LLM 服务。",
        "KNOWLEDGE SEARCH",
    )
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
    _hero(
        "多 Agent 宏观辩论",
        "宏观、技术、情绪、风控四位专家与总协调人。",
        "MULTI-AGENT DEBATE",
    )
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
    _app_header()
    pages = {
        "◫  因子矩阵与警报": page_matrix,
        "◇  因子暴露": page_exposure,
        "↗  模型预测": page_model,
        "◎  Agent 辩论": page_debate,
        "⌕  AI 助手": page_assistant,
    }
    selected = st.radio(
        "页面导航",
        list(pages),
        horizontal=True,
        label_visibility="collapsed",
        key="main_navigation",
    )
    pages[selected]()
    st.markdown(
        '<div class="mq-footer">MACRO QUANT WORKSPACE · 数据与本地研究项目同源</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
