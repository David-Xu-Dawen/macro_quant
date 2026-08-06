#!/usr/bin/env python3
"""
宏观量化 Web 服务：静态页面 + 本地 Ollama LLM 代理。

能力：
1. 注入当前相关矩阵 / 暴露摘要
2. 默认中文模型（Qwen）
3. 项目文档 + JSON 摘要简易检索
4. 工具调用（查 corr / exposure）

启动:
  cd macro_quant/web && pip install -r requirements.txt
  python app.py

浏览器打开: http://127.0.0.1:8765/
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from context import build_live_context
from debate import DebateRequest, DebateResponse, run_debate
from factor_corr import CORR_JSON, compute_corr
from factor_exposure import available_weeks as exposure_available_weeks
from factor_exposure import compute_exposure, load_latest_json as load_exposure_latest
from hf_factor_corr import CORR_JSON as HF_CORR_JSON
from hf_factor_corr import compute_hf_corr
from pair_compare import compare_pair
from rag import format_retrieval, get_corpus, retrieve
from tools import TOOL_DEFS, run_tool
from vol_monitor import compute_vol_monitor

ROOT = Path(__file__).resolve().parent.parent
MP_FIGURES_DIR = ROOT / "model prediction" / "output" / "figures"
OLLAMA_BASE = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")
MAX_TOOL_ROUNDS = 3

SYSTEM_PROMPT = """你是宏观量化投资助手，熟悉增长、通胀、利率、信用、汇率（美元指数）、地缘（沪金+布伦特原油绝对价格拟合）、流动性（M2-社融）等宏观因子。
用户项目会做因子相关性分析、高频因子拟合、波动监控、模型预测与资产配置/风险对冲讨论；
另有 LangGraph 多 Agent 辩论：四位专家（宏观基本面、技术量价、情绪风险、风控对冲）多轮讨论，由总协调人汇总成配置报告。

回答要求：
- 简洁、专业、中文
- 优先依据【当前项目数据摘要】与【检索到的项目资料】作答；摘要已包含月/周相关、因子暴露、波动监控、模型预测，以及辩论角色说明
- 需要精确数字或指定区间时，调用工具 get_corr_matrix / get_hf_corr_matrix / get_factor_exposure / get_vol_monitor / get_model_backtest_summary
- 模型预测中的最新权重/OOF 结果主要用于历史评估，不能当作实时可交易仓位
- 区分历史相关性与因果；不做具体买卖建议；信息不足时明确说明
"""


app = FastAPI(title="macro-quant-web", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(default_factory=list)
    model: str | None = None
    include_system: bool = True
    use_retrieval: bool = True
    use_tools: bool = True
    use_live_context: bool = True


class ChatResponse(BaseModel):
    model: str
    content: str
    retrieved: list[dict] = Field(default_factory=list)
    tool_calls: list[dict] = Field(default_factory=list)


def _latest_user_text(messages: list[dict]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user" and m.get("content"):
            return str(m["content"])
    return ""


def _build_system_prompt(
    user_text: str,
    *,
    use_live_context: bool,
    use_retrieval: bool,
) -> tuple[str, list[dict]]:
    parts = [SYSTEM_PROMPT]
    retrieved: list[dict] = []

    if use_live_context:
        parts.append("\n" + build_live_context())

    if use_retrieval and user_text.strip():
        chunks = retrieve(user_text, top_k=4)
        retrieved = [{"source": c["source"], "text": c["text"][:500]} for c in chunks]
        parts.append("\n【检索到的项目资料】\n" + format_retrieval(chunks))

    return "\n".join(parts), retrieved


async def _ollama_chat(payload: dict) -> dict:
    async with httpx.AsyncClient(timeout=180.0) as client:
        r = await client.post(f"{OLLAMA_BASE}/api/chat", json=payload)
        r.raise_for_status()
        return r.json()


def _normalize_tool_calls(message: dict) -> list[dict]:
    raw = message.get("tool_calls") or []
    out = []
    for item in raw:
        fn = item.get("function") or {}
        name = fn.get("name")
        if not name:
            continue
        args = fn.get("arguments", {})
        out.append({"name": name, "arguments": args})
    return out


@app.get("/api/health")
async def health() -> dict:
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{OLLAMA_BASE}/api/tags")
            r.raise_for_status()
            models = [m.get("name") for m in r.json().get("models", [])]
        return {
            "ok": True,
            "ollama": OLLAMA_BASE,
            "default_model": DEFAULT_MODEL,
            "models": models,
            "corpus_chunks": len(get_corpus()),
        }
    except Exception as exc:
        return {"ok": False, "ollama": OLLAMA_BASE, "error": str(exc)}


@app.get("/api/models")
async def list_models() -> dict:
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(f"{OLLAMA_BASE}/api/tags")
        if r.status_code != 200:
            raise HTTPException(status_code=502, detail="无法连接 Ollama")
        return r.json()


@app.get("/api/retrieve")
async def api_retrieve(q: str = Query(..., min_length=1), top_k: int = Query(4, ge=1, le=10)) -> dict:
    chunks = retrieve(q, top_k=top_k)
    return {"query": q, "chunks": chunks}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    model = req.model or DEFAULT_MODEL
    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    user_text = _latest_user_text(messages)
    retrieved: list[dict] = []
    tool_trace: list[dict] = []

    if req.include_system and not any(m["role"] == "system" for m in messages):
        system_prompt, retrieved = _build_system_prompt(
            user_text,
            use_live_context=req.use_live_context,
            use_retrieval=req.use_retrieval,
        )
        messages = [{"role": "system", "content": system_prompt}, *messages]

    try:
        for _ in range(MAX_TOOL_ROUNDS if req.use_tools else 1):
            payload: dict = {
                "model": model,
                "messages": messages,
                "stream": False,
            }
            if req.use_tools:
                payload["tools"] = TOOL_DEFS

            data = await _ollama_chat(payload)
            message = data.get("message") or {}
            tool_calls = _normalize_tool_calls(message) if req.use_tools else []

            if not tool_calls:
                content = message.get("content") or ""
                return ChatResponse(
                    model=model,
                    content=content,
                    retrieved=retrieved,
                    tool_calls=tool_trace,
                )

            # Keep assistant message (with tool_calls) then append tool results.
            messages.append(
                {
                    "role": "assistant",
                    "content": message.get("content") or "",
                    "tool_calls": message.get("tool_calls") or [],
                }
            )
            for call in tool_calls:
                result = run_tool(call["name"], call.get("arguments"))
                tool_trace.append({"name": call["name"], "arguments": call.get("arguments"), "result": result})
                messages.append(
                    {
                        "role": "tool",
                        "content": result,
                    }
                )

        # Final pass without tools if still looping
        data = await _ollama_chat({"model": model, "messages": messages, "stream": False})
        content = (data.get("message") or {}).get("content") or ""
        return ChatResponse(
            model=model,
            content=content,
            retrieved=retrieved,
            tool_calls=tool_trace,
        )
    except httpx.ConnectError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"无法连接 Ollama ({OLLAMA_BASE})，请先运行 ollama serve",
        ) from exc
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=502, detail=exc.response.text) from exc


@app.post("/api/debate", response_model=DebateResponse)
async def debate(req: DebateRequest) -> DebateResponse:
    """四专家多轮辩论，由总协调人生成大类资产配置报告。"""
    try:
        return await run_debate(req)
    except RuntimeError as exc:
        message = str(exc)
        status = 503 if "Ollama" in message else 500
        raise HTTPException(status_code=status, detail=message) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/macro_factor_corr.json")
async def corr_json() -> dict:
    if not CORR_JSON.exists():
        raise HTTPException(
            status_code=404,
            detail="macro_factor_corr.json not found，请先运行 python plot_macro_factor_corr.py",
        )
    return json.loads(CORR_JSON.read_text(encoding="utf-8"))


@app.get("/api/corr-pair-compare")
async def corr_pair_compare(
    factor_a: str = Query(..., description="因子 A，如 增长因子"),
    factor_b: str = Query(..., description="因子 B，如 汇率因子"),
    start: str | None = Query(None, description="起始月份 YYYY-MM"),
    end: str | None = Query(None, description="结束月份 YYYY-MM"),
) -> dict:
    """点击热力格子后：对比该因子对在月频/周频上的走势与相关差异。"""
    try:
        return compare_pair(factor_a, factor_b, start=start, end=end)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/vol-monitor")
async def vol_monitor(
    window: int = Query(13, ge=4, le=52, description="滚动波动窗口（周）"),
    shock_z: float = Query(2.0, ge=1.0, le=5.0, description="冲击阈值（Z）"),
    end: str | None = Query(None, description="结束周 YYYY-MM-DD，默认最新"),
) -> dict:
    """周频因子波动分位 + 本周冲击 + 暴露压力 + 树模型4周高波动预测。"""
    try:
        return compute_vol_monitor(window=window, shock_z=shock_z, end=end)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/vol-forecast")
async def vol_forecast(
    factor: str = Query("增长因子", description="增长因子/通胀因子/... 或 all"),
    end: str | None = Query(None, description="结束周 YYYY-MM-DD"),
) -> dict:
    """按单因子返回未来4周高波动概率。"""
    try:
        from vol_forecast import predict_all, predict_factor

        if factor == "综合":
            raise ValueError("已移除综合目标，请选择具体因子（如增长因子）或 factor=all")
        if factor in {"all", "*", "全部"}:
            return predict_all(end=end)
        return predict_factor(factor, end=end)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/corr")
async def corr_range(
    start: str | None = Query(None, description="起始月份 YYYY-MM"),
    end: str | None = Query(None, description="结束月份 YYYY-MM"),
) -> dict:
    try:
        if start is None or end is None:
            if not CORR_JSON.exists():
                raise HTTPException(
                    status_code=404,
                    detail="macro_factor_corr.json not found，请先运行 python plot_macro_factor_corr.py",
                )
            return json.loads(CORR_JSON.read_text(encoding="utf-8"))
        return compute_corr(start, end)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/macro_hf_factor_corr.json")
async def hf_corr_json() -> dict:
    if not HF_CORR_JSON.exists():
        raise HTTPException(
            status_code=404,
            detail="macro_hf_factor_corr.json not found，请先运行 python plot_macro_hf_corr.py",
        )
    return json.loads(HF_CORR_JSON.read_text(encoding="utf-8"))


@app.get("/api/hf-corr")
async def hf_corr_range(
    start: str | None = Query(None, description="起始周 YYYY-MM-DD"),
    end: str | None = Query(None, description="结束周 YYYY-MM-DD"),
) -> dict:
    try:
        if start is None or end is None:
            if not HF_CORR_JSON.exists():
                raise HTTPException(
                    status_code=404,
                    detail="macro_hf_factor_corr.json not found，请先运行 python plot_macro_hf_corr.py",
                )
            return json.loads(HF_CORR_JSON.read_text(encoding="utf-8"))
        return compute_hf_corr(start, end)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/factor_exposure_latest.json")
async def factor_exposure_latest() -> dict:
    try:
        return load_exposure_latest()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/factor-exposure/weeks")
async def factor_exposure_weeks(
    rolling_window_weeks: int | None = Query(None, description="滚动窗口周数，默认读取最新 JSON 配置"),
) -> dict:
    try:
        return exposure_available_weeks(rolling_window_weeks)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/factor-exposure")
async def factor_exposure_range(
    end: str = Query(..., description="结束周 YYYY-MM-DD（向前取固定滚动窗口）"),
    rolling_window_weeks: int | None = Query(None, description="滚动窗口周数"),
    sample_length_weeks: int | None = Query(None, description="Bootstrap 连续采样周数"),
    bootstrap: int | None = Query(None, description="Bootstrap 次数"),
    alpha_scale: float | None = Query(None, description="Lasso alpha 缩放系数"),
) -> dict:
    try:
        return compute_exposure(
            end=end,
            rolling_window_weeks=rolling_window_weeks,
            sample_length_weeks=sample_length_weeks,
            bootstrap=bootstrap,
            alpha_scale=alpha_scale,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/model-prediction")
async def model_prediction_summary(
    aggression: str = Query("balanced", description="激进档位: conservative/balanced/aggressive"),
) -> dict:
    """model prediction：滚动回测指标、最新权重、图表列表（可切换激进档位）。"""
    try:
        from model_prediction import summarize

        return summarize(aggression=aggression)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/model-prediction/figures/{name}")
async def model_prediction_figure(
    name: str,
    aggression: str = Query("balanced", description="激进档位"),
) -> FileResponse:
    """提供对应档位 output/.../figures 下的 PNG。"""
    try:
        from model_prediction import figure_path

        path = figure_path(name, aggression=aggression)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"图片不存在: {path.name}")
    return FileResponse(path, media_type="image/png")


@app.get("/")
async def index() -> FileResponse:
    html = ROOT / "macro_factor_corr_interactive.html"
    if not html.exists():
        raise HTTPException(status_code=404, detail="macro_factor_corr_interactive.html not found")
    return FileResponse(html)


@app.get("/macro_factor_monthly.csv")
async def monthly_csv() -> FileResponse:
    path = ROOT / "macro_factor_monthly.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="macro_factor_monthly.csv not found")
    return FileResponse(path)


@app.get("/healthz")
async def healthz() -> dict:
    """云端探针用：不依赖 Ollama，始终返回可用。"""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    # Render / Railway 等平台会注入 PORT；本地默认 8765。
    port = int(os.environ.get("PORT", "8765"))
    host = os.environ.get("HOST", "0.0.0.0")
    uvicorn.run("app:app", host=host, port=port, reload=False)
