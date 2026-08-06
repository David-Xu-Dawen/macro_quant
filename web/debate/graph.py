"""LangGraph 多专家顺序辩论工作流。"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from collections.abc import Callable
from typing import Any, Protocol

import httpx
from langgraph.graph import END, START, StateGraph

from context import build_live_context
from rag import format_retrieval, retrieve

from .prompts import (
    ROLE_QUERY_HINTS,
    build_expert_messages,
    build_moderator_messages,
)
from .schemas import (
    EXPERT_LABELS,
    EXPERT_ORDER,
    DebateRequest,
    DebateResponse,
    DebateState,
    ErrorRecord,
    ExpertName,
    ToolRecord,
    TurnRecord,
)
from .tools import DEBATE_TOOL_DEFS, run_debate_tool


DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")
DEFAULT_OLLAMA_BASE = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
MAX_TOOL_ROUNDS = 2
LLM_TIMEOUT_SECONDS = 180.0

EXPERT_SECTION_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("核心判断", ("核心判断", "主要判断", "观点", "结论")),
    ("数据与证据", ("数据与证据", "证据", "依据", "证据链")),
    (
        "对其他专家的回应",
        ("对其他专家的回应", "对其他专家观点的回应", "专家回应", "回应"),
    ),
    ("风险与反例", ("风险与反例", "风险/反例", "风险和反例", "反例与风险")),
    ("待验证事项", ("待验证事项", "待验证", "验证事项", "后续验证")),
)


def _expert_heading(line: str) -> tuple[str, str] | None:
    cleaned = line.strip()
    cleaned = re.sub(r"^#{1,6}\s*", "", cleaned)
    cleaned = re.sub(r"^\s*(?:\d+[.、]|[一二三四五][、.])\s*", "", cleaned)
    cleaned = cleaned.replace("**", "").replace("__", "").strip("*_ ")
    for canonical, aliases in EXPERT_SECTION_ALIASES:
        for alias in aliases:
            if cleaned == alias:
                return canonical, ""
            if cleaned.startswith(alias):
                remainder = cleaned[len(alias) :]
                if remainder.startswith(("：", ":")):
                    return canonical, remainder[1:].strip()
    return None


def normalize_expert_content(content: str, current_round: int) -> str:
    """把小模型常见的标题变体修正为稳定的五段式记录。"""
    text = str(content or "").strip()
    text = re.sub(r"^```(?:markdown|md)?\s*\n?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n?```\s*$", "", text).strip()

    sections = {name: [] for name, _ in EXPERT_SECTION_ALIASES}
    preface: list[str] = []
    active: str | None = None
    for line in text.splitlines():
        heading = _expert_heading(line)
        if heading:
            active, inline_content = heading
            if inline_content:
                sections[active].append(inline_content)
            continue
        if active is None:
            preface.append(line)
        else:
            sections[active].append(line)

    preface_text = "\n".join(preface).strip()
    if preface_text:
        existing = "\n".join(sections["核心判断"]).strip()
        sections["核心判断"] = [
            value for value in (preface_text, existing) if value
        ]

    defaults = {
        "核心判断": "本轮未形成明确判断。",
        "数据与证据": "本轮未提供可核验的数据证据，结论置信度应降低。",
        "对其他专家的回应": (
            "本轮暂无可回应的先前观点。"
            if current_round == 1
            else "本轮未明确回应其他专家观点。"
        ),
        "风险与反例": "本轮未明确列出风险与反例。",
        "待验证事项": "本轮未明确列出待验证事项。",
    }
    blocks = []
    for name, _ in EXPERT_SECTION_ALIASES:
        body = "\n".join(sections[name]).strip() or defaults[name]
        blocks.append(f"### {name}\n{body}")
    return "\n\n".join(blocks)


class ChatClient(Protocol):
    async def chat(
        self,
        messages: list[dict],
        *,
        model: str,
        tools: list[dict] | None = None,
    ) -> dict: ...


class OllamaChatClient:
    """保持与现有 web/app.py 一致，直接调用 Ollama `/api/chat`。"""

    def __init__(
        self,
        base_url: str = DEFAULT_OLLAMA_BASE,
        timeout: float = LLM_TIMEOUT_SECONDS,
        retries: int = 1,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.retries = max(0, retries)

    async def chat(
        self,
        messages: list[dict],
        *,
        model: str,
        tools: list[dict] | None = None,
    ) -> dict:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.25},
        }
        if tools:
            payload["tools"] = tools

        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(
                        f"{self.base_url}/api/chat",
                        json=payload,
                    )
                    response.raise_for_status()
                    return response.json()
            except (httpx.HTTPError, ValueError) as exc:
                last_error = exc
                if attempt < self.retries:
                    await asyncio.sleep(1.0 + attempt * 2.0)
        raise RuntimeError(f"Ollama 调用失败: {last_error}") from last_error


def _normalize_tool_calls(message: dict) -> list[dict]:
    calls: list[dict] = []
    for raw in message.get("tool_calls") or []:
        fn = raw.get("function") or {}
        name = fn.get("name")
        if name:
            calls.append(
                {
                    "name": str(name),
                    "arguments": fn.get("arguments") or {},
                }
            )
    return calls


def _safe_args(arguments: dict | str | None) -> dict:
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        try:
            value = json.loads(arguments)
            return value if isinstance(value, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


async def _chat_with_tools(
    client: ChatClient,
    messages: list[dict],
    *,
    model: str,
    use_tools: bool,
    tool_runner: Callable[[str, dict | str | None], str],
) -> tuple[str, list[ToolRecord]]:
    working = list(messages)
    trace: list[ToolRecord] = []

    for _ in range(MAX_TOOL_ROUNDS if use_tools else 1):
        data = await client.chat(
            working,
            model=model,
            tools=DEBATE_TOOL_DEFS if use_tools else None,
        )
        message = data.get("message") or {}
        calls = _normalize_tool_calls(message)
        if not calls:
            return str(message.get("content") or "").strip(), trace

        working.append(message)
        for call in calls:
            result = tool_runner(call["name"], call["arguments"])
            trace.append(
                {
                    "name": call["name"],
                    "arguments": _safe_args(call["arguments"]),
                    "result": result[:6000],
                }
            )
            working.append(
                {
                    "role": "tool",
                    "tool_name": call["name"],
                    "content": result,
                }
            )

    # 工具轮数耗尽，强制模型基于已有结果给文本结论。
    data = await client.chat(working, model=model, tools=None)
    message = data.get("message") or {}
    return str(message.get("content") or "").strip(), trace


def _retrieval_for_expert(
    state: DebateState,
    expert: ExpertName,
) -> tuple[list[dict], str]:
    if not state["use_retrieval"]:
        return [], ""
    query = " ".join(
        [
            state["topic"],
            state.get("asset_focus") or "",
            ROLE_QUERY_HINTS[expert],
        ]
    )
    chunks = retrieve(query, top_k=5)
    return chunks, format_retrieval(chunks)


def build_debate_graph(
    *,
    client: ChatClient | None = None,
    tool_runner: Callable[[str, dict | str | None], str] = run_debate_tool,
):
    """构建并编译 LangGraph；依赖可注入，便于 Fake LLM 测试。"""

    llm = client or OllamaChatClient()

    async def prepare(state: DebateState) -> dict:
        errors: list[ErrorRecord] = []
        live_context = ""
        retrieved: list[dict] = []

        if state["use_live_context"]:
            try:
                live_context = build_live_context()
            except Exception as exc:
                errors.append(
                    {
                        "node": "prepare",
                        "phase": "context",
                        "message": str(exc),
                        "recoverable": True,
                    }
                )

        if state["use_retrieval"]:
            try:
                query = f"{state['topic']} {state.get('asset_focus') or ''}"
                retrieved = retrieve(query, top_k=7)
            except Exception as exc:
                errors.append(
                    {
                        "node": "prepare",
                        "phase": "retrieve",
                        "message": str(exc),
                        "recoverable": True,
                    }
                )

        return {
            "live_context": live_context,
            "retrieved": retrieved,
            "current_round": 1,
            "status": "running",
            "errors": errors,
        }

    def make_expert_node(expert: ExpertName):
        async def expert_node(state: DebateState) -> dict:
            sources: list[str] = []
            tool_trace: list[ToolRecord] = []
            error: str | None = None
            try:
                chunks, retrieval_text = _retrieval_for_expert(state, expert)
                sources = [str(c.get("source", "")) for c in chunks]
                messages = build_expert_messages(
                    expert=expert,
                    topic=state["topic"],
                    asset_focus=state.get("asset_focus"),
                    current_round=state["current_round"],
                    max_rounds=state["max_rounds"],
                    live_context=state["live_context"],
                    retrieval_text=retrieval_text,
                    turns=state["turns"],
                )
                content, tool_trace = await _chat_with_tools(
                    llm,
                    messages,
                    model=state["model"],
                    use_tools=state["use_tools"],
                    tool_runner=tool_runner,
                )
                if not content:
                    content = "本轮未生成有效文本，请在最终报告中标记该专家意见缺失。"
                    error = "Ollama 返回空内容"
            except Exception as exc:
                content = "本轮专家调用失败；协调人应基于其他专家意见并明确数据缺口。"
                error = str(exc)

            content = normalize_expert_content(content, state["current_round"])
            turn: TurnRecord = {
                "round": state["current_round"],
                "expert": expert,
                "expert_label": EXPERT_LABELS[expert],
                "content": content,
                "sources": sources,
                "tool_calls": tool_trace,
                "error": error,
            }
            patch: dict[str, Any] = {"turns": [turn]}
            if error:
                patch["errors"] = [
                    {
                        "node": expert,
                        "phase": "llm",
                        "message": error,
                        "recoverable": True,
                    }
                ]
            return patch

        return expert_node

    def route_after_risk(state: DebateState) -> str:
        return (
            "advance_round"
            if state["current_round"] < state["max_rounds"]
            else "moderator"
        )

    async def advance_round(state: DebateState) -> dict:
        return {"current_round": state["current_round"] + 1}

    async def moderator(state: DebateState) -> dict:
        try:
            retrieval_text = format_retrieval(state["retrieved"])
            messages = build_moderator_messages(
                topic=state["topic"],
                asset_focus=state.get("asset_focus"),
                live_context=state["live_context"],
                retrieval_text=retrieval_text,
                turns=state["turns"],
            )
            data = await llm.chat(messages, model=state["model"], tools=None)
            report = str((data.get("message") or {}).get("content") or "").strip()
            required_sections = [
                "执行摘要",
                "数据时点与口径",
                "四类专家核心证据",
                "共识与主要分歧",
                "基准/乐观/悲观情景",
                "配置框架与风险预算",
                "对冲与止损触发条件",
                "模型可信度与数据缺口",
                "下次更新清单",
            ]
            missing = [section for section in required_sections if section not in report]
            if report and missing:
                repair_messages = [
                    *messages,
                    {"role": "assistant", "content": report},
                    {
                        "role": "user",
                        "content": (
                            "上版报告缺少以下必需章节："
                            + "、".join(missing)
                            + "。请完整重写，严格使用指定九章结构；"
                            "删除输入资料没有支持的仓位比例、止损比例和预测数字。"
                        ),
                    },
                ]
                repaired = await llm.chat(
                    repair_messages,
                    model=state["model"],
                    tools=None,
                )
                repaired_text = str(
                    (repaired.get("message") or {}).get("content") or ""
                ).strip()
                if repaired_text:
                    report = repaired_text
            if not report:
                raise RuntimeError("协调人返回空报告")
            status = "partial" if state["errors"] else "completed"
            return {"final_report": report, "status": status}
        except Exception as exc:
            return {
                "final_report": "报告生成失败，请查看 errors 与专家发言记录。",
                "status": "failed",
                "errors": [
                    {
                        "node": "moderator",
                        "phase": "llm",
                        "message": str(exc),
                        "recoverable": False,
                    }
                ],
            }

    graph = StateGraph(DebateState)
    graph.add_node("prepare", prepare)
    for expert in EXPERT_ORDER:
        graph.add_node(expert, make_expert_node(expert))
    graph.add_node("advance_round", advance_round)
    graph.add_node("moderator", moderator)

    graph.add_edge(START, "prepare")
    graph.add_edge("prepare", "macro")
    graph.add_edge("macro", "technical")
    graph.add_edge("technical", "sentiment")
    graph.add_edge("sentiment", "risk")
    graph.add_conditional_edges(
        "risk",
        route_after_risk,
        {
            "advance_round": "advance_round",
            "moderator": "moderator",
        },
    )
    graph.add_edge("advance_round", "macro")
    graph.add_edge("moderator", END)
    return graph.compile()


async def run_debate(
    request: DebateRequest,
    *,
    client: ChatClient | None = None,
    tool_runner: Callable[[str, dict | str | None], str] = run_debate_tool,
) -> DebateResponse:
    started = time.perf_counter()
    initial: DebateState = {
        "topic": request.topic,
        "asset_focus": request.asset_focus,
        "model": request.model or DEFAULT_MODEL,
        "max_rounds": request.rounds,
        "use_retrieval": request.use_retrieval,
        "use_tools": request.use_tools,
        "use_live_context": request.use_live_context,
        "current_round": 1,
        "live_context": "",
        "retrieved": [],
        "turns": [],
        "errors": [],
        "final_report": "",
        "status": "running",
    }
    graph = build_debate_graph(client=client, tool_runner=tool_runner)
    result = await graph.ainvoke(initial)
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    return DebateResponse(
        status=result["status"],
        topic=result["topic"],
        asset_focus=result.get("asset_focus"),
        model=result["model"],
        rounds=request.rounds,
        turns=result["turns"],
        retrieved=result["retrieved"],
        errors=result["errors"],
        final_report=result["final_report"],
        elapsed_ms=elapsed_ms,
    )
