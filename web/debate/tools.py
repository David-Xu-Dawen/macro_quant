"""辩论 Agent 可调用的数据工具注册表。"""

from __future__ import annotations

import json
from typing import Any

from rag import retrieve
from tools import TOOL_DEFS, run_tool

from .prompts import DATA_QUALITY_NOTICE


EXTRA_TOOL_DEFS = [
    {
        "type": "function",
        "function": {
            "name": "retrieve_project_docs",
            "description": "检索项目 README、因子相关、暴露与回归摘要等本地资料。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "检索问题"},
                    "top_k": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 8,
                        "default": 4,
                    },
                },
                "required": ["query"],
            },
        },
    },
]


def _debate_base_tool_defs() -> list[dict]:
    definitions: list[dict] = []
    for item in TOOL_DEFS:
        copied = {**item, "function": dict(item["function"])}
        if copied["function"].get("name") == "get_hf_corr_matrix":
            copied["function"]["description"] = (
                f"{copied['function'].get('description', '')} "
                "该矩阵是对月频数据的尽力周频拟合，可能不准确或不完整；"
                "同区间月频与周频相关性可能不同甚至相反，必须谨慎解释。"
            )
        definitions.append(copied)
    return definitions


DEBATE_TOOL_DEFS = [*_debate_base_tool_defs(), *EXTRA_TOOL_DEFS]


def _parse_args(arguments: dict[str, Any] | str | None) -> dict[str, Any]:
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str) and arguments.strip():
        try:
            parsed = json.loads(arguments)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def run_debate_tool(
    name: str,
    arguments: dict[str, Any] | str | None,
) -> str:
    args = _parse_args(arguments)
    try:
        if name in {d["function"]["name"] for d in TOOL_DEFS}:
            result = run_tool(name, args)
            if name == "get_hf_corr_matrix":
                try:
                    payload = json.loads(result)
                    if isinstance(payload, dict):
                        payload["data_quality_warning"] = DATA_QUALITY_NOTICE
                        return json.dumps(payload, ensure_ascii=False)
                except json.JSONDecodeError:
                    pass
                return f"{result}\n\n{DATA_QUALITY_NOTICE}"
            return result

        if name == "retrieve_project_docs":
            query = str(args.get("query") or "").strip()
            if not query:
                return json.dumps({"error": "query 不能为空"}, ensure_ascii=False)
            top_k = max(1, min(int(args.get("top_k", 4)), 8))
            chunks = retrieve(query, top_k=top_k)
            return json.dumps({"query": query, "chunks": chunks}, ensure_ascii=False)

        return json.dumps({"error": f"未知工具: {name}"}, ensure_ascii=False)
    except Exception as exc:
        return json.dumps({"error": str(exc)}, ensure_ascii=False)
