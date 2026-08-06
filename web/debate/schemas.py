"""多 Agent 辩论的 API 与运行时数据结构。"""

from __future__ import annotations

import operator
from typing import Annotated, Literal

from pydantic import BaseModel, Field
from typing_extensions import TypedDict


ExpertName = Literal["macro", "technical", "sentiment", "risk"]

EXPERT_ORDER: tuple[ExpertName, ...] = (
    "macro",
    "technical",
    "sentiment",
    "risk",
)

EXPERT_LABELS: dict[ExpertName, str] = {
    "macro": "宏观基本面专家",
    "technical": "技术量价专家",
    "sentiment": "情绪风险专家",
    "risk": "风控对冲专家",
}


class ToolRecord(TypedDict):
    name: str
    arguments: dict
    result: str


class TurnRecord(TypedDict):
    round: int
    expert: ExpertName
    expert_label: str
    content: str
    sources: list[str]
    tool_calls: list[ToolRecord]
    error: str | None


class ErrorRecord(TypedDict):
    node: str
    phase: str
    message: str
    recoverable: bool


class DebateState(TypedDict):
    topic: str
    asset_focus: str | None
    model: str
    max_rounds: int
    use_retrieval: bool
    use_tools: bool
    use_live_context: bool
    current_round: int
    live_context: str
    retrieved: list[dict]
    turns: Annotated[list[TurnRecord], operator.add]
    errors: Annotated[list[ErrorRecord], operator.add]
    final_report: str
    status: str


class DebateRequest(BaseModel):
    topic: str = Field(
        ...,
        min_length=4,
        max_length=2000,
        description="本次辩论要回答的宏观配置问题",
    )
    asset_focus: str | None = Field(
        default=None,
        max_length=300,
        description="可选：关注的资产或组合",
    )
    rounds: int = Field(default=2, ge=1, le=3, description="专家辩论轮数")
    model: str | None = Field(default=None, description="Ollama 模型名")
    use_retrieval: bool = True
    use_tools: bool = True
    use_live_context: bool = True


class DebateResponse(BaseModel):
    status: Literal["completed", "partial", "failed"]
    topic: str
    asset_focus: str | None = None
    model: str
    rounds: int
    turns: list[TurnRecord] = Field(default_factory=list)
    retrieved: list[dict] = Field(default_factory=list)
    errors: list[ErrorRecord] = Field(default_factory=list)
    final_report: str = ""
    elapsed_ms: int
