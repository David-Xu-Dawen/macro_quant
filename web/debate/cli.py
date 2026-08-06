"""命令行运行多 Agent 辩论。

示例：
  cd web
  python -m debate.cli --topic "当前增长与通胀组合下如何配置大类资产？" --rounds 2
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from context import build_live_context
from rag import format_retrieval, retrieve

from .graph import DEFAULT_MODEL, run_debate
from .schemas import DebateRequest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="宏观多 Agent 辩论")
    parser.add_argument("--topic", required=True, help="辩论议题")
    parser.add_argument("--asset-focus", default=None, help="关注资产/组合")
    parser.add_argument("--rounds", type=int, default=2, choices=(1, 2, 3))
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama 模型")
    parser.add_argument("--no-tools", action="store_true")
    parser.add_argument("--no-rag", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只检查上下文与 RAG，不调用 Ollama",
    )
    parser.add_argument("--output", default=None, help="可选：报告 Markdown 输出路径")
    parser.add_argument("--json", default=None, help="可选：完整记录 JSON 输出路径")
    return parser.parse_args()


async def main_async(args: argparse.Namespace) -> None:
    if args.dry_run:
        chunks = [] if args.no_rag else retrieve(args.topic, top_k=7)
        print(build_live_context())
        print("\n【RAG】")
        print(format_retrieval(chunks) if chunks else "无")
        return

    request = DebateRequest(
        topic=args.topic,
        asset_focus=args.asset_focus,
        rounds=args.rounds,
        model=args.model,
        use_retrieval=not args.no_rag,
        use_tools=not args.no_tools,
    )
    result = await run_debate(request)
    print(result.final_report)
    print(
        f"\nstatus={result.status} | model={result.model} | "
        f"turns={len(result.turns)} | elapsed={result.elapsed_ms / 1000:.1f}s"
    )

    if args.output:
        Path(args.output).write_text(result.final_report, encoding="utf-8")
    if args.json:
        Path(args.json).write_text(
            json.dumps(result.model_dump(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def main() -> None:
    asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    main()
