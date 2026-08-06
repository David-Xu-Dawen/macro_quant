"""多 Agent 图的离线测试，不依赖 Ollama。"""

from __future__ import annotations

import unittest

from debate.graph import normalize_expert_content, run_debate
from debate.schemas import DebateRequest


class FakeChatClient:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def chat(self, messages, *, model, tools=None):
        self.calls.append(
            {
                "messages": messages,
                "model": model,
                "tools": tools,
            }
        )
        call_no = len(self.calls)
        is_moderator = "请生成最终《大类资产配置策略报告》" in messages[-1]["content"]
        if is_moderator:
            sections = [
                "一、执行摘要",
                "二、数据时点与口径",
                "三、四类专家核心证据",
                "四、共识与主要分歧",
                "五、基准/乐观/悲观情景",
                "六、配置框架与风险预算",
                "七、对冲与止损触发条件",
                "八、模型可信度与数据缺口",
                "九、下次更新清单",
            ]
            content = "# 大类资产配置策略报告\n\n" + "\n\n".join(
                f"## {section}\n离线测试。" for section in sections
            )
        else:
            content = f"专家离线测试发言 {call_no}。"
        return {"message": {"role": "assistant", "content": content}}


class DebateGraphTest(unittest.IsolatedAsyncioTestCase):
    async def test_one_round_has_four_turns_and_report(self) -> None:
        fake = FakeChatClient()
        response = await run_debate(
            DebateRequest(
                topic="测试当前宏观环境下的大类资产配置框架",
                rounds=1,
                use_retrieval=False,
                use_tools=False,
                use_live_context=False,
            ),
            client=fake,
        )

        self.assertEqual(response.status, "completed")
        self.assertEqual(len(response.turns), 4)
        self.assertEqual(
            [turn["expert"] for turn in response.turns],
            ["macro", "technical", "sentiment", "risk"],
        )
        self.assertIn("大类资产配置策略报告", response.final_report)
        for turn in response.turns:
            self.assertEqual(turn["content"].count("### "), 5)
            self.assertIn("### 数据与证据", turn["content"])
            self.assertIn("### 风险与反例", turn["content"])
        self.assertEqual(len(fake.calls), 5)

    async def test_second_round_sees_previous_history(self) -> None:
        fake = FakeChatClient()
        response = await run_debate(
            DebateRequest(
                topic="测试信用收缩和通胀冲击的组合风险",
                rounds=2,
                use_retrieval=False,
                use_tools=False,
                use_live_context=False,
            ),
            client=fake,
        )

        self.assertEqual(len(response.turns), 8)
        self.assertEqual(len(fake.calls), 9)
        second_round_macro_prompt = fake.calls[4]["messages"][0]["content"]
        self.assertIn("第 2/2 轮", second_round_macro_prompt)
        self.assertIn("专家离线测试发言 1", second_round_macro_prompt)
        self.assertIn("周频相关性矩阵来自对月频宏观数据的尽力拟合", second_round_macro_prompt)
        self.assertIn("甚至方向相反", second_round_macro_prompt)

    async def test_expert_format_variants_are_normalized(self) -> None:
        content = normalize_expert_content(
            "**核心判断**：方向偏谨慎。\n"
            "## 证据\n周频相关仅作参考。\n"
            "3. 风险/反例\n频率差异可能反转结论。",
            current_round=2,
        )

        self.assertEqual(content.count("### "), 5)
        self.assertIn("### 核心判断\n方向偏谨慎。", content)
        self.assertIn("### 数据与证据\n周频相关仅作参考。", content)
        self.assertIn("### 对其他专家的回应\n本轮未明确回应其他专家观点。", content)
        self.assertIn("### 风险与反例\n频率差异可能反转结论。", content)


if __name__ == "__main__":
    unittest.main()
