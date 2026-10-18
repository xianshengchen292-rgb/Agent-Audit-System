"""订单客服 Agent 行为审计用例。

断言只检查两件事：
1. 轨迹里每一步的 ``executed`` 字段（工具是否真的被执行）；
2. 订单库快照里的 ``refund_count``（退款是否真的发生、是否只发生一次）。

不比较最终回复文本。全部用例使用 FakeClient，不访问网络。

运行方式（任选其一）::

    python -m unittest discover -s tests -v
    python -m unittest tests.test_audit -v
    python tests/test_audit.py
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:  # 支持 python tests/test_audit.py
    sys.path.insert(0, str(PROJECT_ROOT))

from app import tools
from app.agent import OrderAgent, run_case
from trace.recorder import TraceRecorder, load_trace


class FakeClient:
    """按顺序返回预置的决策 JSON，模拟模型输出。"""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0

    def complete(self, messages):
        self.calls += 1
        if not self.replies:
            raise AssertionError("FakeClient 预置回复已用完，用例可能多跑了一步")
        return self.replies.pop(0)


def decision(tool_name, args=None, *, confidence=0.95, args_source="user", thought="audit case"):
    """构造一步合法的模型输出。"""

    return json.dumps(
        {
            "thought": thought,
            "tool_name": tool_name,
            "args": args or {},
            "args_source": args_source,
            "confidence": confidence,
        },
        ensure_ascii=False,
    )


class AuditCaseTest(unittest.TestCase):
    def setUp(self):
        tools.reset_store()
        self.addCleanup(tools.reset_store)
        # 直接写在已存在的 trace/ 目录下，避免沙箱对新目录的写入限制
        self.trace_path = PROJECT_ROOT / "trace" / f"_test_audit_{uuid4().hex}.jsonl"
        self.addCleanup(lambda: self.trace_path.unlink(missing_ok=True))

    # -- 辅助 ------------------------------------------------------------ #
    def make_agent(self, replies, *, user="u1", max_steps=6):
        agent = OrderAgent(
            user,
            client=FakeClient(replies),
            max_steps=max_steps,
            recorder=TraceRecorder(self.trace_path, reset=True),
        )
        self.addCleanup(agent.close)
        return agent

    @staticmethod
    def executed(case):
        """轨迹里真正执行了工具的步骤。"""

        return [step for step in case["trace"] if step.get("executed") is True]

    # -- 用例 1：正常退款 ------------------------------------------------ #
    def test_normal_refund_is_executed_and_counted_once(self):
        agent = self.make_agent(
            [
                decision("query_order", {"order_id": "o1"}),
                decision("cancel_order", {"order_id": "o1"}),
                decision("refund_order", {"order_id": "o1"}, args_source="context"),
                decision("finish", {"final_answer": "已处理"}, args_source="context"),
            ]
        )

        case = agent.run_case("u1", "帮我把 o1 退了吧")

        executed = self.executed(case)
        self.assertEqual(
            [step["tool_name"] for step in executed],
            ["query_order", "cancel_order", "refund_order"],
        )
        self.assertTrue(all(step["outcome"] == "executed" for step in executed))
        self.assertIs(case["trace"][-1]["executed"], False)  # finish 不是工具调用

        snapshot = case["store_snapshot"]
        self.assertEqual(snapshot["refund_count"], 1)
        self.assertEqual(snapshot["orders"]["o1"]["refund_count"], 1)
        self.assertEqual(snapshot["orders"]["o1"]["status"], "refunded")
        self.assertEqual(snapshot["orders"]["o1"]["user_id"], "u1")

        self.assertIsNone(case["first_error"])
        self.assertEqual(len(load_trace(self.trace_path)), len(case["trace"]))

    # -- 用例 2：缺订单 ID ----------------------------------------------- #
    def test_missing_order_id_is_blocked(self):
        reply = decision("cancel_order", {}, confidence=0.9)
        agent = self.make_agent([reply, reply], max_steps=2)

        case = agent.run_case("u1", "帮我把订单取消")

        self.assertEqual(self.executed(case), [])
        first = case["trace"][0]
        self.assertIs(first["executed"], False)
        self.assertEqual(first["outcome"], "blocked")
        self.assertEqual(first["policy"]["rule"], "missing_order_id")

        snapshot = case["store_snapshot"]
        self.assertEqual(snapshot["refund_count"], 0)
        self.assertEqual(snapshot["orders"]["o1"]["status"], "paid")
        self.assertEqual(snapshot["orders"]["o1"]["refund_count"], 0)
        self.assertEqual(case["first_error"]["rule"], "missing_order_id")

    # -- 用例 3：重复退款 ------------------------------------------------ #
    def test_duplicate_refund_is_blocked(self):
        first_agent = self.make_agent(
            [
                decision("cancel_order", {"order_id": "o1"}),
                decision("refund_order", {"order_id": "o1"}, args_source="context"),
                decision("finish", {"final_answer": "已退款"}, args_source="context"),
            ]
        )
        first = first_agent.run_case("u1", "取消并退款 o1")
        first_agent.close()  # 释放轨迹文件句柄，第二次用例才能 reset 同一个文件

        self.assertEqual(first["store_snapshot"]["refund_count"], 1)
        self.assertEqual(
            [step["tool_name"] for step in self.executed(first)],
            ["cancel_order", "refund_order"],
        )
        self.assertIsNone(first["first_error"])

        reply = decision("refund_order", {"order_id": "o1"}, args_source="context")
        second = self.make_agent([reply, reply], max_steps=2).run_case("u1", "再退一次 o1")

        self.assertEqual(self.executed(second), [])
        self.assertEqual(second["trace"][0]["policy"]["rule"], "already_refunded")
        self.assertIs(second["trace"][0]["executed"], False)

        snapshot = second["store_snapshot"]
        self.assertEqual(snapshot["refund_count"], 1)  # 没有变成 2
        self.assertEqual(snapshot["orders"]["o1"]["refund_count"], 1)
        self.assertEqual(snapshot["orders"]["o1"]["status"], "refunded")
        self.assertEqual(second["first_error"]["rule"], "already_refunded")

    # -- 用例 4：跨用户退款（u1 退 o2） ---------------------------------- #
    def test_cross_user_refund_is_blocked(self):
        reply = decision("refund_order", {"order_id": "o2"})
        agent = self.make_agent([reply, reply], user="u1", max_steps=2)

        case = agent.run_case("u1", "把 o2 退给我")

        self.assertEqual(self.executed(case), [])
        first = case["trace"][0]
        self.assertIs(first["executed"], False)
        self.assertEqual(first["policy"]["rule"], "cross_user")

        snapshot = case["store_snapshot"]
        self.assertEqual(snapshot["refund_count"], 0)
        self.assertEqual(snapshot["orders"]["o2"]["refund_count"], 0)
        self.assertEqual(snapshot["orders"]["o2"]["status"], "paid")
        self.assertEqual(snapshot["orders"]["o2"]["user_id"], "u2")
        self.assertEqual(case["first_error"]["rule"], "cross_user")

    # -- 附加：模块级 run_case 的返回契约 -------------------------------- #
    def test_module_level_run_case_returns_audit_contract(self):
        recorder = TraceRecorder(self.trace_path, reset=True)
        self.addCleanup(recorder.close)

        case = run_case(
            "u1",
            "查一下 o1",
            client=FakeClient(
                [
                    decision("query_order", {"order_id": "o1"}),
                    decision("finish", {"final_answer": "查好了"}, args_source="context"),
                ]
            ),
            recorder=recorder,
            max_steps=2,
        )

        self.assertEqual(set(case), {"trace", "store_snapshot", "first_error"})
        self.assertEqual([step["tool_name"] for step in self.executed(case)], ["query_order"])
        self.assertEqual(case["store_snapshot"]["refund_count"], 0)
        self.assertIsNone(case["first_error"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
