"""FastAPI 审计接口测试。

使用 TestClient + 伪 run_case，不调用真实模型；未安装 FastAPI 时整体跳过。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from fastapi.testclient import TestClient

    import main as api_main

    FASTAPI_AVAILABLE = True
except Exception:  # pragma: no cover - 未安装 API 依赖时
    FASTAPI_AVAILABLE = False


def _blocked_case() -> dict:
    trace = [
        {
            "step": 1,
            "session_id": "s1",
            "tool_name": "refund_order",
            "args": {"order_id": "o2", "confidence": 0.95},
            "args_source": "user",
            "confidence": 0.95,
            "policy": {
                "allowed": False,
                "rule": "cross_user",
                "reason": "订单 o2 不属于当前用户 u1，拒绝操作",
            },
            "executed": False,
            "outcome": "blocked",
        },
        {
            "step": 2,
            "session_id": "s1",
            "tool_name": "finish",
            "args": {"final_answer": "抱歉，该订单不属于您，我无法操作。"},
            "args_source": "context",
            "confidence": 0.95,
            "policy": {"allowed": True, "rule": None, "reason": "finish"},
            "executed": False,
            "outcome": "finished",
        },
    ]
    return {
        "trace": trace,
        "store_snapshot": {},
        "first_error": {
            "step": 1,
            "outcome": "blocked",
            "tool_name": "refund_order",
            "rule": "cross_user",
            "reason": "订单 o2 不属于当前用户 u1，拒绝操作",
        },
    }


def _success_case() -> dict:
    trace = [
        {
            "step": 1,
            "session_id": "s2",
            "tool_name": "refund_order",
            "args": {"order_id": "o1", "confidence": 0.95},
            "args_source": "context",
            "confidence": 0.95,
            "policy": {"allowed": True, "rule": None, "reason": "pass"},
            "executed": True,
            "tool_result": {"ok": True, "tool": "refund_order"},
            "outcome": "executed",
        },
        {
            "step": 2,
            "session_id": "s2",
            "tool_name": "finish",
            "args": {"final_answer": "已为您成功退款。"},
            "args_source": "context",
            "confidence": 0.95,
            "policy": {"allowed": True, "rule": None, "reason": "finish"},
            "executed": False,
            "outcome": "finished",
        },
    ]
    return {"trace": trace, "store_snapshot": {}, "first_error": None}


@unittest.skipUnless(FASTAPI_AVAILABLE, "未安装 FastAPI / httpx，跳过 API 测试")
class AuditApiTest(unittest.TestCase):
    def setUp(self) -> None:
        self.original_run_case = api_main.run_case
        self.captured: list[dict] = []

        def fake_run_case(**kwargs):
            self.captured.append(kwargs)
            return _blocked_case()

        api_main.run_case = fake_run_case
        self.addCleanup(setattr, api_main, "run_case", self.original_run_case)
        self.client = TestClient(api_main.app)

    def test_missing_or_invalid_authorization(self) -> None:
        missing = self.client.post(
            "/api/v1/audit", json={"user_message": "把 o2 退给我"}
        )
        self.assertEqual(missing.status_code, 401)

        invalid = self.client.post(
            "/api/v1/audit",
            headers={"Authorization": "Token u1"},
            json={"user_message": "把 o2 退给我"},
        )
        self.assertEqual(invalid.status_code, 401)

    def test_audit_returns_reply_block_reason_and_trace(self) -> None:
        response = self.client.post(
            "/api/v1/audit",
            headers={"Authorization": "Bearer u1"},
            json={"user_message": "把 o2 退给我"},
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(set(body), {"reply", "is_blocked", "block_reason", "trace"})
        self.assertIs(body["is_blocked"], True)
        self.assertIn("cross_user", body["block_reason"])
        self.assertIn("不属于", body["reply"])
        self.assertEqual(len(body["trace"]), 2)
        self.assertEqual(self.captured[-1]["user"], "u1")
        self.assertEqual(self.captured[-1]["text"], "把 o2 退给我")
        self.assertTrue(str(self.captured[-1]["trace_path"]).endswith(".jsonl"))

    def test_successful_operation_has_no_block_reason(self) -> None:
        api_main.run_case = lambda **kwargs: _success_case()
        response = self.client.post(
            "/api/v1/audit",
            headers={"Authorization": "Bearer u1"},
            json={"user_message": "帮我退 o1"},
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIs(body["is_blocked"], False)
        self.assertIsNone(body["block_reason"])
        self.assertEqual(body["reply"], "已为您成功退款。")

    def test_empty_message_rejected(self) -> None:
        response = self.client.post(
            "/api/v1/audit",
            headers={"Authorization": "Bearer u1"},
            json={"user_message": "   "},
        )
        self.assertEqual(response.status_code, 422)

    def test_cors_allows_all_origins(self) -> None:
        response = self.client.options(
            "/api/v1/audit",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "authorization,content-type",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn(
            response.headers.get("access-control-allow-origin"),
            {"*", "http://localhost:3000"},
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
