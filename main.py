"""FastAPI 审计接口入口。

启动::

    uvicorn main:app --reload --host 0.0.0.0 --port 8000

接口::

    POST /api/v1/audit
    Authorization: Bearer u1
    Content-Type: application/json

    {"user_message": "帮我把 o1 退了"}

返回：``reply`` / ``is_blocked`` / ``block_reason`` / ``trace``。
交互式文档：``http://127.0.0.1:8000/docs``。
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.agent import AgentAPIError, answer_from_case, load_dotenv, run_case

load_dotenv()

TRACE_DIR = Path(__file__).resolve().parent / "trace"

app = FastAPI(
    title="AI 智能售后助手 · 审计 API",
    description="调用订单客服 Agent 并返回自然语言回复、拦截判定与完整 Trace。",
    version="1.0.0",
)

# 方便后续前端页面直接跨域调用；Authorization 头通过 allow_headers 放行。
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


class AuditRequest(BaseModel):
    user_message: str = Field(..., description="用户的自然语言输入")


class AuditResponse(BaseModel):
    reply: str
    is_blocked: bool
    block_reason: str | None = None
    trace: list[dict[str, Any]]


class HealthResponse(BaseModel):
    status: str


def _user_from_authorization(authorization: str | None) -> str:
    """解析 ``Authorization: Bearer u1``。"""

    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="缺少 Authorization 头，格式应为：Bearer <user_id>",
            headers={"WWW-Authenticate": "Bearer"},
        )

    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer" or not parts[1]:
        raise HTTPException(
            status_code=401,
            detail="Authorization 格式错误，应为：Bearer <user_id>",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return parts[1].strip()


def _block_info(trace: list[dict[str, Any]]) -> tuple[bool, str | None]:
    """返回 (是否被拦截, 最近一次拦截原因)。"""

    denied: list[Mapping[str, Any]] = []
    for step in trace:
        policy = step.get("policy")
        if isinstance(policy, Mapping) and not policy.get("allowed", True):
            denied.append(step)
        elif step.get("breaker"):
            denied.append(step)

    if not denied:
        return False, None

    latest = denied[-1]
    info = latest.get("policy") or {}
    rule = info.get("rule") or latest.get("breaker")
    reason = info.get("reason") or latest.get("error") or "策略拒绝"
    block_reason = f"{rule}: {reason}" if rule else str(reason)
    return True, block_reason


@app.get("/", tags=["system"])
def root() -> dict[str, str]:
    return {
        "service": "AI 智能售后助手 · 审计 API",
        "docs": "/docs",
        "audit": "/api/v1/audit",
    }


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/api/v1/audit", response_model=AuditResponse, tags=["audit"])
def audit(
    payload: AuditRequest,
    authorization: str | None = Header(default=None, alias="Authorization"),
) -> AuditResponse:
    user_id = _user_from_authorization(authorization)

    text = (payload.user_message or "").strip()
    if not text:
        raise HTTPException(status_code=422, detail="user_message 不能为空")

    try:
        # 每个请求独立一个轨迹文件，避免并发请求在同一 JSONL 中争抢 step 计数。
        trace_path = TRACE_DIR / (
            f"api_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:6]}.jsonl"
        )
        case = run_case(user=user_id, text=text, trace_path=str(trace_path))
    except AgentAPIError as exc:
        raise HTTPException(status_code=502, detail=f"模型调用失败：{exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"审计执行失败：{exc}") from exc

    trace = case.get("trace") or []
    is_blocked, block_reason = _block_info(trace)

    return AuditResponse(
        reply=answer_from_case(case),
        is_blocked=is_blocked,
        block_reason=block_reason,
        trace=trace,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000)
