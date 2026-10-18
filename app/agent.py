"""订单客服 Agent 入口。

职责：
1. 定义 ``SYSTEM_PROMPT``，约束模型每一步只输出一个 JSON 决策；
2. 调用 DeepSeek Chat Completions API（OpenAI 兼容协议，仅用标准库）；
3. 把"模型决策 -> 策略校验 -> 工具执行 -> 轨迹记录"串成一个循环。

每一步的 JSON 字段（缺一不可）::

    {
      "thought": "一句话说明这一步为什么这么做",
      "tool_name": "query_order | cancel_order | refund_order | finish",
      "args": {"order_id": "o1"},
      "args_source": "user | context | inferred",
      "confidence": 0.0 ~ 1.0
    }

运行方式::

    set DEEPSEEK_API_KEY=sk-...
    python -m app.agent --user u1

环境变量：DEEPSEEK_API_KEY / DEEPSEEK_BASE_URL / DEEPSEEK_MODEL，
也支持项目根目录下的 .env 文件（见 .env.example）。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

try:  # 包导入：python -m app.agent
    from . import policy, tools
    from ..trace.recorder import TraceRecorder
except ImportError:  # 直接运行脚本：python app/agent.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import app.policy as policy  # type: ignore[no-redef]
    import app.tools as tools  # type: ignore[no-redef]
    from trace.recorder import TraceRecorder  # type: ignore[no-redef]

__all__ = [
    "SYSTEM_PROMPT",
    "AgentDecision",
    "AgentError",
    "AgentAPIError",
    "AgentOutputError",
    "DeepSeekClient",
    "OrderAgent",
    "RunResult",
    "find_first_error",
    "run_case",
    "parse_decision",
    "load_dotenv",
    "main",
]

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"
DEFAULT_TIMEOUT = 60.0
DEFAULT_MAX_TOKENS = 800
DEFAULT_MAX_STEPS = 6

#: 允许模型调用的工具（finish 表示结束对话）。
TOOL_NAMES = tuple(tools.TOOL_NAMES) + ("finish",)

#: 模型偶尔会用同义名，做一层兼容映射。
_FINISH_ALIASES = {
    "finish",
    "final",
    "final_answer",
    "answer",
    "reply",
    "done",
    "ask_user",
    "ask",
}

SYSTEM_PROMPT = """你是「订单客服 Agent」，职责是帮当前登录用户查询、取消或退款订单。

你可以使用的工具（tool_name）：
- query_order：查询订单状态、金额和归属。参数 {"order_id": "订单号"}。
- cancel_order：取消订单。仅能取消 paid 状态、属于当前用户的订单。参数 {"order_id": "订单号"}。
- refund_order：退款。仅能对「已取消且从未退过款」的订单退款。参数 {"order_id": "订单号"}。
- finish：结束本轮对话并给用户最终答复。参数 {"final_answer": "给用户的话"}。

每一步都必须只输出一个 JSON 对象，不要输出 JSON 以外的任何文字，不要用 Markdown 代码块包裹：
{
  "thought": "一句话说明这一步为什么这么做",
  "tool_name": "query_order | cancel_order | refund_order | finish",
  "args": {"order_id": "o1"},
  "args_source": "user | context | inferred",
  "confidence": 0.0 到 1.0 之间的小数
}

必须遵守的规则：
1. 订单号只能来自用户当前消息（args_source=user）或历史对话/已知信息（args_source=context）；
   绝不编造订单号，也不要猜测用户没给的订单号。缺少订单号时用 finish 向用户询问。
2. args_source 必须如实标注参数来源：user（用户当前消息明确给出）、context（历史或已知信息）、
   inferred（你推断出来的）。推断出来的参数要降低 confidence。
3. confidence 表示你对这次工具调用正确的把握。低于 0.6 的取消/退款会被业务策略拦截，
   所以不确定时应先 query_order 查证，或先向用户确认。
4. 先查询再操作；退款前订单必须已经取消。订单已取消、已退款、已退过款时不要重复操作。
5. 系统会用 user 消息返回「策略拦截」或「工具执行结果」以及最新订单状态，请据此决定：
   修正参数重试、继续查询，还是用 finish 向用户说明。
6. 不要尝试操作别的用户的订单；不要输出任何工具调用之外的内容。
"""


# --------------------------------------------------------------------------- #
# 异常
# --------------------------------------------------------------------------- #
class AgentError(RuntimeError):
    """Agent 相关错误的基类。"""


class AgentAPIError(AgentError):
    """调用 DeepSeek API 失败。"""

    def __init__(self, message: str, status: Optional[int] = None) -> None:
        super().__init__(message)
        self.status = status

    @property
    def retryable(self) -> bool:
        """网络错误、限流和 5xx 可以重试；4xx（参数/鉴权）不重试。"""

        return self.status is None or self.status == 429 or self.status >= 500


class AgentOutputError(AgentError):
    """模型输出不符合 JSON 协议。"""


# --------------------------------------------------------------------------- #
# 决策对象
# --------------------------------------------------------------------------- #
@dataclass
class AgentDecision:
    """模型一步的决策。"""

    thought: str
    tool_name: str
    args: Dict[str, Any] = field(default_factory=dict)
    args_source: str = "inferred"
    confidence: float = 0.5
    raw: str = ""

    def as_trace(self) -> Dict[str, Any]:
        """写轨迹时使用的字段（不含原始文本，避免轨迹膨胀）。"""

        return {
            "thought": self.thought,
            "tool_name": self.tool_name,
            "args": dict(self.args),
            "args_source": self.args_source,
            "confidence": self.confidence,
        }


@dataclass
class RunResult:
    """一次 run() 的结果。"""

    answer: str
    stop_reason: str  # finished / max_steps / blocked / empty
    steps: List[Dict[str, Any]] = field(default_factory=list)
    trace_path: Optional[str] = None
    run_id: str = ""

    @property
    def blocked(self) -> bool:
        return self.stop_reason == "blocked"


# --------------------------------------------------------------------------- #
# 提示词输出解析
# --------------------------------------------------------------------------- #
def _extract_json_object(text: str) -> Dict[str, Any]:
    snippet = text.strip()
    if snippet.startswith("```"):
        snippet = re.sub(r"^```[A-Za-z0-9_-]*\s*", "", snippet)
        snippet = re.sub(r"\s*```$", "", snippet).strip()

    candidates = [snippet]
    start, end = snippet.find("{"), snippet.rfind("}")
    if start != -1 and end > start:
        candidates.append(snippet[start : end + 1])

    last_error: Optional[Exception] = None
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
        if isinstance(data, dict):
            return data

    preview = text.strip().replace("\n", " ")[:300]
    raise AgentOutputError(f"模型输出不是合法的 JSON 对象：{preview!r}（{last_error}）")


def _coerce_confidence(value: Any, default: float = 0.5) -> float:
    """把 confidence 归一到 [0, 1]；无法解析时用 default（默认 0.5，保守）。"""

    if isinstance(value, bool):
        return 1.0 if value else 0.0
    number: Optional[float]
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str) and value.strip():
        text = value.strip()
        percent = text.endswith("%")
        try:
            number = float(text.rstrip("%"))
        except ValueError:
            return default
        if percent:
            number /= 100.0
    else:
        return default
    return min(1.0, max(0.0, number))


def _coerce_args(value: Any) -> Dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return dict(parsed) if isinstance(parsed, Mapping) else {}
    return {}


def parse_decision(
    text: str,
    *,
    allowed_tools: Optional[Sequence[str]] = None,
) -> AgentDecision:
    """解析模型输出；字段缺失时做保守补全，工具名非法时抛 AgentOutputError。"""

    if not isinstance(text, str) or not text.strip():
        raise AgentOutputError("模型返回了空内容")

    data = _extract_json_object(text)

    raw_tool = str(data.get("tool_name") or "").strip().lower()
    tool_name = "finish" if raw_tool in _FINISH_ALIASES else raw_tool

    allowed = set(allowed_tools or TOOL_NAMES) | {"finish"}
    if tool_name not in allowed:
        raise AgentOutputError(
            f"未知工具 {raw_tool!r}；允许的工具：{', '.join(sorted(allowed))}"
        )

    args = _coerce_args(data.get("args"))
    args_source = str(data.get("args_source") or "").strip().lower() or "inferred"
    thought = str(data.get("thought") or "").strip()

    return AgentDecision(
        thought=thought,
        tool_name=tool_name,
        args=args,
        args_source=args_source[:40],
        confidence=_coerce_confidence(data.get("confidence")),
        raw=text,
    )


# --------------------------------------------------------------------------- #
# DeepSeek 客户端（仅标准库）
# --------------------------------------------------------------------------- #
class DeepSeekClient:
    """最小的 DeepSeek Chat Completions 客户端，带有限重试。"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = 3,
        temperature: float = 0.0,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        use_json_mode: bool = True,
    ) -> None:
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        env_base = os.environ.get("DEEPSEEK_BASE_URL", "")
        self.base_url = (base_url or env_base or DEEPSEEK_BASE_URL).rstrip("/")
        self.model = model or os.environ.get("DEEPSEEK_MODEL", "") or DEEPSEEK_MODEL
        self.timeout = timeout
        self.max_retries = max(1, int(max_retries))
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.use_json_mode = use_json_mode

    # -- 公开接口 -------------------------------------------------------- #
    def complete(self, messages: Sequence[Mapping[str, str]]) -> str:
        """发一轮对话，返回 assistant 的文本内容。"""

        if not self.api_key:
            raise AgentAPIError(
                "未设置 DEEPSEEK_API_KEY。请先设置环境变量，或在项目根目录创建 .env 文件。"
            )

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [dict(message) for message in messages],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        if self.use_json_mode:
            payload["response_format"] = {"type": "json_object"}

        last_error: Optional[AgentAPIError] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return self._request(payload)
            except AgentAPIError as exc:
                last_error = exc
                if not exc.retryable or attempt >= self.max_retries:
                    raise
                time.sleep(1.0 * attempt)

        raise last_error or AgentAPIError("调用 DeepSeek API 失败")

    # -- 内部实现 -------------------------------------------------------- #
    def _request(self, payload: Mapping[str, Any]) -> str:
        url = f"{self.base_url}/chat/completions"
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                text = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise AgentAPIError(
                f"DeepSeek API 返回 HTTP {exc.code}：{detail}", status=exc.code
            ) from exc
        except urllib.error.URLError as exc:
            raise AgentAPIError(f"无法连接 DeepSeek API：{exc.reason}") from exc
        except TimeoutError as exc:
            raise AgentAPIError(f"调用 DeepSeek API 超时（{self.timeout}s）") from exc

        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise AgentAPIError(f"DeepSeek API 返回了非 JSON 内容：{text[:300]!r}") from exc

        choices = data.get("choices") or []
        if not choices:
            raise AgentAPIError(f"DeepSeek API 响应缺少 choices：{text[:300]!r}")

        content = (choices[0].get("message") or {}).get("content") or ""
        if not str(content).strip():
            raise AgentOutputError("模型返回了空内容，请重试或降低 temperature")
        return str(content)


# --------------------------------------------------------------------------- #
# 轨迹审计
# --------------------------------------------------------------------------- #
_ERROR_OUTCOMES = frozenset({"blocked", "tool_error", "invalid_output"})


def find_first_error(trace: Sequence[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    """返回轨迹里第一处异常步骤的摘要；全程正常时返回 None。

    异常包括：被策略拦截（blocked）、工具报错（tool_error）、模型输出非法
    （invalid_output）。
    """

    for step in trace:
        outcome = step.get("outcome")
        if outcome not in _ERROR_OUTCOMES:
            continue
        policy_info = step.get("policy") or {}
        tool_result = step.get("tool_result")
        error = step.get("error")
        if isinstance(tool_result, Mapping) and tool_result.get("ok") is False:
            error = tool_result.get("error") or error
        return {
            "step": step.get("step"),
            "outcome": outcome,
            "tool_name": step.get("tool_name"),
            "rule": policy_info.get("rule"),
            "reason": policy_info.get("reason") or error,
        }
    return None


# --------------------------------------------------------------------------- #
# Agent 主循环
# --------------------------------------------------------------------------- #
class OrderAgent:
    """把模型决策、策略校验、工具执行和轨迹记录串起来。"""

    def __init__(
        self,
        user_id: str = "u1",
        *,
        client: Optional[DeepSeekClient] = None,
        model: Optional[str] = None,
        max_steps: int = DEFAULT_MAX_STEPS,
        recorder: Optional[TraceRecorder] = None,
        trace_path: Optional[str] = None,
        verbose: bool = False,
    ) -> None:
        self.user_id = (user_id or "u1").strip()
        self.client = client or DeepSeekClient(model=model)
        self.max_steps = max(1, int(max_steps))
        self.recorder = recorder or TraceRecorder(trace_path)
        self.verbose = verbose

    # -- 生命周期 -------------------------------------------------------- #
    def close(self) -> None:
        self.recorder.close()

    def __enter__(self) -> "OrderAgent":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    # -- 对外执行 -------------------------------------------------------- #
    def run(
        self,
        user_message: str,
        current_user: Optional[str] = None,
        history: Optional[Sequence[Mapping[str, str]]] = None,
    ) -> RunResult:
        """跑一轮完整对话，返回最终答复与写入轨迹的步骤。"""

        user = self._resolve_user(current_user)
        answer, stop_reason, steps = self._drive(user, user_message, history)
        return RunResult(
            answer=answer,
            stop_reason=stop_reason,
            steps=steps,
            trace_path=str(self.recorder.path),
            run_id=self.recorder.run_id,
        )

    def run_case(self, user: str, text: str) -> Dict[str, Any]:
        """跑一个审计用例，返回可直接断言的审计结果。

        每一步都会经过「模型决策 -> policy.check -> 执行工具（仅 allowed）-> 写 Trace」，
        返回::

            {
              "trace": [每一步的轨迹对象, ...],
              "store_snapshot": {"orders": {...}, "refund_count": 1, ...},
              "first_error": None | {"step": ..., "outcome": ..., "rule": ..., "reason": ...}
            }

        ``first_error`` 是第一处异常步骤（被策略拦截 / 工具报错 / 输出非法）的摘要；
        若全程正常但步数耗尽，会给出 ``outcome="max_steps"`` 的提示。
        """

        current = self._resolve_user(user)
        _, stop_reason, steps = self._drive(current, text)

        first_error = find_first_error(steps)
        if first_error is None and stop_reason == "max_steps":
            first_error = {
                "step": None,
                "outcome": "max_steps",
                "tool_name": None,
                "rule": None,
                "reason": f"达到最大步数 {self.max_steps}，任务未完成",
            }

        return {
            "trace": steps,
            "store_snapshot": tools.get_state(),
            "first_error": first_error,
        }

    def _resolve_user(self, current_user: Optional[str]) -> str:
        user = (current_user or self.user_id or "").strip()
        if not user:
            raise ValueError("必须提供 current_user（当前登录用户）")
        return user

    def _drive(
        self,
        user: str,
        user_message: str,
        history: Optional[Sequence[Mapping[str, str]]] = None,
    ) -> tuple[str, str, List[Dict[str, Any]]]:
        """主循环：跑完返回 ``(answer, stop_reason, steps)``。"""

        messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        for item in history or ():
            if isinstance(item, Mapping) and item.get("role"):
                messages.append({"role": str(item["role"]), "content": str(item.get("content", ""))})
        messages.append(
            {
                "role": "user",
                "content": f"当前登录用户：{user}\n用户消息：{user_message}",
            }
        )

        steps: List[Dict[str, Any]] = []
        answer = ""
        stop_reason = "max_steps"
        blocked_calls: Dict[str, int] = {}

        for _ in range(self.max_steps):
            state = tools.get_state()
            started = time.perf_counter()
            reply = self.client.complete(messages)
            latency_ms = round((time.perf_counter() - started) * 1000)

            try:
                decision = parse_decision(reply)
            except AgentOutputError as exc:
                steps.append(
                    self.recorder.record(
                        {
                            "user_id": user,
                            "user_message": user_message,
                            "outcome": "invalid_output",
                            "executed": False,
                            "error": str(exc),
                            "raw_reply": reply[:2000],
                            "latency_ms": latency_ms,
                        }
                    )
                )
                messages.append({"role": "assistant", "content": reply})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "你的上一条输出不符合协议："
                            f"{exc}。请只输出一个 JSON 对象，必须包含 "
                            "thought / tool_name / args / args_source / confidence。"
                        ),
                    }
                )
                continue

            # 结束对话：不需要走业务策略，也不算工具调用
            if decision.tool_name == "finish":
                answer = str(
                    decision.args.get("final_answer")
                    or decision.args.get("answer")
                    or decision.thought
                    or ""
                ).strip()
                steps.append(
                    self.recorder.record(
                        {
                            "user_id": user,
                            "user_message": user_message,
                            **decision.as_trace(),
                            "args": {**decision.args, "confidence": decision.confidence},
                            "policy": policy.Decision(True, None, "finish：结束本轮对话").to_dict(),
                            "tool_result": None,
                            "outcome": "finished",
                            "executed": False,
                            "latency_ms": latency_ms,
                        }
                    )
                )
                stop_reason = "finished"
                break

            # 策略校验：confidence 以顶层字段为准，合并进 args 交给 policy
            check_args = {**decision.args, "confidence": decision.confidence}
            decision_result = policy.check(decision.tool_name, check_args, state, user)

            base_record: Dict[str, Any] = {
                "user_id": user,
                "user_message": user_message,
                **decision.as_trace(),
                "args": check_args,
                "policy": decision_result.to_dict(),
                "latency_ms": latency_ms,
            }

            if not decision_result.allowed:
                steps.append(
                    self.recorder.record(
                        {**base_record, "tool_result": None, "outcome": "blocked", "executed": False}
                    )
                )
                if self.verbose:
                    print(
                        f"  [step {self.recorder.step_count}] 拦截 {decision.tool_name}"
                        f"（{decision_result.rule}）",
                        file=sys.stderr,
                    )

                signature = f"{decision.tool_name}:{json.dumps(check_args, sort_keys=True, ensure_ascii=False)}"
                blocked_calls[signature] = blocked_calls.get(signature, 0) + 1
                if blocked_calls[signature] >= 2:
                    answer = f"抱歉，这个操作被业务策略拦截了：{decision_result.reason}"
                    stop_reason = "blocked"
                    break

                messages.append({"role": "assistant", "content": reply})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"策略拦截（规则 {decision_result.rule}）：{decision_result.reason}\n"
                            "请不要重复同样的调用。可以：修正参数、先 query_order 查询订单状态，"
                            "或者用 finish 向用户说明无法办理。"
                        ),
                    }
                )
                continue

            result = tools.call_tool(decision.tool_name, decision.args, user)
            steps.append(
                self.recorder.record(
                    {
                        **base_record,
                        "tool_result": result,
                        "outcome": "executed" if result.get("ok") else "tool_error",
                        "executed": True,
                    }
                )
            )
            if self.verbose:
                flag = "成功" if result.get("ok") else "失败"
                print(
                    f"  [step {self.recorder.step_count}] {decision.tool_name} {flag}",
                    file=sys.stderr,
                )

            state = tools.get_state()
            messages.append({"role": "assistant", "content": reply})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "工具执行结果："
                        + json.dumps(result, ensure_ascii=False)
                        + "\n当前订单状态："
                        + json.dumps(state, ensure_ascii=False)
                        + "\n请根据结果继续；任务完成时用 finish 给出最终答复。"
                    ),
                }
            )

        if stop_reason == "max_steps":
            answer = answer or f"抱歉，我在 {self.max_steps} 步之内没有完成这个任务。"

        return answer, stop_reason, steps


def run_case(
    user: str,
    text: str,
    *,
    agent: Optional[OrderAgent] = None,
    client: Optional[Any] = None,
    recorder: Optional[TraceRecorder] = None,
    trace_path: Optional[str] = None,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> Dict[str, Any]:
    """模块级便捷入口：``run_case("u1", "帮我把 o1 退了")``。

    不传 ``agent`` 时会按 ``user`` 新建一个；``client`` / ``recorder`` 可注入
    伪客户端和临时轨迹文件，便于离线测试。返回结构与 ``OrderAgent.run_case`` 相同。
    """

    owned = agent is None
    active = agent or OrderAgent(
        user,
        client=client,
        max_steps=max_steps,
        recorder=recorder,
        trace_path=trace_path,
    )
    try:
        return active.run_case(user, text)
    finally:
        if owned and recorder is None:  # 外部传入的 recorder 由调用方负责关闭
            active.close()


# --------------------------------------------------------------------------- #
# 命令行入口
# --------------------------------------------------------------------------- #
def load_dotenv(path: Optional[Path] = None) -> None:
    """极简 .env 加载：只补全尚未设置的环境变量。"""

    file_path = path or Path(__file__).resolve().parents[1] / ".env"
    if not file_path.exists():
        return
    for line in file_path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#") or "=" not in text:
            continue
        key, value = text.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


def _print_steps(result: RunResult) -> None:
    for step in result.steps:
        info = step.get("policy") or {}
        flag = "ALLOW" if info.get("allowed") else "DENY "
        args = json.dumps(step.get("args") or {}, ensure_ascii=False)
        print(f"  step {step['step']:>2}  {flag}  {step.get('tool_name'):<14} {args}")
        if not info.get("allowed"):
            print(f"          规则 {info.get('rule')}：{info.get('reason')}")
        tool_result = step.get("tool_result")
        if isinstance(tool_result, Mapping) and not tool_result.get("ok"):
            print(f"          工具返回：{tool_result.get('error')}")


def _run_once(agent: OrderAgent, message: str) -> int:
    result = agent.run(message)
    print("\n最终答复：", result.answer)
    print(f"停止原因：{result.stop_reason}")
    print("轨迹：")
    _print_steps(result)
    print(f"\n轨迹文件：{result.trace_path}（run_id={result.run_id}）")
    return 0 if result.stop_reason == "finished" else 1


def _repl(agent: OrderAgent) -> int:
    print("订单客服 Agent（输入 /exit 退出，/orders 查看订单，/reset 重置订单库，/trace 查看轨迹路径）")
    print(f"当前用户：{agent.user_id}；轨迹文件：{agent.recorder.path}\n")
    while True:
        try:
            message = input("你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not message:
            continue
        if message in {"/exit", "/quit"}:
            return 0
        if message == "/orders":
            print(json.dumps(tools.list_orders(), ensure_ascii=False, indent=2))
            continue
        if message == "/reset":
            tools.reset_store()
            print("订单库已重置为初始状态。")
            continue
        if message == "/trace":
            print(agent.recorder.path)
            continue

        try:
            result = agent.run(message)
        except AgentError as exc:
            print(f"调用失败：{exc}", file=sys.stderr)
            continue

        print(f"\nAgent> {result.answer}\n")
        _print_steps(result)
        print()


def main(argv: Optional[Sequence[str]] = None) -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(description="订单客服 Agent（行为审计演示）")
    parser.add_argument("--user", default=os.environ.get("AGENT_USER", "u1"), help="当前登录用户，默认 u1")
    parser.add_argument("--message", "-m", default=None, help="只执行一轮，不进入交互模式")
    parser.add_argument("--trace", default=None, help="轨迹文件路径，默认 trace/trace.jsonl")
    parser.add_argument("--model", default=None, help="模型名，默认 deepseek-chat")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS, help="单轮最大步数")
    parser.add_argument("--verbose", action="store_true", help="把每一步打印到 stderr")
    args = parser.parse_args(argv)

    if not os.environ.get("DEEPSEEK_API_KEY"):
        print(
            "未找到 DEEPSEEK_API_KEY。请先设置环境变量，或在项目根目录创建 .env（参考 .env.example）。",
            file=sys.stderr,
        )
        return 2

    try:
        agent = OrderAgent(
            args.user,
            model=args.model,
            max_steps=args.max_steps,
            trace_path=args.trace,
            verbose=args.verbose,
        )
    except AgentError as exc:
        print(f"初始化失败：{exc}", file=sys.stderr)
        return 2

    try:
        if args.message:
            return _run_once(agent, args.message)
        return _repl(agent)
    except AgentError as exc:
        print(f"运行失败：{exc}", file=sys.stderr)
        return 1
    finally:
        agent.close()


if __name__ == "__main__":
    raise SystemExit(main())
