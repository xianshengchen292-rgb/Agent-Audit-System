"""轨迹（trace）记录器与首个错误定位。

每一步都追加一行 JSON 到 ``trace/trace.jsonl``，一行一个对象，便于 diff 和回放。

轨迹步骤的字段约定::

    {
      "step": 1,                 # 全局单调递增
      "run_id": "20261018T...",  # 记录器实例 ID
      "ts": "2026-10-18T09:49:37+00:00",
      "session_id": "a1b2c3...", # 单次 Agent 会话 ID
      "user_id": "u1",
      "user_message": "帮我把 o1 退了吧",
      "thought": "...",
      "tool_name": "refund_order",
      "args": {"order_id": "o1", "confidence": 0.92},
      "args_source": "user",
      "confidence": 0.92,
      "policy": {"allowed": false, "rule": "not_cancelled", "reason": "..."},
      "tool_result": null,
      "outcome": "blocked",
      "session_steps": 1,
      "session_tokens": 1234,
      "session_cost_cny": 0.0031
    }

熔断会额外写入 ``breaker`` / ``outcome``（``max_steps_exceeded`` 或
``cost_limit_exceeded``），并由 ``record_breaker()`` 统一生成。

``locate_first_error(expected, actual)`` 用期望轨迹和实际轨迹逐步对比，
返回第一处不一致的位置；完全一致时返回 ``None``。
"""

from __future__ import annotations

import dataclasses
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence, Union
from uuid import uuid4

__all__ = [
    "TraceRecorder",
    "DEFAULT_TRACE_PATH",
    "DEFAULT_COMPARE_FIELDS",
    "BREAKER_MAX_STEPS",
    "BREAKER_COST_LIMIT",
    "BREAKER_TYPES",
    "load_trace",
    "locate_first_error",
    "format_first_error",
]

DEFAULT_TRACE_PATH = Path(__file__).resolve().parents[1] / "trace" / "trace.jsonl"

#: 两类硬熔断：超过最大步数、超过单会话预估成本。
BREAKER_MAX_STEPS = "max_steps_exceeded"
BREAKER_COST_LIMIT = "cost_limit_exceeded"
BREAKER_TYPES = frozenset({BREAKER_MAX_STEPS, BREAKER_COST_LIMIT})

#: 默认只比对行为相关字段；``policy.allowed`` 在期望轨迹里没有时会自动跳过。
DEFAULT_COMPARE_FIELDS: tuple[str, ...] = ("tool_name", "args", "policy.allowed")

#: args 里的通配符：表示"这个字段不关心"。
ANY = "__any__"

_MISSING = object()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json_default(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    if isinstance(value, (set, frozenset)):
        return sorted(value, key=str)
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _dumps(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, default=_json_default)


# --------------------------------------------------------------------------- #
# 写入
# --------------------------------------------------------------------------- #
class TraceRecorder:
    """把每一步追加写入 JSONL，并在同一文件上保持 step 连续递增。"""

    def __init__(
        self,
        path: Optional[Union[str, os.PathLike[str]]] = None,
        *,
        reset: bool = False,
        run_id: Optional[str] = None,
    ) -> None:
        self.path = Path(path).expanduser() if path else DEFAULT_TRACE_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if reset and self.path.exists():
            self.path.unlink()
        self.run_id = run_id or (
            datetime.now().strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:6]
        )
        self._step = self._count_existing()
        self._handle = self.path.open("a", encoding="utf-8")

    def _count_existing(self) -> int:
        if not self.path.exists():
            return 0
        count = 0
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    count += 1
        return count

    @property
    def step_count(self) -> int:
        return self._step

    def record(self, step: Mapping[str, Any]) -> dict[str, Any]:
        """追加一步，返回实际写入的对象（含 step / run_id / ts）。"""

        if self._handle.closed:
            raise RuntimeError("TraceRecorder 已关闭，无法继续记录")

        self._step += 1
        payload: dict[str, Any] = {
            "step": self._step,
            "run_id": self.run_id,
            "ts": _now(),
        }
        for key, value in step.items():
            if key in payload:  # step / run_id / ts 以记录器为准
                continue
            payload[key] = value

        self._handle.write(_dumps(payload) + "\n")
        self._handle.flush()
        return payload

    def record_breaker(self, breaker: str, reason: str, **fields: Any) -> dict[str, Any]:
        """记录一次硬熔断（步数超限 / 成本超限）。

        breaker 必须是 ``BREAKER_MAX_STEPS`` 或 ``BREAKER_COST_LIMIT``。
        写入的步骤会带 ``breaker`` 字段、``outcome`` 与固定的 deny policy，
        供面板 / 断言直接识别。
        """

        if breaker not in BREAKER_TYPES:
            raise ValueError(f"未知熔断类型：{breaker!r}")

        payload: dict[str, Any] = dict(fields)
        payload.update(
            {
                "outcome": breaker,
                "breaker": breaker,
                "executed": False,
                "policy": {"allowed": False, "rule": breaker, "reason": reason},
            }
        )
        return self.record(payload)

    def read(self) -> list[dict[str, Any]]:
        """读回当前文件里的全部步骤。"""

        return load_trace(self.path)

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.close()

    def __enter__(self) -> "TraceRecorder":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


# --------------------------------------------------------------------------- #
# 读取
# --------------------------------------------------------------------------- #
def load_trace(path: Union[str, os.PathLike[str]]) -> list[dict[str, Any]]:
    """读取 JSONL 轨迹文件；空行忽略，坏行会指出具体行号。"""

    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"轨迹文件不存在：{file_path}")

    steps: list[dict[str, Any]] = []
    with file_path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                item = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{file_path}:{lineno} 不是合法 JSON：{exc}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"{file_path}:{lineno} 不是 JSON 对象")
            steps.append(item)
    return steps


# --------------------------------------------------------------------------- #
# 比对
# --------------------------------------------------------------------------- #
def _as_steps(source: Union[str, os.PathLike[str], Mapping[str, Any], Iterable[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    if isinstance(source, (str, os.PathLike)):
        return load_trace(source)
    if isinstance(source, Mapping):
        return [dict(source)]
    steps = []
    for item in source:
        if not isinstance(item, Mapping):
            raise TypeError(f"轨迹步骤必须是 mapping，收到 {type(item)!r}")
        steps.append(dict(item))
    return steps


def _resolve(obj: Mapping[str, Any], dotted: str) -> tuple[bool, Any]:
    current: Any = obj
    for part in dotted.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return False, None
        current = current[part]
    return True, current


def _matches(expected: Any, actual: Any) -> bool:
    """期望值是"子集"语义：只比较期望里出现的键，``__any__`` 表示不关心。"""

    if expected == ANY:
        return True
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            return False
        for key, want in expected.items():
            if key not in actual:
                return False
            if not _matches(want, actual[key]):
                return False
        return True
    return expected == actual


def locate_first_error(
    expected: Union[str, os.PathLike[str], Mapping[str, Any], Iterable[Mapping[str, Any]]],
    actual: Union[str, os.PathLike[str], Mapping[str, Any], Iterable[Mapping[str, Any]]],
    *,
    fields: Sequence[str] = DEFAULT_COMPARE_FIELDS,
) -> Optional[dict[str, Any]]:
    """定位期望轨迹与实际轨迹的第一处差异。

    返回 ``None`` 表示两条轨迹在这些字段上一致；否则返回::

        {"index": 0, "step": 2, "field": "args.order_id",
         "expected": "o1", "actual": "o2", "reason": "field_mismatch",
         "message": "..."}

    ``reason`` 取值：``field_mismatch`` / ``missing_field`` / ``missing_step`` /
    ``extra_step``。``index`` 从 0 开始，``step`` 是轨迹里的 step 编号。
    """

    exp = _as_steps(expected)
    act = _as_steps(actual)

    for index in range(max(len(exp), len(act))):
        if index >= len(exp):
            return {
                "index": index,
                "step": act[index].get("step", index + 1),
                "field": None,
                "expected": None,
                "actual": act[index],
                "reason": "extra_step",
                "message": f"实际轨迹比期望多出第 {index + 1} 步（step={act[index].get('step')}）",
            }
        if index >= len(act):
            return {
                "index": index,
                "step": exp[index].get("step", index + 1),
                "field": None,
                "expected": exp[index],
                "actual": None,
                "reason": "missing_step",
                "message": f"实际轨迹缺少第 {index + 1} 步（step={exp[index].get('step')}）",
            }

        expected_step, actual_step = exp[index], act[index]
        for field in fields:
            found, want = _resolve(expected_step, field)
            if not found:  # 期望里没写就不比较
                continue
            found_actual, got = _resolve(actual_step, field)
            if not found_actual:
                return {
                    "index": index,
                    "step": expected_step.get("step", index + 1),
                    "field": field,
                    "expected": want,
                    "actual": None,
                    "reason": "missing_field",
                    "message": f"第 {index + 1} 步缺少字段 {field}",
                }
            if not _matches(want, got):
                return {
                    "index": index,
                    "step": expected_step.get("step", index + 1),
                    "field": field,
                    "expected": want,
                    "actual": got,
                    "reason": "field_mismatch",
                    "message": (
                        f"第 {index + 1} 步字段 {field} 不一致："
                        f"期望 {want!r}，实际 {got!r}"
                    ),
                }

    return None


def format_first_error(info: Optional[Mapping[str, Any]]) -> str:
    """把 ``locate_first_error`` 的结果格式化成一行中文说明。"""

    if not info:
        return "轨迹一致，未发现错误。"
    return f"[{info.get('reason')}] {info.get('message')}"
