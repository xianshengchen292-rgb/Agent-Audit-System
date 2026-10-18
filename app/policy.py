"""订单客服 Agent 的业务策略：五条规则。

``check(tool_name, args, state, current_user)`` 是一个纯函数式判定：

* ``state`` 是订单库快照（``tools.get_state()`` 的返回值）。传 ``None`` 或空字典时，
  会读取 ``tools`` 里的实时内存订单库，方便在脚本里直接试用。
* ``confidence`` 从 ``args["confidence"]`` 读取，缺省视为 ``1.0``（即"没有提供置信度
  就不触发低置信度规则"）。Agent 在调用前会强制把模型输出的 confidence 合并进 args。
* 返回值是 ``Decision``，``bool(decision)`` 直接表示是否放行。

五条规则（按判定顺序）：
    1. missing_order_id —— 缺少订单 ID，拒绝；
    2. cross_user       —— 订单不属于当前用户（含订单不存在），拒绝；
    3. not_cancelled    —— 未取消的订单不能退款；
    4. already_refunded —— 同一订单只能退款一次；
    5. low_confidence   —— confidence < 0.6 时拒绝高风险操作（取消/退款）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

__all__ = [
    "Decision",
    "check",
    "check_decision",
    "is_high_risk",
    "CONFIDENCE_THRESHOLD",
    "HIGH_RISK_TOOLS",
    "ORDER_TOOLS",
    "RULES",
]

#: 低置信度阈值：严格小于该值才拦截。
CONFIDENCE_THRESHOLD = 0.6

ORDER_TOOLS = frozenset({"query_order", "cancel_order", "refund_order"})
HIGH_RISK_TOOLS = frozenset({"cancel_order", "refund_order"})
ORDER_ID_TOOLS = ORDER_TOOLS  # 三个工具都需要 order_id

RULE_MISSING_ORDER_ID = "missing_order_id"
RULE_CROSS_USER = "cross_user"
RULE_NOT_CANCELLED = "not_cancelled"
RULE_ALREADY_REFUNDED = "already_refunded"
RULE_LOW_CONFIDENCE = "low_confidence"
RULE_UNKNOWN_TOOL = "unknown_tool"

#: 规则说明，供报告/前端/README 复用。
RULES: Dict[str, str] = {
    RULE_MISSING_ORDER_ID: "缺少订单 ID：没有 order_id 的调用一律拒绝。",
    RULE_CROSS_USER: "跨用户操作：订单不存在或不属于当前用户时拒绝，且不区分两种情况。",
    RULE_NOT_CANCELLED: "未取消不能退款：只有 cancelled 状态的订单可以退款。",
    RULE_ALREADY_REFUNDED: "同一订单只能退一次：退款台账里出现过的订单再次退款即拒绝。",
    RULE_LOW_CONFIDENCE: f"低置信度：confidence < {CONFIDENCE_THRESHOLD} 时拒绝高风险操作。",
}


@dataclass(frozen=True)
class Decision:
    """策略判定结果。"""

    allowed: bool
    rule: Optional[str] = None
    reason: str = ""

    def __bool__(self) -> bool:  # 方便 `if check(...):`
        return self.allowed

    def to_dict(self) -> Dict[str, Any]:
        return {"allowed": self.allowed, "rule": self.rule, "reason": self.reason}


def _allow(reason: str = "策略校验通过") -> Decision:
    return Decision(True, None, reason)


def _deny(rule: str, reason: str) -> Decision:
    return Decision(False, rule, reason)


def _live_state() -> Dict[str, Any]:
    try:  # 包导入
        from . import tools
    except ImportError:  # 直接把 app/ 当脚本目录
        import tools  # type: ignore[no-redef]

    return tools.get_state()


def _normalize_args(args: Any) -> Dict[str, Any]:
    return dict(args) if isinstance(args, Mapping) else {}


def _normalize_order_id(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _confidence(args: Mapping[str, Any]) -> float:
    value = args.get("confidence")
    if value is None:
        return 1.0
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 1.0
    return min(1.0, max(0.0, number))


def _order_view(orders: Mapping[str, Any], order_id: str) -> Optional[Mapping[str, Any]]:
    """把订单记录统一成 mapping，兼容 dict / Order 数据类。"""

    raw = orders.get(order_id)
    if raw is None:
        return None
    if isinstance(raw, Mapping):
        return raw
    to_dict = getattr(raw, "to_dict", None)
    if callable(to_dict):
        result = to_dict()
        if isinstance(result, Mapping):
            return result
    return {
        "order_id": getattr(raw, "order_id", order_id),
        "user_id": getattr(raw, "user_id", None),
        "status": getattr(raw, "status", None),
    }


def check(
    tool_name: str,
    args: Optional[Mapping[str, Any]],
    state: Optional[Mapping[str, Any]] = None,
    current_user: Optional[str] = None,
) -> Decision:
    """按五条规则判定一次工具调用是否放行。"""

    tool = str(tool_name or "").strip()
    parsed = _normalize_args(args)

    if tool not in ORDER_TOOLS:
        return _deny(RULE_UNKNOWN_TOOL, f"未知工具：{tool!r}，策略无法放行")

    # 规则 1：缺少订单 ID -------------------------------------------------- #
    order_id = _normalize_order_id(parsed.get("order_id"))
    if order_id is None:
        return _deny(RULE_MISSING_ORDER_ID, f"{tool} 缺少 order_id，拒绝执行")

    # 读取状态快照 -------------------------------------------------------- #
    if state is None or (isinstance(state, Mapping) and not state):
        state = _live_state()
    if not isinstance(state, Mapping):
        return _deny(RULE_CROSS_USER, "订单状态不可用，无法确认订单归属，拒绝执行")

    orders = state.get("orders")
    orders = orders if isinstance(orders, Mapping) else {}
    refunded_raw = state.get("refunded_orders") or ()
    refunded = {str(item) for item in refunded_raw}

    # 规则 2：跨用户 ------------------------------------------------------ #
    order = _order_view(orders, order_id)
    user = str(current_user).strip() if current_user is not None else ""
    if order is None:
        return _deny(RULE_CROSS_USER, f"订单 {order_id} 不存在，或不属于当前用户，拒绝操作")
    owner = str(order.get("user_id") or "").strip()
    if not user or owner != user:
        return _deny(
            RULE_CROSS_USER,
            f"订单 {order_id} 不属于当前用户 {user or '<未登录>'}，拒绝操作",
        )

    # 规则 3、4：退款相关 -------------------------------------------------- #
    if tool == "refund_order":
        status = str(order.get("status") or "").strip().lower()
        already_refunded = status == "refunded" or order_id in refunded
        if not already_refunded and status != "cancelled":
            return _deny(
                RULE_NOT_CANCELLED,
                f"订单 {order_id} 当前状态为 {status or '未知'}，尚未取消，不能退款",
            )
        if already_refunded:
            return _deny(RULE_ALREADY_REFUNDED, f"订单 {order_id} 已经退过款，不能重复退款")

    # 规则 5：低置信度拦截高风险操作 --------------------------------------- #
    if tool in HIGH_RISK_TOOLS:
        confidence = _confidence(parsed)
        if confidence < CONFIDENCE_THRESHOLD:
            return _deny(
                RULE_LOW_CONFIDENCE,
                f"置信度 {confidence:.2f} 低于阈值 {CONFIDENCE_THRESHOLD}，拒绝高风险操作 {tool}",
            )

    return _allow(f"{tool} 通过策略校验")


def check_decision(
    decision: Any,
    state: Optional[Mapping[str, Any]] = None,
    current_user: Optional[str] = None,
) -> Decision:
    """便捷入口：直接对 Agent 的决策对象做策略校验。"""

    args = getattr(decision, "args", None)
    merged = _normalize_args(args)
    confidence = getattr(decision, "confidence", None)
    if confidence is not None:
        merged["confidence"] = confidence
    return check(getattr(decision, "tool_name", ""), merged, state, current_user)


def is_high_risk(tool_name: str) -> bool:
    return str(tool_name).strip() in HIGH_RISK_TOOLS
