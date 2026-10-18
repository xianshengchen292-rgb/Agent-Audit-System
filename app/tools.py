"""订单工具与内存订单库。

本模块是 Agent 唯一能"接触世界"的地方：所有订单操作都从这里走。
工具不会因为业务错误而抛异常，而是返回 ``{"ok": False, "error": ...}``，
这样调用方可以稳定地把结果写进轨迹（trace），也方便定位错误发生在哪一步。

内存订单库：
    o1 属于 u1（机械键盘 K8，499 元，初始 paid）
    o2 属于 u2（降噪耳机 Air，899 元，初始 paid）
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Mapping, Optional

__all__ = [
    "Order",
    "ToolSpec",
    "TOOL_SPECS",
    "TOOL_NAMES",
    "HIGH_RISK_TOOLS",
    "call_tool",
    "query_order",
    "cancel_order",
    "refund_order",
    "get_order",
    "get_state",
    "list_orders",
    "reset_store",
    "ORDER_STATUS_PAID",
    "ORDER_STATUS_CANCELLED",
    "ORDER_STATUS_REFUNDED",
]

ORDER_STATUS_PAID = "paid"
ORDER_STATUS_CANCELLED = "cancelled"
ORDER_STATUS_REFUNDED = "refunded"

#: 对外统一使用这句话，避免泄露"订单是否存在"。
NOT_FOUND_MESSAGE = "订单不存在或不属于当前用户"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Order:
    """一笔订单。"""

    order_id: str
    user_id: str
    title: str
    amount: float
    status: str = ORDER_STATUS_PAID

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _seed_orders() -> Dict[str, Order]:
    return {
        "o1": Order(order_id="o1", user_id="u1", title="机械键盘 K8", amount=499.0),
        "o2": Order(order_id="o2", user_id="u2", title="降噪耳机 Air", amount=899.0),
    }


#: 内存订单库：order_id -> Order
_ORDERS: Dict[str, Order] = _seed_orders()

#: 退款台账：order_id -> 退款时间。同一订单只允许出现一次。
_REFUND_LEDGER: Dict[str, str] = {}


# --------------------------------------------------------------------------- #
# 订单库读写
# --------------------------------------------------------------------------- #
def reset_store() -> None:
    """把订单库恢复成初始状态（演示/测试用）。"""

    _ORDERS.clear()
    _ORDERS.update(_seed_orders())
    _REFUND_LEDGER.clear()


def get_order(order_id: Any) -> Optional[Order]:
    """按订单号查订单，查不到返回 None。"""

    if not isinstance(order_id, str):
        return None
    return _ORDERS.get(order_id.strip())


def list_orders(user_id: Optional[str] = None) -> list[Dict[str, Any]]:
    """列出订单；给了 user_id 只返回该用户的订单。"""

    orders = _ORDERS.values()
    if user_id is not None:
        orders = [o for o in orders if o.user_id == str(user_id).strip()]
    return [o.to_dict() for o in orders]


def get_state() -> Dict[str, Any]:
    """返回一份可直接 JSON 化的订单库快照，供 policy.check 与审计断言使用。

    顶层 ``refund_count`` 是退款总次数，每笔订单内也带一个 ``refund_count``（0/1）。
    """

    orders: Dict[str, Any] = {}
    for order_id, order in _ORDERS.items():
        item = order.to_dict()
        item["refund_count"] = 1 if order_id in _REFUND_LEDGER else 0
        orders[order_id] = item

    return {
        "orders": orders,
        "refunded_orders": sorted(_REFUND_LEDGER),
        "refunds": dict(_REFUND_LEDGER),
        "refund_count": len(_REFUND_LEDGER),
    }


# --------------------------------------------------------------------------- #
# 参数与返回值的辅助函数
# --------------------------------------------------------------------------- #
def _as_args(args: Any) -> Dict[str, Any]:
    if args is None:
        return {}
    if isinstance(args, Mapping):
        return dict(args)
    return {}


def _order_id(args: Mapping[str, Any]) -> Optional[str]:
    value = args.get("order_id")
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _fail(tool: str, message: str, **extra: Any) -> Dict[str, Any]:
    return {"ok": False, "tool": tool, "error": message, **extra}


def _authorized(order: Optional[Order], current_user: Optional[str]) -> bool:
    if order is None or not current_user:
        return False
    return order.user_id == str(current_user).strip()


# --------------------------------------------------------------------------- #
# 三个工具
# --------------------------------------------------------------------------- #
def query_order(args: Any = None, current_user: Optional[str] = None) -> Dict[str, Any]:
    """查询订单。只能查自己的订单。"""

    tool = "query_order"
    parsed = _as_args(args)
    order_id = _order_id(parsed)
    if order_id is None:
        return _fail(tool, "缺少 order_id，无法查询订单")

    order = get_order(order_id)
    if not _authorized(order, current_user):
        return _fail(tool, NOT_FOUND_MESSAGE)

    assert order is not None  # _authorized 已保证
    return {"ok": True, "tool": tool, "order": order.to_dict()}


def cancel_order(args: Any = None, current_user: Optional[str] = None) -> Dict[str, Any]:
    """取消订单。只能取消自己名下、尚未取消/退款的订单。"""

    tool = "cancel_order"
    parsed = _as_args(args)
    order_id = _order_id(parsed)
    if order_id is None:
        return _fail(tool, "缺少 order_id，无法取消订单")

    order = get_order(order_id)
    if not _authorized(order, current_user):
        return _fail(tool, NOT_FOUND_MESSAGE)

    assert order is not None
    if order.status == ORDER_STATUS_REFUNDED:
        return _fail(tool, "订单已退款，无法取消", order=order.to_dict())
    if order.status == ORDER_STATUS_CANCELLED:
        return _fail(tool, "订单已经取消过了，无需重复操作", order=order.to_dict())

    order.status = ORDER_STATUS_CANCELLED
    return {
        "ok": True,
        "tool": tool,
        "order": order.to_dict(),
        "message": "订单已取消",
    }


def refund_order(args: Any = None, current_user: Optional[str] = None) -> Dict[str, Any]:
    """退款。只有"已取消且未退过款"的订单才能退款。"""

    tool = "refund_order"
    parsed = _as_args(args)
    order_id = _order_id(parsed)
    if order_id is None:
        return _fail(tool, "缺少 order_id，无法退款")

    order = get_order(order_id)
    if not _authorized(order, current_user):
        return _fail(tool, NOT_FOUND_MESSAGE)

    assert order is not None
    if order_id in _REFUND_LEDGER or order.status == ORDER_STATUS_REFUNDED:
        return _fail(tool, "该订单已经退过款，同一订单只能退款一次", order=order.to_dict())
    if order.status != ORDER_STATUS_CANCELLED:
        return _fail(tool, "订单尚未取消，不能退款；请先取消订单", order=order.to_dict())

    refunded_at = _now()
    _REFUND_LEDGER[order_id] = refunded_at
    order.status = ORDER_STATUS_REFUNDED
    return {
        "ok": True,
        "tool": tool,
        "order": order.to_dict(),
        "refund": {
            "order_id": order_id,
            "user_id": order.user_id,
            "amount": order.amount,
            "refunded_at": refunded_at,
        },
        "message": "退款成功",
    }


# --------------------------------------------------------------------------- #
# 工具注册表
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ToolSpec:
    """工具元信息，Agent 的提示词和策略都从这里取。"""

    name: str
    description: str
    risk: str  # low / high
    handler: Callable[[Any, Optional[str]], Dict[str, Any]]


TOOL_SPECS: Dict[str, ToolSpec] = {
    "query_order": ToolSpec(
        name="query_order",
        description="查询订单状态、金额与归属；只能查询当前用户自己的订单。",
        risk="low",
        handler=query_order,
    ),
    "cancel_order": ToolSpec(
        name="cancel_order",
        description="取消订单；只有 paid 状态、属于当前用户的订单可以取消。",
        risk="high",
        handler=cancel_order,
    ),
    "refund_order": ToolSpec(
        name="refund_order",
        description="退款；只有已取消且从未退过款的订单可以退款。",
        risk="high",
        handler=refund_order,
    ),
}

TOOL_NAMES = tuple(TOOL_SPECS)
HIGH_RISK_TOOLS = frozenset(name for name, spec in TOOL_SPECS.items() if spec.risk == "high")


def call_tool(
    name: str,
    args: Any = None,
    current_user: Optional[str] = None,
) -> Dict[str, Any]:
    """按名字执行工具。未知工具和意外异常也会变成结构化的失败结果。"""

    spec = TOOL_SPECS.get(str(name).strip())
    if spec is None:
        return {"ok": False, "tool": name, "error": f"未知工具：{name}"}
    try:
        return spec.handler(args, current_user)
    except Exception as exc:  # 兜底：轨迹里必须总有一条结果
        return {"ok": False, "tool": spec.name, "error": f"工具执行异常：{exc}"}
