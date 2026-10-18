"""订单工具与 SQLite 订单库。

本模块是 Agent 唯一能"接触世界"的地方：所有订单操作都从这里走。
工具不会因为业务错误而抛异常，而是返回 ``{"ok": False, "error": ...}``，
这样调用方可以稳定地把结果写进轨迹（trace），也方便定位错误发生在哪一步。

订单库默认位于项目根目录的 ``orders.db``，可通过环境变量
``ORDERS_DB_PATH`` 覆盖。建表与初始数据见根目录 ``init_db.py``：

    o1 属于 u1（机械键盘 K8，499 元，初始 paid）
    o2 属于 u2（降噪耳机 Air，899 元，初始 paid）
"""

from __future__ import annotations

import os
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Mapping, Optional

__all__ = [
    "DB_PATH",
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
    "record_agent_run",
    "init_db",
    "ORDER_STATUS_PAID",
    "ORDER_STATUS_CANCELLED",
    "ORDER_STATUS_REFUNDED",
]

ORDER_STATUS_PAID = "paid"
ORDER_STATUS_CANCELLED = "cancelled"
ORDER_STATUS_REFUNDED = "refunded"

#: 对外统一使用这句话，避免泄露"订单是否存在"。
NOT_FOUND_MESSAGE = "订单不存在或不属于当前用户"

#: 默认数据库位置：项目根目录 / orders.db
_DEFAULT_DB_PATH = Path(__file__).resolve().parents[1] / "orders.db"

_env_db = os.environ.get("ORDERS_DB_PATH") or os.environ.get("ORDERS_DB")
DB_PATH = Path(_env_db).expanduser() if _env_db else _DEFAULT_DB_PATH
if not DB_PATH.is_absolute():
    DB_PATH = (Path.cwd() / DB_PATH).resolve()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS orders (
    order_id     TEXT PRIMARY KEY,
    user_id      TEXT NOT NULL,
    title        TEXT NOT NULL,
    amount       REAL NOT NULL DEFAULT 0.0,
    status       TEXT NOT NULL DEFAULT 'paid'
                 CHECK (status IN ('paid', 'cancelled', 'refunded')),
    refund_count INTEGER NOT NULL DEFAULT 0
                 CHECK (refund_count >= 0),
    refunded_at  TEXT,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_orders_user_id ON orders(user_id);

CREATE TABLE IF NOT EXISTS agent_runs (
    run_id           TEXT PRIMARY KEY,
    user_id          TEXT NOT NULL,
    created_at       TEXT NOT NULL,
    steps            INTEGER NOT NULL DEFAULT 0,
    session_tokens   INTEGER NOT NULL DEFAULT 0,
    session_cost_cny REAL NOT NULL DEFAULT 0.0,
    stop_reason      TEXT NOT NULL,
    breaker          TEXT
);

CREATE INDEX IF NOT EXISTS idx_agent_runs_created_at ON agent_runs(created_at);
"""

#: 初始订单：order_id, user_id, title, amount, status
_INITIAL_ORDERS: tuple[tuple[str, str, str, float, str], ...] = (
    ("o1", "u1", "机械键盘 K8", 499.0, ORDER_STATUS_PAID),
    ("o2", "u2", "降噪耳机 Air", 899.0, ORDER_STATUS_PAID),
)

_READY = False
_READY_LOCK = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Order:
    """一笔订单（从 SQLite 行映射而来）。"""

    order_id: str
    user_id: str
    title: str
    amount: float
    status: str = ORDER_STATUS_PAID
    refund_count: int = 0
    refunded_at: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# 连接与建表
# --------------------------------------------------------------------------- #
@contextmanager
def _db() -> Iterator[sqlite3.Connection]:
    """每次操作一个短连接，避免 Streamlit 多线程复用连接的问题。"""

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(DB_PATH), timeout=10.0)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA busy_timeout = 10000")
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def _seed_rows() -> list[tuple[Any, ...]]:
    now = _now()
    return [
        (order_id, user_id, title, amount, status, 0, None, now, now)
        for order_id, user_id, title, amount, status in _INITIAL_ORDERS
    ]


def init_db(reset: bool = False) -> Path:
    """创建订单表并注入初始数据；``reset=True`` 时先删表重建。"""

    global _READY

    with _db() as connection:
        connection.execute("PRAGMA journal_mode = WAL")
        if reset:
            connection.execute("DROP TABLE IF EXISTS orders")
            connection.execute("DROP TABLE IF EXISTS agent_runs")
        connection.executescript(_SCHEMA)
        connection.executemany(
            """
            INSERT OR IGNORE INTO orders
                (order_id, user_id, title, amount, status,
                 refund_count, refunded_at, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            _seed_rows(),
        )

    _READY = True
    return DB_PATH


def _ensure_ready() -> None:
    """首次调用时自动建库，避免忘记运行 init_db.py 直接报 no such table。"""

    if _READY:
        return
    with _READY_LOCK:
        if not _READY:
            init_db()


# --------------------------------------------------------------------------- #
# 订单库读写
# --------------------------------------------------------------------------- #
def _row_to_order(row: sqlite3.Row) -> Order:
    return Order(
        order_id=str(row["order_id"]),
        user_id=str(row["user_id"]),
        title=str(row["title"]),
        amount=float(row["amount"]),
        status=str(row["status"]),
        refund_count=int(row["refund_count"]),
        refunded_at=row["refunded_at"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def reset_store() -> None:
    """把订单库恢复成初始状态（演示/测试用）：删表重建并重新注入 o1/o2。

    同时清空 ``agent_runs`` 运行审计表，保证测试之间互不影响。
    """

    init_db(reset=True)


def get_order(order_id: Any) -> Optional[Order]:
    """按订单号查订单，查不到返回 None。"""

    if not isinstance(order_id, str):
        return None
    oid = order_id.strip()
    if not oid:
        return None

    _ensure_ready()
    with _db() as connection:
        row = connection.execute(
            "SELECT * FROM orders WHERE order_id = ?", (oid,)
        ).fetchone()
    return _row_to_order(row) if row is not None else None


def list_orders(user_id: Optional[str] = None) -> list[Dict[str, Any]]:
    """列出订单；给了 user_id 只返回该用户的订单。"""

    _ensure_ready()
    with _db() as connection:
        if user_id is None:
            rows = connection.execute(
                "SELECT * FROM orders ORDER BY order_id"
            ).fetchall()
        else:
            rows = connection.execute(
                "SELECT * FROM orders WHERE user_id = ? ORDER BY order_id",
                (str(user_id).strip(),),
            ).fetchall()
    return [_row_to_order(row).to_dict() for row in rows]


def get_state() -> Dict[str, Any]:
    """从 SQLite 读取实时快照，供 policy.check、UI 与审计断言使用。

    顶层 ``refund_count`` 是退款总次数，每笔订单内也带自己的 ``refund_count``。
    """

    _ensure_ready()
    with _db() as connection:
        rows = connection.execute("SELECT * FROM orders ORDER BY order_id").fetchall()

    orders: Dict[str, Any] = {}
    for row in rows:
        order = _row_to_order(row)
        orders[order.order_id] = order.to_dict()

    refunded_orders = sorted(
        order_id
        for order_id, order in orders.items()
        if int(order["refund_count"]) > 0 or order["status"] == ORDER_STATUS_REFUNDED
    )
    refunds = {
        order_id: order["refunded_at"]
        for order_id, order in orders.items()
        if order.get("refunded_at")
    }

    return {
        "orders": orders,
        "refunded_orders": refunded_orders,
        "refunds": refunds,
        "refund_count": sum(int(order["refund_count"]) for order in orders.values()),
    }


def record_agent_run(
    *,
    run_id: str,
    user_id: str,
    steps: int,
    session_tokens: int,
    session_cost_cny: float,
    stop_reason: str,
    breaker: Optional[str] = None,
) -> None:
    """把一次 Agent 会话的步数、Token、预估成本和熔断状态写入审计表。"""

    _ensure_ready()
    with _db() as connection:
        connection.execute(
            """
            INSERT OR REPLACE INTO agent_runs
                (run_id, user_id, created_at, steps, session_tokens,
                 session_cost_cny, stop_reason, breaker)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(run_id),
                str(user_id),
                _now(),
                int(steps),
                int(session_tokens),
                float(session_cost_cny),
                str(stop_reason),
                breaker,
            ),
        )


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


def _owner_matches(row: sqlite3.Row, current_user: Optional[str]) -> bool:
    if not current_user:
        return False
    return str(row["user_id"]) == str(current_user).strip()


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
    if order is None or not current_user or order.user_id != str(current_user).strip():
        return _fail(tool, NOT_FOUND_MESSAGE)

    return {"ok": True, "tool": tool, "order": order.to_dict()}


def cancel_order(args: Any = None, current_user: Optional[str] = None) -> Dict[str, Any]:
    """取消订单。只能取消自己名下、尚未取消/退款的订单。"""

    tool = "cancel_order"
    parsed = _as_args(args)
    order_id = _order_id(parsed)
    if order_id is None:
        return _fail(tool, "缺少 order_id，无法取消订单")

    _ensure_ready()
    with _db() as connection:
        row = connection.execute(
            "SELECT * FROM orders WHERE order_id = ?", (order_id,)
        ).fetchone()
        if row is None or not _owner_matches(row, current_user):
            return _fail(tool, NOT_FOUND_MESSAGE)

        if row["status"] == ORDER_STATUS_REFUNDED:
            return _fail(
                tool, "订单已退款，无法取消", order=_row_to_order(row).to_dict()
            )
        if row["status"] == ORDER_STATUS_CANCELLED:
            return _fail(
                tool,
                "订单已经取消过了，无需重复操作",
                order=_row_to_order(row).to_dict(),
            )

        now = _now()
        cursor = connection.execute(
            """
            UPDATE orders
               SET status = ?, updated_at = ?
             WHERE order_id = ? AND status = ?
            """,
            (ORDER_STATUS_CANCELLED, now, order_id, ORDER_STATUS_PAID),
        )
        if cursor.rowcount != 1:
            fresh = connection.execute(
                "SELECT * FROM orders WHERE order_id = ?", (order_id,)
            ).fetchone()
            return _fail(
                tool,
                "订单状态已变化，请刷新后重试",
                order=_row_to_order(fresh).to_dict() if fresh is not None else None,
            )

        updated = connection.execute(
            "SELECT * FROM orders WHERE order_id = ?", (order_id,)
        ).fetchone()

    return {
        "ok": True,
        "tool": tool,
        "order": _row_to_order(updated).to_dict(),
        "message": "订单已取消",
    }


def refund_order(args: Any = None, current_user: Optional[str] = None) -> Dict[str, Any]:
    """退款。只有"已取消且未退过款"的订单才能退款。"""

    tool = "refund_order"
    parsed = _as_args(args)
    order_id = _order_id(parsed)
    if order_id is None:
        return _fail(tool, "缺少 order_id，无法退款")

    _ensure_ready()
    with _db() as connection:
        row = connection.execute(
            "SELECT * FROM orders WHERE order_id = ?", (order_id,)
        ).fetchone()
        if row is None or not _owner_matches(row, current_user):
            return _fail(tool, NOT_FOUND_MESSAGE)

        if int(row["refund_count"]) > 0 or row["status"] == ORDER_STATUS_REFUNDED:
            return _fail(
                tool,
                "该订单已经退过款，同一订单只能退款一次",
                order=_row_to_order(row).to_dict(),
            )
        if row["status"] != ORDER_STATUS_CANCELLED:
            return _fail(
                tool,
                "订单尚未取消，不能退款；请先取消订单",
                order=_row_to_order(row).to_dict(),
            )

        now = _now()
        cursor = connection.execute(
            """
            UPDATE orders
               SET status = ?,
                   refund_count = refund_count + 1,
                   refunded_at = ?,
                   updated_at = ?
             WHERE order_id = ?
               AND status = ?
               AND refund_count = 0
            """,
            (ORDER_STATUS_REFUNDED, now, now, order_id, ORDER_STATUS_CANCELLED),
        )
        if cursor.rowcount != 1:
            fresh = connection.execute(
                "SELECT * FROM orders WHERE order_id = ?", (order_id,)
            ).fetchone()
            return _fail(
                tool,
                "订单状态已变化，请刷新后重试",
                order=_row_to_order(fresh).to_dict() if fresh is not None else None,
            )

        updated = connection.execute(
            "SELECT * FROM orders WHERE order_id = ?", (order_id,)
        ).fetchone()
        order = _row_to_order(updated)

    return {
        "ok": True,
        "tool": tool,
        "order": order.to_dict(),
        "refund": {
            "order_id": order.order_id,
            "user_id": order.user_id,
            "amount": order.amount,
            "refunded_at": order.refunded_at,
            "refund_count": order.refund_count,
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
