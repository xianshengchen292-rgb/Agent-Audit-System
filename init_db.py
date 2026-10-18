"""初始化本地订单数据库 orders.db。

用法::

    python init_db.py            # 建表 + 补注入初始数据（已有数据保留）
    python init_db.py --reset    # 删表重建，恢复 o1/o2 的初始状态

表结构：
    orders：order_id / user_id / title / amount / status / refund_count
            refunded_at / created_at / updated_at
    agent_runs：run_id / user_id / steps / session_tokens / session_cost_cny
                stop_reason / breaker / created_at
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:  # 支持从任意目录执行
    sys.path.insert(0, str(PROJECT_ROOT))

from app.tools import DB_PATH, get_state, init_db


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="初始化订单 SQLite 数据库")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="删除并重建 orders 表，重新注入初始订单（会清空已有变更）",
    )
    args = parser.parse_args(argv)

    path = init_db(reset=args.reset)
    state = get_state()

    print(f"数据库：{path}")
    print(f"模式：{'重置并重建' if args.reset else '建表 / 补数据'}")
    print("审计表：agent_runs（记录每次 Agent 会话的步数 / 成本 / 熔断）")
    print(f"订单数：{len(state['orders'])}，退款总数：{state['refund_count']}")
    print("订单列表：")
    for order_id, order in state["orders"].items():
        print(
            f"  {order_id}  归属={order['user_id']}  状态={order['status']:<9} "
            f"refund_count={order['refund_count']}  {order['title']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
