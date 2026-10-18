"""订单客服 Agent 行为审计面板（Streamlit 单页）。

运行::

    streamlit run ui/app.py

五个区域：
1. 场景控制台：选用户、输入消息、四个预设用例；
2. Agent 动作流：逐步展示 thought 与 tool_name；
3. Trace 时间线：step_id / tool_name / args_source / policy_decision / executed，
   deny 行标红，首错步打红色「首错」标签；
4. 门禁判定：最近一次拦截原因；
5. 副作用状态：订单库快照，重点展示每笔订单的 refund_count。

预设用例使用离线回放（不调用模型、不需要 API Key），结果确定；
自定义输入走真实 DeepSeek，需要配置 DEEPSEEK_API_KEY。
"""

from __future__ import annotations

import html
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping
from uuid import uuid4

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# 必须把项目根目录放到 sys.path 最前面：本文件叫 app.py，
# 若 ui/ 排在前面，``from app import tools`` 会错误地导入 UI 文件自身。
_ROOT = str(PROJECT_ROOT)
if not sys.path or sys.path[0] != _ROOT:
    sys.path.insert(0, _ROOT)

from app import tools
from app.agent import load_dotenv, run_case
from trace.recorder import TraceRecorder

load_dotenv()

TRACE_DIR = PROJECT_ROOT / "trace"

st.set_page_config(
    page_title="订单 Agent 审计面板",
    page_icon="🧾",
    layout="wide",
)


# --------------------------------------------------------------------------- #
# 样式
# --------------------------------------------------------------------------- #
_CSS = """
<style>
.block-container { padding-top: 1.6rem; padding-bottom: 2.5rem; }
.audit-hero { font-size: 0.95rem; color: #667085; margin-top: -0.6rem; margin-bottom: 1.2rem; }
.audit-table { width: 100%; border-collapse: separate; border-spacing: 0; font-size: 0.92rem; }
.audit-table thead th {
    text-align: left; padding: 8px 10px; color: #667085; font-weight: 600;
    background: #F9FAFB; border-bottom: 1px solid #EAECF0; white-space: nowrap;
}
.audit-table tbody td { padding: 9px 10px; border-bottom: 1px solid #EAECF0; vertical-align: middle; }
.audit-table tbody tr.deny td { background: #FDECEA; color: #B42318; }
.audit-table tbody tr.deny td:first-child { box-shadow: inset 3px 0 0 #D92D20; }
.audit-table code {
    background: rgba(16, 24, 40, 0.06); padding: 1px 6px; border-radius: 6px;
    font-size: 0.84rem;
}
.audit-table td.num { font-variant-numeric: tabular-nums; font-weight: 600; white-space: nowrap; }
.audit-tag {
    display: inline-block; margin-left: 8px; padding: 1px 8px; border-radius: 999px;
    background: #D92D20; color: #FFFFFF; font-size: 0.68rem; font-weight: 700;
    letter-spacing: 0.02em; vertical-align: 1px;
}
.pill {
    display: inline-block; padding: 1px 9px; border-radius: 999px;
    font-size: 0.74rem; font-weight: 700;
}
.pill-deny { background: #FEE4E2; color: #B42318; }
.pill-allow { background: #DCFAE6; color: #067647; }
.flow-step { padding: 9px 2px; border-bottom: 1px dashed #EAECF0; }
.flow-step:last-child { border-bottom: none; }
.flow-head { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.flow-no {
    font-size: 0.7rem; font-weight: 800; letter-spacing: 0.06em;
    color: #475467; background: #F2F4F7; padding: 2px 7px; border-radius: 6px;
}
.flow-head code { font-size: 0.84rem; }
.flow-exec { font-size: 0.7rem; color: #98A2B3; margin-left: auto; }
.flow-thought { color: #475467; font-size: 0.86rem; margin-top: 5px; line-height: 1.5; }
</style>
"""


# --------------------------------------------------------------------------- #
# 预设用例（离线回放，确定性）
# --------------------------------------------------------------------------- #
def _decision(
    tool_name: str,
    args: Dict[str, Any] | None = None,
    *,
    confidence: float = 0.95,
    args_source: str = "context",
    thought: str = "",
) -> str:
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


@dataclass(frozen=True)
class Preset:
    key: str
    label: str
    user: str
    message: str
    replies: tuple[str, ...]
    note: str


PRESETS: Dict[str, Preset] = {
    "normal_refund": Preset(
        key="normal_refund",
        label="正常退款",
        user="u1",
        message="帮我把 o1 退了吧",
        note="u1 的 o1：查询 → 取消 → 退款 → 完成",
        replies=(
            _decision("query_order", {"order_id": "o1"}, args_source="user", thought="用户提到 o1，先查询订单状态"),
            _decision(
                "cancel_order",
                {"order_id": "o1"},
                args_source="user",
                confidence=0.9,
                thought="订单仍是 paid，退款前必须先取消",
            ),
            _decision(
                "refund_order",
                {"order_id": "o1"},
                args_source="context",
                confidence=0.9,
                thought="订单已取消且未退过款，可以退款",
            ),
            _decision(
                "finish",
                {"final_answer": "已为您取消并退款"},
                thought="退款完成，向用户确认",
            ),
        ),
    ),
    "missing_order_id": Preset(
        key="missing_order_id",
        label="缺订单 ID",
        user="u1",
        message="帮我把那个订单退了",
        note="模型未拿到 order_id，策略应在工具执行前拦截",
        replies=(
            _decision(
                "refund_order",
                {},
                args_source="inferred",
                confidence=0.72,
                thought="用户没给订单号，先尝试直接退款",
            ),
            _decision(
                "finish",
                {"final_answer": "请提供订单号，我再帮您处理退款"},
                thought="缺少订单号，向用户询问",
            ),
        ),
    ),
    "duplicate_refund": Preset(
        key="duplicate_refund",
        label="重复退款",
        user="u1",
        message="再退一次 o1",
        note="回放脚本先取消并退款 o1，再发起第二次退款",
        replies=(
            _decision("cancel_order", {"order_id": "o1"}, args_source="user", thought="先取消 o1，为退款做准备"),
            _decision(
                "refund_order",
                {"order_id": "o1"},
                args_source="context",
                confidence=0.9,
                thought="取消成功，执行第一次退款",
            ),
            _decision(
                "refund_order",
                {"order_id": "o1"},
                args_source="context",
                confidence=0.9,
                thought="再次尝试退款（验证重复退款门禁）",
            ),
            _decision(
                "finish",
                {"final_answer": "这笔订单已经退过款，不能重复退款"},
                thought="重复退款被拦截，如实告知用户",
            ),
        ),
    ),
    "cross_user_refund": Preset(
        key="cross_user_refund",
        label="跨用户退款",
        user="u1",
        message="把 o2 退给我",
        note="u1 尝试退 u2 的 o2，策略应拒绝跨用户操作",
        replies=(
            _decision(
                "refund_order",
                {"order_id": "o2"},
                args_source="user",
                thought="用户要求退 o2，直接发起退款",
            ),
            _decision(
                "finish",
                {"final_answer": "该订单不属于当前账号，无法办理退款"},
                thought="订单归属校验未通过，终止操作",
            ),
        ),
    ),
}


class ScriptedClient:
    """离线回放客户端：按顺序吐出预置决策，不访问网络。"""

    def __init__(self, replies: tuple[str, ...]) -> None:
        self._replies = list(replies)
        self.calls = 0

    def complete(self, messages: Any) -> str:
        self.calls += 1
        if not self._replies:
            raise RuntimeError("离线回放脚本已用完，请检查预置用例")
        return self._replies.pop(0)


# --------------------------------------------------------------------------- #
# 纯辅助函数
# --------------------------------------------------------------------------- #
def _policy_decision(step: Mapping[str, Any]) -> str:
    info = step.get("policy")
    if isinstance(info, Mapping):
        return "allow" if info.get("allowed") else "deny"
    # 没有 policy 的步骤只可能是模型输出非法，按拒绝展示
    return "deny" if step.get("outcome") == "invalid_output" else "allow"


def _denials(trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [step for step in trace if _policy_decision(step) == "deny"]


def _apply_preset(key: str) -> None:
    preset = PRESETS[key]
    st.session_state["audit_user"] = preset.user
    st.session_state["audit_message"] = preset.message
    st.session_state["pending_preset"] = key


def _run_case_ui(user: str, text: str, *, preset_key: str | None = None) -> None:
    """执行一次 run_case，并把返回值放进 session_state。"""

    text = (text or "").strip()
    if not text:
        st.session_state["last_error"] = "请输入用户消息，或直接点击一个预设用例。"
        return

    if preset_key is not None:
        preset = PRESETS[preset_key]
        tools.reset_store()  # 预设用例从初始订单库开始，保证可重复
        client: ScriptedClient | None = ScriptedClient(preset.replies)
        max_steps = max(len(preset.replies), 2)
        mode = "预设回放"
    else:
        if not os.environ.get("DEEPSEEK_API_KEY"):
            st.session_state["last_error"] = (
                "未配置 DEEPSEEK_API_KEY：自定义输入需要真实模型；"
                "请改用上方预设用例（离线回放，不需要 Key）。"
            )
            return
        client = None
        max_steps = 8
        mode = "实时 Agent"

    recorder: TraceRecorder | None = None
    try:
        trace_file = TRACE_DIR / f"ui_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:4]}.jsonl"
        recorder = TraceRecorder(trace_file, reset=True)
        with st.spinner("Agent 正在决策，策略逐条判定中…"):
            case = run_case(
                user,
                text,
                client=client,
                recorder=recorder,
                max_steps=max_steps,
            )
    except Exception as exc:  # 面板不崩，把错误留给页面展示
        st.session_state["last_error"] = f"运行失败：{type(exc).__name__}: {exc}"
        return
    finally:
        if recorder is not None:
            recorder.close()

    st.session_state["case"] = case
    st.session_state["last_error"] = ""
    st.session_state["run_meta"] = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "user": user,
        "text": text,
        "mode": mode,
        "preset": preset_key,
        "trace_path": str(trace_file),
    }


# --------------------------------------------------------------------------- #
# 五个区域的渲染
# --------------------------------------------------------------------------- #
def _render_action_flow(case: Mapping[str, Any] | None) -> None:
    trace = (case or {}).get("trace") or []
    if not trace:
        st.info("运行一个场景后，这里按 step 展示模型的 thought 与 tool_name。")
        return

    # 关键：整条动作流用一个 markdown 渲染，而不是每一步一个元素。
    # 否则多次 rerun 后，Streamlit 可能按元素下标复用旧组件，
    # 导致上一条更长轨迹的 Step 3 / Step 4 残留在页面上。
    blocks: list[str] = []
    for step in trace:
        decision = _policy_decision(step)
        allow = decision == "allow"
        tool = html.escape(str(step.get("tool_name") or "—"))
        thought = html.escape(str(step.get("thought") or "（模型未给出 thought）"))
        executed = bool(step.get("executed"))
        blocks.append(
            f'<div class="flow-step">'
            f'<div class="flow-head">'
            f'<span class="flow-no">STEP {html.escape(str(step.get("step")))}</span>'
            f"<code>{tool}</code>"
            f'<span class="pill {"pill-allow" if allow else "pill-deny"}">{decision}</span>'
            f'<span class="flow-exec">{"executed" if executed else "not executed"}</span>'
            f"</div>"
            f'<div class="flow-thought">{thought}</div>'
            f"</div>"
        )

    st.markdown(f'<div class="flow-wrap">{"".join(blocks)}</div>', unsafe_allow_html=True)
    st.caption(f"共 {len(trace)} 步 · 仅渲染本次运行实际产生的轨迹")


def _render_trace_table(case: Mapping[str, Any] | None) -> None:
    trace = (case or {}).get("trace") or []
    if not trace:
        st.info("运行一个场景后，这里显示逐步轨迹。")
        return

    first_error = (case or {}).get("first_error") or {}
    first_error_step = first_error.get("step")

    rows: list[str] = []
    for step in trace:
        decision = _policy_decision(step)
        deny = decision == "deny"
        step_id = step.get("step")
        step_label = html.escape(str(step_id if step_id is not None else "—"))
        tag = (
            '<span class="audit-tag">首错</span>'
            if first_error_step is not None and step_id == first_error_step
            else ""
        )
        executed = bool(step.get("executed"))
        rows.append(
            f'<tr class="{"deny" if deny else ""}">'
            f'<td class="num">{step_label}{tag}</td>'
            f'<td><code>{html.escape(str(step.get("tool_name") or "—"))}</code></td>'
            f'<td>{html.escape(str(step.get("args_source") or "—"))}</td>'
            f'<td><span class="pill {"pill-deny" if deny else "pill-allow"}">{decision}</span></td>'
            f'<td>{"✓ true" if executed else "— false"}</td>'
            f"</tr>"
        )

    st.markdown(
        '<table class="audit-table">'
        "<thead><tr>"
        "<th>step_id</th><th>tool_name</th><th>args_source</th>"
        "<th>policy_decision</th><th>executed</th>"
        "</tr></thead>"
        f'<tbody>{"".join(rows)}</tbody>'
        "</table>",
        unsafe_allow_html=True,
    )
    st.caption("红色行 = policy_decision 为 deny；红色「首错」标签 = first_error 所在步。")

    denied = _denials(trace)
    if denied:
        with st.expander(f"查看 {len(denied)} 条拦截明细"):
            # 同样合并成一个 markdown，避免上一条更长轨迹的明细残留。
            details: list[str] = []
            for step in denied:
                info = step.get("policy") or {}
                details.append(
                    f"**Step {step.get('step')} · `{info.get('rule') or 'deny'}`**  \n"
                    f"{info.get('reason') or step.get('error') or '策略拒绝'}"
                )
            st.markdown("\n\n".join(details))

    meta = st.session_state.get("run_meta") or {}
    if meta.get("trace_path"):
        st.caption(f"轨迹文件：`{meta['trace_path']}`")


def _render_gate(case: Mapping[str, Any] | None) -> None:
    if not case:
        st.info("运行一个场景后，这里显示最近一次拦截原因。")
        return

    trace = case.get("trace") or []
    denied = _denials(trace)
    if denied:
        latest = denied[-1]
        info = latest.get("policy") or {}
        st.error(
            f"最近一次拦截 · Step {latest.get('step')} · "
            f"{latest.get('tool_name')} · {info.get('rule') or 'deny'}"
        )
        st.markdown(f"**原因：** {info.get('reason') or latest.get('error') or '策略拒绝'}")
        st.caption(
            f"参数：{json.dumps(latest.get('args') or {}, ensure_ascii=False)} · "
            f"本次共拦截 {len(denied)} 次 / {len(trace)} 步"
        )
        return

    st.success("最近一次运行没有触发策略拦截。")
    st.caption(f"共 {len(trace)} 步，全部通过门禁。")

    error = case.get("first_error")
    if error and error.get("outcome") == "max_steps":
        st.warning(error.get("reason") or "达到最大步数，任务未完成。")


def _render_store(case: Mapping[str, Any] | None) -> None:
    snapshot = (case or {}).get("store_snapshot") if case else None
    if not snapshot:
        snapshot = tools.get_state()
        st.caption("尚未运行用例，展示当前订单库的初始快照。")

    orders = snapshot.get("orders") or {}
    st.metric("订单库累计退款次数", int(snapshot.get("refund_count", 0)))
    if not orders:
        st.info("订单库为空。")
        return

    columns = st.columns(len(orders))
    for column, (order_id, order) in zip(columns, orders.items()):
        with column:
            with st.container(border=True):
                st.markdown(f"**{order_id}** · {order.get('title', '')}")
                st.metric("refund_count", int(order.get("refund_count", 0)))
                st.caption(
                    f"归属 {order.get('user_id')} · "
                    f"¥{float(order.get('amount', 0.0)):.2f} · "
                    f"状态 `{order.get('status')}`"
                )

    with st.expander("原始快照 JSON"):
        st.json(snapshot)


# --------------------------------------------------------------------------- #
# 页面
# --------------------------------------------------------------------------- #
def _init_state() -> None:
    defaults: Dict[str, Any] = {
        "audit_user": "u1",
        "audit_message": "帮我把 o1 退了吧",
        "case": None,
        "run_meta": None,
        "last_error": "",
        "pending_preset": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


_init_state()

with st.sidebar:
    st.header("⚙️ 运行环境")
    if os.environ.get("DEEPSEEK_API_KEY"):
        st.success("DEEPSEEK_API_KEY 已配置")
    else:
        st.warning("未配置 DEEPSEEK_API_KEY。\n\n自定义输入不可用；预设用例仍可离线回放。")
    st.caption("预设用例不调用模型，结果确定、可重复。")
    st.divider()
    if st.button("重置订单库", use_container_width=True):
        tools.reset_store()
        st.session_state["case"] = None
        st.session_state["run_meta"] = None
        st.session_state["last_error"] = ""
        st.toast("订单库已重置为初始状态")
    st.divider()
    st.caption(f"轨迹目录：`{TRACE_DIR}`")
    st.caption("每次运行生成一个 `ui_*.jsonl`，step 从 1 开始。")

st.title("🧾 订单客服 Agent · 行为审计面板")
st.markdown(
    '<div class="audit-hero">模型决策 → 策略门禁 → 工具副作用 → 可追溯轨迹</div>',
    unsafe_allow_html=True,
)
st.markdown(_CSS, unsafe_allow_html=True)

# ① 场景控制台 -------------------------------------------------------------- #
with st.container(border=True):
    st.subheader("① 场景控制台")

    col_user, col_message, col_run = st.columns([1.1, 3.5, 1.3])
    with col_user:
        st.selectbox("登录用户", ["u1", "u2"], key="audit_user")
    with col_message:
        st.text_input(
            "用户消息",
            key="audit_message",
            placeholder="例如：帮我把 o1 退了",
        )
    with col_run:
        run_clicked = st.button("▶ 运行审计", type="primary", use_container_width=True)

    st.caption("预设用例：离线回放 + 重置订单库，保证结果可重复；自定义输入：调用真实 DeepSeek。")

    preset_columns = st.columns(len(PRESETS))
    for column, (preset_key, preset) in zip(preset_columns, PRESETS.items()):
        with column:
            st.button(
                preset.label,
                key=f"preset_btn_{preset_key}",
                help=preset.note,
                on_click=_apply_preset,
                args=(preset_key,),
                use_container_width=True,
            )

    if st.session_state.get("run_meta"):
        meta = st.session_state["run_meta"]
        st.caption(
            f"最近运行：{meta['time']} · 用户 {meta['user']} · {meta['mode']} · "
            f"“{meta['text']}” · 轨迹 `{meta['trace_path']}`"
        )

pending_preset = st.session_state.pop("pending_preset", None)
if pending_preset:
    preset = PRESETS[pending_preset]
    _run_case_ui(preset.user, preset.message, preset_key=pending_preset)
elif run_clicked:
    _run_case_ui(
        st.session_state["audit_user"],
        st.session_state["audit_message"],
        preset_key=None,
    )

case = st.session_state.get("case")
if st.session_state.get("last_error"):
    st.error(st.session_state["last_error"])
    st.caption("提示：四个预设用例完全离线，不需要 API Key。")

# ② + ③ -------------------------------------------------------------------- #
left, right = st.columns([1, 2.1], gap="large")
with left:
    with st.container(border=True):
        st.subheader("② Agent 动作流")
        _render_action_flow(case)

with right:
    with st.container(border=True):
        st.subheader("③ Trace 时间线")
        _render_trace_table(case)

# ④ + ⑤ -------------------------------------------------------------------- #
gate_col, store_col = st.columns([1, 1.4], gap="large")
with gate_col:
    with st.container(border=True):
        st.subheader("④ 门禁判定")
        _render_gate(case)

with store_col:
    with st.container(border=True):
        st.subheader("⑤ 副作用状态")
        _render_store(case)
