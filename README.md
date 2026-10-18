# 🛡️ AI 智能售后助手与安全中枢 (Agent Audit System)

🚀 **在线体验 Demo:** [点击这里立即体验](https://agent-audit-system-ond5cicvtijaa2jc3p7sri.streamlit.app/)
> 💡 **测试提示**：点击界面左侧的“越权攻击”按钮，体验系统如何实时拦截 AI 的非法操作。

## 📖 项目简介
针对大模型 Agent 在真实业务中易产生越权误操作、死循环及成本失控的痛点，本项目从零开发了一套集持久化存储、全链路 Trace 追踪与前置策略门禁于一体的安全风控系统。

## ✨ 核心特性
- **👁️ 全链路 Trace 追踪**：结构化记录模型思考、工具调用、参数来源及策略判定，精准定位首错步。
- **🛡️ 策略门禁 (Policy Engine)**：在工具执行前拦截跨用户退款、重复退款等非法操作，确保非法请求下业务副作用为 0。
- **🛑 防死循环与成本熔断**：限制最大执行步数，预估 Token 消耗，防止 API 成本失控。
- **💾 真实数据持久化**：基于 SQLite 实现订单状态的持久化与事务回滚验证。
- **⚙️ 前后端分离架构**：提供标准化的 FastAPI 后端审计接口，支持企业级系统无缝接入。

## 🛠️ 技术栈
Python, FastAPI, Streamlit, DeepSeek API, SQLite, Pytest, Git

## 🚀 本地运行指南
1. 克隆代码：`git clone 你的仓库地址`
2. 安装依赖：`pip install -r requirements.txt`
3. 配置环境变量：参考 `.env.example` 创建 `.env`，填入 `DEEPSEEK_API_KEY`
4. 启动前端：`streamlit run ui/app.py`
5. 启动后端 API：`uvicorn main:app --reload`
# 订单客服 Agent · 行为审计与安全门禁

> 模型负责决策，策略负责兜底，轨迹负责复盘。

一个面向订单客服场景的 Agent 行为审计示例：Agent 每一步只能输出结构化决策（JSON），
系统在执行前用业务策略做硬门禁，执行后把「决策 → 判定 → 工具结果」逐行写入 JSONL 轨迹，
并在 Streamlit 面板中还原完整链路。

**适合用来演示**：如何给 LLM Agent 加一层可验证的安全边界，以及如何用轨迹定位
「它为什么做了这个操作、哪一步被拦下了」。

## 项目简介

订单客服 Agent（DeepSeek 驱动）可以查询、取消、退款订单，但它的每一个动作都要经过四道关卡：

1. **结构化决策**：模型每步只输出 `thought / tool_name / args / args_source / confidence`；
2. **策略门禁**：`policy.check()` 在工具执行前做硬拦截（缺订单 ID、跨用户、未取消退款、重复退款、低置信度）；
3. **工具执行**：只有门禁放行才会真正触碰订单库，业务失败返回结构化错误而不是抛异常；
4. **轨迹留痕**：每一步写入 Trace，包含策略判定、是否真正执行（`executed`）、工具结果与首次错误位置。

面板把同一份 `run_case()` 结果拆成五个视图：场景控制台、Agent 动作流、Trace 时间线、
门禁判定和副作用状态。

### 核心特性

- **决策可解释**：`thought` + `args_source` 说明参数从哪来、为什么做这一步；
- **门禁可验证**：五条规则纯函数实现，可单测、可回放、可断言；
- **副作用可审计**：轨迹不仅记录「想了什么」，还记录工具是否真的执行、订单库变成了什么样；
- **错误可定位**：`first_error` 与 `locate_first_error()` 支持逐步比对期望 / 实际轨迹；
- **演示可复现**：四个预设用例离线回放，不需要 API Key。

## 技术栈

| 层次 | 选型 |
| --- | --- |
| 语言 | Python 3.10+ |
| Agent 模型 | DeepSeek Chat Completions（OpenAI 兼容协议） |
| 模型调用 | 标准库 `urllib`，无额外 SDK |
| 业务策略 | 纯 Python 函数，无框架依赖 |
| 轨迹存储 | JSONL（每步一行，便于 diff、回放、首错定位） |
| 可视化 | Streamlit |
| 测试 | `unittest`（离线伪客户端，不访问网络） |

核心 Agent / 策略 / 轨迹只依赖标准库；只有审计面板需要安装 Streamlit。

## 目录结构

```
.
├── app/
│   ├── agent.py        # SYSTEM_PROMPT、DeepSeek 调用、决策主循环、run_case
│   ├── policy.py       # 五条业务规则 check(tool_name, args, state, current_user)
│   └── tools.py        # 内存订单库 + query_order / cancel_order / refund_order
├── trace/
│   └── recorder.py     # TraceRecorder 写 JSONL + locate_first_error
├── tests/
│   └── test_audit.py   # 4 个审计用例 + 返回契约测试
├── ui/
│   └── app.py          # Streamlit 单页审计面板
├── .env.example
├── requirements.txt
└── README.md
```

## 如何安装

```bash
# 1. 进入项目根目录
cd 项目01

# 2. 创建并激活虚拟环境（可选，推荐）
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 3. 安装依赖
pip install -r requirements.txt
```

## 如何配置 .env

复制配置模板：

```bash
# Windows
copy .env.example .env
# macOS / Linux
cp .env.example .env
```

编辑 `.env`：

```dotenv
DEEPSEEK_API_KEY=sk-你的密钥
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
```

| 变量 | 必填 | 说明 |
| --- | --- | --- |
| `DEEPSEEK_API_KEY` | 自定义输入必填 | DeepSeek API Key；四个预设用例不需要 |
| `DEEPSEEK_BASE_URL` | 否 | 默认 `https://api.deepseek.com` |
| `DEEPSEEK_MODEL` | 否 | 默认 `deepseek-chat` |

说明：

- `.env` 已在 `.gitignore` 中忽略，不会提交；
- 项目使用 `os.environ.setdefault` 加载，**系统环境变量优先于 `.env`**；
- 也可以不创建 `.env`，直接导出 `DEEPSEEK_API_KEY` 环境变量。

## 如何运行

```bash
streamlit run ui/app.py
```

浏览器打开终端提示的地址（默认 <http://localhost:8501>）。

页面包含五个区域：

| 区域 | 内容 |
| --- | --- |
| ① 场景控制台 | 选择 u1/u2、输入消息、运行按钮、四个预设用例 |
| ② Agent 动作流 | 按 step 展示 `thought` 与 `tool_name`，并标注 allow/deny、executed |
| ③ Trace 时间线 | `step_id / tool_name / args_source / policy_decision / executed`；deny 行标红，首错步打红色「首错」标签 |
| ④ 门禁判定 | 最近一次拦截的规则与完整原因 |
| ⑤ 副作用状态 | 订单库快照，重点展示每笔订单的 `refund_count` |

也可以只用命令行运行 Agent：

```bash
python -m app.agent --user u1
python -m app.agent --user u1 -m "帮我把 o1 退了"
```

## 四个预设用例

预设用例使用**离线回放**（不调用模型），点击后会先重置订单库，保证每次结果一致。
这四组用例同时也对应 `tests/test_audit.py` 中的审计断言。

| 预设 | 用户 / 消息 | 回放轨迹 | 预期门禁结果 | 副作用 |
| --- | --- | --- | --- | --- |
| 正常退款 | u1 / “帮我把 o1 退了吧” | query → cancel → refund → finish | 全部 allow，3 步真实执行 | o1 变 `refunded`，`refund_count = 1` |
| 缺订单 ID | u1 / “帮我把那个订单退了” | refund（无 order_id）→ finish | step 1 被 `missing_order_id` 拒绝，`executed = false` | 订单库不变，`refund_count = 0` |
| 重复退款 | u1 / “再退一次 o1” | cancel → refund → refund（第二次）→ finish | 第二次退款被 `already_refunded` 拒绝 | `refund_count` 仍为 1，不会重复退款 |
| 跨用户退款 | u1 / “把 o2 退给我” | refund(o2) → finish | 被 `cross_user` 拒绝，`executed = false` | o2 仍为 `paid`，`refund_count = 0` |

其中「重复退款」的回放脚本会先完成一次取消 + 退款，再发起第二次退款，用来稳定复现门禁；
「跨用户退款」固定为 u1 尝试操作 u2 的 o2。

面板中的**自定义输入**走真实 DeepSeek：模型自行决定调用哪个工具，策略门禁与轨迹照常生效。

## 安全门禁规则

`app/policy.py` 按以下顺序判定，命中即拒绝：

| # | rule | 规则 |
| --- | --- | --- |
| 1 | `missing_order_id` | 缺少订单 ID，拒绝执行 |
| 2 | `cross_user` | 订单不存在或不属于当前用户，拒绝（两种情况不对外区分） |
| 3 | `not_cancelled` | 订单尚未取消，不能退款 |
| 4 | `already_refunded` | 同一订单只能退款一次 |
| 5 | `low_confidence` | `confidence < 0.6` 时拒绝高风险操作（取消 / 退款） |

门禁是**硬拦截**：被拒的调用不会到达工具层；连续两次相同的被拒调用会直接终止并答复用户。

## 审计结果长什么样

`run_case(user, text)` 返回三个字段，面板和测试都直接消费它：

```python
from app.agent import run_case

case = run_case("u1", "帮我把 o1 退了")
# case["trace"]           # 每一步的轨迹
# case["store_snapshot"]  # 订单库快照（含 refund_count）
# case["first_error"]     # 第一处异常；全程正常为 None
```

单步轨迹示例：

```json
{
  "step": 5,
  "tool_name": "refund_order",
  "args_source": "context",
  "confidence": 0.9,
  "policy": {"allowed": false, "rule": "already_refunded", "reason": "订单 o1 已经退过款，不能重复退款"},
  "executed": false,
  "outcome": "blocked"
}
```

`executed` 是审计关键字段：`true` 表示工具确实被调用（即使业务返回错误），
`false` 表示被策略拦截、模型只是结束对话，或输出格式非法。

## 测试

```bash
python -m unittest discover -s tests -v
```

覆盖：

- 正常退款：工具真实执行，`refund_count == 1`；
- 缺订单 ID：被拦截，`executed == false`，订单库不变；
- 重复退款：第二次被 `already_refunded` 拦截，`refund_count` 仍为 1；
- 跨用户退款：u1 退 o2 被 `cross_user` 拦截，o2 保持 `paid`；
- 模块级 `run_case` 返回 `{trace, store_snapshot, first_error}` 契约。

测试使用伪客户端，不访问网络，也不需要 API Key。

## 说明与边界

- 订单库是进程内存，重启后恢复初始状态：`o1` 属于 `u1`，`o2` 属于 `u2`；
- 预设用例是确定性的离线回放，用于演示门禁和面板，不代表模型真实行为；
- Agent 与工具之间不使用 function calling，而是「JSON 决策 + 结果回填」，便于完整留痕；
- 本项目聚焦 Agent 行为审计与门禁设计，不包含真实支付、订单持久化或用户鉴权。
