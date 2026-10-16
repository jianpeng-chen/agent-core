# Agent Core · Python

一个可以读懂、运行和逐步改进的最小 Agent 核心。提供会话、完整轮次上下文、模型协议、带校验与权限的工具执行、OpenAI-compatible Chat Completions 适配器和命令行。

支持 Python 3.12+。此分支按 Python 工程方式重新设计；TypeScript 原版位于 `main` 的 `6440371`。

## 目录

```text
src/core/        核心模块和命令行入口
tests/           场景与模块测试
pyproject.toml   项目、依赖和检查配置
uv.lock          依赖锁文件
```

安装项目使用名称 `agent-core`，Python 导入使用 `from core import ...`。

## 先跑通一个场景

安装 [uv](https://docs.astral.sh/uv/getting-started/installation/) 后，在项目根目录运行：

```bash
uv sync --locked --dev
uv run agent-core --demo --once "你好，Python"
```

离线演示不需要 API key、模型服务或网络请求。`DemoModel` 固定请求 `echo` 工具，工具执行、校验、历史记录和循环结束都走真实核心流程；它不模拟智能推理。首次安装依赖需要网络。

交互方式：

```bash
uv run agent-core --demo
```

输入 `/exit`、`/quit` 或 EOF 退出，Ctrl+C 退出码为 130。命令行也支持 `uv run python -m core`。

## 连接实际模型

```bash
export OPENAI_API_KEY="你的密钥"
export OPENAI_MODEL="你的模型名称"
export OPENAI_BASE_URL="https://api.openai.com/v1"
uv run agent-core --once "调用 echo 回传 hello"
```

环境变量不从 `.env` 自动加载。`OPENAI_BASE_URL` 可填写服务根路径（通常含 `/v1`）或完整 `/chat/completions` 地址；不含 query/fragment。默认模型名沿用旧版的 `gpt-4.1-mini`，可按接入服务覆盖。

| 配置 | CLI 覆盖 | 默认值 / 语义 |
| --- | --- | --- |
| `OPENAI_API_KEY` | 无 | 远端模式必填 |
| `OPENAI_MODEL` | `--model` | `gpt-4.1-mini` |
| `OPENAI_BASE_URL` | `--base-url` | `https://api.openai.com/v1` |
| `OPENAI_TIMEOUT` | `--timeout` | 30 秒，各 HTTP 阶段超时，须为有限正数 |
| `OPENAI_TEMPERATURE` | `--temperature` | 不传；仅在服务支持时显式设置 |
| `AGENT_MAX_STEPS` | `--max-steps` | 最多 8 批工具调用，允许 0 |
| `AGENT_CONTEXT_TURNS` | `--context-turns` | 最近 12 个完整用户轮次，必须至少 1 |
| `AGENT_SYSTEM_PROMPT` | 无 | 通用助手提示词 |
| `AGENT_USER_ID` | 无 | `repl_user` |
| 无 | `--parallel-tools` | 默认串行；开启后并行执行同批工具并保持结果记录顺序 |

一次性模式在模型错误或模型拒绝遵守步数上限时返回 1；配置错误返回 2。

## 在应用中定义工具

```python
import asyncio
import os

from core import Agent, OpenAIChatModel, Session, Tool, ToolContext, ToolInput, ToolResult


class AddInput(ToolInput):
    left: int
    right: int


def add(args: AddInput, context: ToolContext) -> ToolResult:
    return ToolResult.success({"sum": args.left + args.right})


async def main() -> None:
    tool = Tool("add", "Add two integers.", AddInput, add)
    async with OpenAIChatModel(
        api_key=os.environ["OPENAI_API_KEY"],
        model=os.environ["OPENAI_MODEL"],
    ) as model:
        agent = Agent(model, tools=[tool])
        result = await agent.run(Session("user_1"), "计算 2 + 3")
        print(result.content)


asyncio.run(main())
```

`ToolInput` 同时生成 JSON Schema 和校验模型输入，默认严格类型、拒绝额外字段与非有限浮点值。`handler`、`authorize` 可为同步或异步函数。`enabled(context)` 是快速同步状态谓词，控制可见性和执行可用性；`authorize(args, context)` 根据已校验参数返回 `Permission`，默认允许。工具重名直接报错。

自定义模型只需实现 `async complete(request: ModelRequest) -> AssistantMessage`；传输和协议失败抛出 `ModelError`。不必继承基类，也不必构造 OpenAI 字段。

## 从哪里开始读

```text
cli → Agent.run → build_context → Model.complete
                ↑                    ↓
            记录结果 ← ToolRegistry.invoke
```

| 想弄清的问题 | 先看场景 | 再看实现 |
| --- | --- | --- |
| 一次输入如何经过工具得到回答？ | `tests/test_runner.py::test_complete_tool_roundtrip` | `src/core/runner.py` |
| 禁用工具为什么不能被执行？ | `test_disabled_tool_is_neither_advertised_nor_executed` | `tools.py` 的 `definitions` / `invoke` |
| 长历史如何保留完整工具调用？ | `tests/test_session.py` | `session.py` 的 `build_context` |
| 替换模型服务影响哪些地方？ | `tests/test_openai.py` | `model.py` 与 `openai.py` |
| Ctrl+C 后会话是否还能继续？ | `test_cancellation_closes_tool_history_and_releases_guard` | `runner.py` 的取消处理 |

建议先追踪第一个场景，再带着具体问题深入一个模块。下文记录了当前的行为边界与取舍，方便之后比较替代方案。

## 行为边界与取舍

- 上下文按用户轮次裁剪，完整保留轮内工具链；不是 token 预算，单轮仍可能很长。历史中调用与结果必须配对，适配器会拒绝损坏的历史。
- 每轮请求重新计算启用工具；执行时和授权完成后再次校验。未在本次请求中提供的工具不会执行。启用判断异常会记录日志并隐藏故障工具；强制总结不再依赖工具发现。
- 工具默认串行。并行模式仅用于互相独立的工具；它们共享 Session 状态，调用方负责避免竞争。
- 同步 handler / authorize 放到工作线程。取消等待不能强制终止 Python 线程或撤销已发生的副作用；需要可协作取消的 I/O 请用异步函数。Session 不支持跨线程并发修改，也不应在一次运行尚未结束时由外部修改历史。
- 取消异步执行会向上抛出 `CancelledError`，并补齐本批未完成调用的中断结果，保持历史配对；并行子任务需协作完成清理。普通工具失败写为失败结果，完整异常留在应用日志中。
- 工具结果超过字符预算后变成合法 JSON 摘要（`truncated` 与 `preview`），不保留完整 `data`。预算是输出字符数，不是字节、token 或内存上限。
- 最多执行 `max_tool_rounds` 批工具，之后额外请求一次禁用工具的总结；模型仍请求工具则返回 `max_steps`。`RunResult.model_calls` 含失败请求，`usage` 按成功响应逐次保存，避免猜测供应商计费字段的累加规则。
- 同一 Session 的同时运行会被拒绝；不同 Session 可以共用 Agent。消息和上下文深复制，状态更新通过 `session.state` 显式完成。
- 适配器仅实现非流式文本/function 工具调用，不自动重试。内部 HTTP 客户端由适配器关闭；注入客户端由调用方关闭。实际远端服务兼容性需要自己的凭证验证。

## 工程检查

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest
uv build
```

测试使用脚本模型和 HTTPX MockTransport，无需模型密钥。生产代码执行严格 mypy；所有 Python 文件执行 Ruff。CI 配置覆盖 Python 3.12 / 3.13 / 3.14，构建 wheel 和 sdist。`uv.lock` 纳入版本控制，`.venv` 和缓存不入库。

工程依据：[uv 项目管理](https://docs.astral.sh/uv/guides/projects/)、[Pydantic 模型](https://docs.pydantic.dev/latest/concepts/models/)、[HTTPX 异步客户端](https://www.python-httpx.org/async/)。
