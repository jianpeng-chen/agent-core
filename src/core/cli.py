"""用同步终端读取和单个 asyncio.Runner 驱动异步 Agent。"""

import argparse
import asyncio
import math
import os
import sys
from collections.abc import Sequence

from core.demo import DemoModel, echo_tool
from core.model import Model
from core.openai import OpenAIChatModel
from core.runner import Agent, FinishReason
from core.session import Session


def _non_negative(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("必须为非负整数")
    return parsed


def _positive_int(value: str) -> int:
    parsed = _non_negative(value)
    if parsed == 0:
        raise argparse.ArgumentTypeError("必须为正整数")
    return parsed


def _finite(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise argparse.ArgumentTypeError("必须为有限数字")
    return parsed


def _timeout(value: str) -> float:
    parsed = _finite(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("超时必须大于 0")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="最小 Python Agent 工具调用循环")
    parser.add_argument("--demo", action="store_true", help="运行无需 API key 的离线工具调用演示")
    parser.add_argument("--once", metavar="MESSAGE", help="执行一次后退出")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"))
    parser.add_argument(
        "--base-url", default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    )
    parser.add_argument("--temperature", type=_finite, default=os.getenv("OPENAI_TEMPERATURE"))
    parser.add_argument("--timeout", type=_timeout, default=os.getenv("OPENAI_TIMEOUT", "30"))
    parser.add_argument(
        "--max-steps", type=_non_negative, default=os.getenv("AGENT_MAX_STEPS", "8")
    )
    parser.add_argument(
        "--context-turns", type=_positive_int, default=os.getenv("AGENT_CONTEXT_TURNS", "12")
    )
    parser.add_argument("--parallel-tools", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    remote: OpenAIChatModel | None = None
    model: Model
    if args.demo:
        model = DemoModel()
    else:
        key = os.getenv("OPENAI_API_KEY", "")
        if not key.strip():
            parser.error("请设置 OPENAI_API_KEY，或使用 --demo")
        try:
            remote = OpenAIChatModel(
                api_key=key,
                model=args.model,
                base_url=args.base_url,
                temperature=args.temperature,
                timeout=args.timeout,
            )
        except ValueError as error:
            parser.error(str(error))
        model = remote
    agent = Agent(
        model,
        tools=[echo_tool()],
        max_tool_rounds=args.max_steps,
        context_turns=args.context_turns,
        parallel_tools=args.parallel_tools,
        system_prompt=os.getenv(
            "AGENT_SYSTEM_PROMPT", "You are a helpful assistant. Use tools when useful."
        ),
    )
    session = Session(os.getenv("AGENT_USER_ID", "repl_user"))
    try:
        with asyncio.Runner() as runner:
            try:
                if args.once is not None:
                    result = runner.run(agent.run(session, args.once))
                    print(result.content)
                    return (
                        1
                        if result.finish_reason
                        in {
                            FinishReason.MODEL_ERROR,
                            FinishReason.MAX_STEPS,
                        }
                        else 0
                    )
                print("Agent Core。输入 /exit 退出。")
                while True:
                    try:
                        message = input("> ").strip()
                    except EOFError:
                        break
                    if message in {"/exit", "/quit"}:
                        break
                    if message:
                        print(runner.run(agent.run(session, message)).content)
            finally:
                if remote is not None:
                    runner.run(remote.aclose())
    except KeyboardInterrupt:
        print("\n已退出。", file=sys.stderr)
        return 130
    return 0
