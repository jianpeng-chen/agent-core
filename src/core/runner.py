"""以显式迭代组织模型和工具，保留清晰的退出与取消边界。"""

import asyncio
import json
from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from pydantic import JsonValue

from core.messages import AssistantMessage, SystemMessage, ToolCall, ToolMessage, UserMessage
from core.model import Model, ModelError, ModelRequest
from core.session import Session, build_context
from core.tools import Tool, ToolContext, ToolRegistry, ToolResult


class FinishReason(StrEnum):
    COMPLETED = "completed"
    FORCED_SUMMARY = "forced_summary"
    MAX_STEPS = "max_steps"
    MODEL_ERROR = "model_error"


@dataclass(frozen=True, slots=True)
class RunResult:
    content: str
    finish_reason: FinishReason
    model_calls: int
    tool_rounds: int
    model: str | None = None
    usage: tuple[dict[str, JsonValue], ...] = ()


class Agent:
    def __init__(
        self,
        model: Model,
        *,
        tools: Iterable[Tool[Any]] = (),
        system_prompt: str = "You are a helpful assistant.",
        max_tool_rounds: int = 8,
        context_turns: int = 12,
        parallel_tools: bool = False,
        max_tool_result_chars: int = 1200,
    ) -> None:
        for name, value, minimum in (
            ("max_tool_rounds", max_tool_rounds, 0),
            ("context_turns", context_turns, 1),
            ("max_tool_result_chars", max_tool_result_chars, 128),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        self.model = model
        self.tools = ToolRegistry(tools)
        self.system_prompt = system_prompt
        self.max_tool_rounds = max_tool_rounds
        self.context_turns = context_turns
        self.parallel_tools = parallel_tools
        self.max_tool_result_chars = max_tool_result_chars

    async def run(self, session: Session, user_message: str) -> RunResult:
        with session.run_guard():
            session.append(UserMessage(user_message))
            usage: list[dict[str, JsonValue]] = []
            last_model: str | None = None
            for tool_rounds in range(self.max_tool_rounds + 1):
                forced = tool_rounds == self.max_tool_rounds
                request = self._request(session, forced=forced)
                try:
                    response = await self.model.complete(request)
                    self._validate_calls(response)
                except ModelError as error:
                    content = str(error)
                    session.append(AssistantMessage(content=content))
                    return RunResult(
                        content,
                        FinishReason.MODEL_ERROR,
                        tool_rounds + 1,
                        tool_rounds,
                        last_model,
                        tuple(usage),
                    )
                usage.append(deepcopy(response.usage))
                last_model = response.model
                if not response.tool_calls:
                    content = response.content or ("Tool round limit reached." if forced else "")
                    session.append(
                        AssistantMessage(
                            content=content,
                            model=response.model,
                            usage=response.usage,
                        )
                    )
                    return RunResult(
                        content,
                        FinishReason.FORCED_SUMMARY if forced else FinishReason.COMPLETED,
                        tool_rounds + 1,
                        tool_rounds,
                        last_model,
                        tuple(usage),
                    )
                if forced:
                    content = (
                        "Tool round limit reached; model requested more tools instead of a summary."
                    )
                    session.append(
                        AssistantMessage(
                            content=content,
                            model=response.model,
                            usage=response.usage,
                        )
                    )
                    return RunResult(
                        content,
                        FinishReason.MAX_STEPS,
                        tool_rounds + 1,
                        tool_rounds,
                        last_model,
                        tuple(usage),
                    )
                session.append(response)
                offered = {tool.name for tool in request.tools}
                await self._execute_calls(session, response.tool_calls, offered)
            raise AssertionError("Unreachable agent loop state")

    def _request(self, session: Session, *, forced: bool) -> ModelRequest:
        context = build_context(session, max_turns=self.context_turns)
        runtime = json.dumps(
            {"session_id": context.session_id, "state": context.state},
            ensure_ascii=False,
            allow_nan=False,
        )
        system = SystemMessage(f"{self.system_prompt}\n\nRuntime context:\n{runtime}")
        return ModelRequest(
            messages=(system, *context.messages),
            tools=() if forced else self.tools.definitions(ToolContext(session)),
            tool_choice="none" if forced else "auto",
        )

    @staticmethod
    def _validate_calls(response: AssistantMessage) -> None:
        ids = [call.id for call in response.tool_calls]
        if len(ids) != len(set(ids)) or any(
            not call.id or not call.name for call in response.tool_calls
        ):
            raise ModelError("Model returned invalid or duplicate tool call IDs/names")

    async def _execute_calls(
        self, session: Session, calls: tuple[ToolCall, ...], offered: set[str]
    ) -> None:
        context = ToolContext(session)

        async def execute(call: ToolCall) -> ToolMessage:
            result = (
                await self.tools.invoke(call.name, call.arguments, context)
                if call.name in offered
                else ToolResult.failure(f"Tool was not offered: {call.name}")
            )
            return ToolMessage(
                call.id, call.name, result.to_json(max_chars=self.max_tool_result_chars)
            )

        completed: list[ToolMessage] = []
        tasks: list[asyncio.Task[ToolMessage]] = []
        try:
            if self.parallel_tools:
                tasks = [asyncio.create_task(execute(call)) for call in calls]
                completed = list(await asyncio.gather(*tasks))
            else:
                for call in calls:
                    completed.append(await execute(call))
        except asyncio.CancelledError:
            for task in tasks:
                task.cancel()
            if tasks:
                cleanup = asyncio.gather(*tasks, return_exceptions=True)
                # 重复取消不能跳过收拢任务和补齐历史；等待协作式清理完成。
                while not cleanup.done():
                    try:
                        await asyncio.shield(cleanup)
                    except asyncio.CancelledError:
                        continue
                completed = [
                    task.result()
                    for task in tasks
                    if not task.cancelled() and task.exception() is None
                ]
            by_id = {message.tool_call_id: message for message in completed}
            # 取消后也闭合本批调用记录，使会话后续仍可继续使用。
            for call in calls:
                session.append(
                    by_id.get(call.id)
                    or ToolMessage(
                        call.id,
                        call.name,
                        ToolResult.failure("Tool execution interrupted by cancellation").to_json(),
                    )
                )
            raise
        for message in completed:
            session.append(message)
