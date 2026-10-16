"""工具的输入契约、可用性、授权和执行错误在此统一处理。"""

import asyncio
import inspect
import json
import logging
import math
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, JsonValue, ValidationError

from core.session import Session

logger = logging.getLogger(__name__)


def _finite_json_number(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("JSON numbers must be finite")
    return number


class ToolInput(BaseModel):
    """工具参数严格校验，未知字段视为错误。"""

    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


@dataclass(frozen=True, slots=True)
class ToolContext:
    session: Session


@dataclass(frozen=True, slots=True)
class ToolResult:
    ok: bool
    data: JsonValue = None
    error: str | None = None
    metadata: dict[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.ok and self.error is not None:
            raise ValueError("Successful tool results cannot include an error")
        if not self.ok and not self.error:
            raise ValueError("Failed tool results require an error")

    @classmethod
    def success(cls, data: JsonValue = None) -> "ToolResult":
        return cls(ok=True, data=data)

    @classmethod
    def failure(cls, error: str) -> "ToolResult":
        return cls(ok=False, error=error)

    def to_json(self, *, max_chars: int = 1200) -> str:
        """大结果以合法 JSON 摘要返回，避免截断后的内容破坏 JSON 结构。"""
        if max_chars < 128:
            raise ValueError("max_chars must be at least 128")
        payload: dict[str, JsonValue] = {"ok": self.ok, "metadata": self.metadata}
        payload["data" if self.ok else "error"] = self.data if self.ok else self.error
        text = json.dumps(payload, ensure_ascii=False, allow_nan=False)
        if len(text) <= max_chars:
            return text
        low, high = 0, len(text)
        while low < high:
            middle = (low + high + 1) // 2
            preview = json.dumps(
                {"ok": self.ok, "truncated": True, "preview": text[:middle]},
                ensure_ascii=False,
            )
            if len(preview) <= max_chars:
                low = middle
            else:
                high = middle - 1
        return json.dumps(
            {"ok": self.ok, "truncated": True, "preview": text[:low]}, ensure_ascii=False
        )


@dataclass(frozen=True, slots=True)
class Permission:
    allowed: bool = True
    reason: str = "Tool execution denied"


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]


async def _invoke[**P, R](
    function: Callable[P, R | Awaitable[R]], *args: P.args, **kwargs: P.kwargs
) -> R:
    result: R | Awaitable[R]
    if inspect.iscoroutinefunction(function):
        result = function(*args, **kwargs)
    else:
        result = await asyncio.to_thread(function, *args, **kwargs)
    if inspect.isawaitable(result):
        return cast(R, await result)
    return result


@dataclass(frozen=True)
class Tool[InputT: ToolInput]:
    name: str
    description: str
    input_model: type[InputT]
    handler: Callable[[InputT, ToolContext], ToolResult | Awaitable[ToolResult]]
    enabled: Callable[[ToolContext], bool] | None = None
    authorize: Callable[[InputT, ToolContext], Permission | Awaitable[Permission]] | None = None

    def __post_init__(self) -> None:
        if not self.name or self.name.strip() != self.name:
            raise ValueError("Tool name must be non-empty and have no surrounding whitespace")

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(self.name, self.description, self.input_model.model_json_schema())

    def is_enabled(self, context: ToolContext) -> bool:
        """启用谓词只做快速的内存状态判断，不应包含 I/O。"""
        return self.enabled is None or self.enabled(context)

    async def invoke(self, arguments: str, context: ToolContext) -> ToolResult:
        try:
            if not self.is_enabled(context):
                return ToolResult.failure(f"Tool disabled: {self.name}")
            try:
                # 独立检查数字后仍传原始 JSON，保留日期、元组等 JSON 模式语义。
                json.loads(
                    arguments, parse_float=_finite_json_number, parse_constant=_finite_json_number
                )
                parsed = self.input_model.model_validate_json(arguments, strict=True)
            except ValidationError as error:
                issues = "; ".join(
                    f"{'.'.join(map(str, issue['loc'])) or '<root>'}: {issue['msg']}"
                    for issue in error.errors(include_input=False, include_url=False)
                )
                return ToolResult.failure(f"Invalid tool input: {issues}")
            except ValueError:
                return ToolResult.failure("Invalid tool input: expected JSON with finite numbers")
            if self.authorize is not None:
                permission = await _invoke(self.authorize, parsed, context)
                if not permission.allowed:
                    return ToolResult.failure(permission.reason or "Tool execution denied")
                if not self.is_enabled(context):
                    return ToolResult.failure(f"Tool disabled: {self.name}")
            result = await _invoke(self.handler, parsed, context)
            # 及时验证输出可序列化，防止结果写入历史时才中断整轮运行。
            result.to_json()
            return result
        except Exception:
            logger.exception("Tool %s failed", self.name)
            return ToolResult.failure(f"Tool execution failed: {self.name}")


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool[Any]] = ()) -> None:
        self._tools: dict[str, Tool[Any]] = {}
        for tool in tools:
            if tool.name in self._tools:
                raise ValueError(f"Duplicate tool name: {tool.name}")
            self._tools[tool.name] = tool

    def definitions(self, context: ToolContext) -> tuple[ToolDefinition, ...]:
        definitions = []
        for tool in self._tools.values():
            try:
                if tool.is_enabled(context):
                    definitions.append(tool.definition)
            except Exception:
                logger.exception("Tool %s could not be advertised", tool.name)
        return tuple(definitions)

    async def invoke(self, name: str, arguments: str, context: ToolContext) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult.failure(f"Unknown tool: {name}")
        return await tool.invoke(arguments, context)
