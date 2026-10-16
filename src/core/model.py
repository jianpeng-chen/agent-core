"""执行循环依赖此协议，不依赖任何供应商的 HTTP 格式。"""

from dataclasses import dataclass
from typing import Literal, Protocol

from core.messages import AssistantMessage, Message
from core.tools import ToolDefinition


class ModelError(RuntimeError):
    """模型传输失败或响应不符合约定。"""


@dataclass(frozen=True, slots=True)
class ModelRequest:
    messages: tuple[Message, ...]
    tools: tuple[ToolDefinition, ...] = ()
    tool_choice: Literal["auto", "none"] = "auto"


class Model(Protocol):
    async def complete(self, request: ModelRequest) -> AssistantMessage:
        """返回助手消息；传输和协议错误统一抛出 ModelError。"""
        ...
