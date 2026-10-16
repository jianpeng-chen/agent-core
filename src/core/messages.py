"""供应商无关的消息类型；协议字段由模型适配器负责转换。"""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal

from pydantic import JsonValue


def utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str
    name: str
    arguments: str = "{}"


@dataclass(frozen=True, slots=True)
class UserMessage:
    content: str
    created_at: datetime = field(default_factory=utc_now)
    role: Literal["user"] = field(default="user", init=False)


@dataclass(frozen=True, slots=True)
class SystemMessage:
    content: str
    created_at: datetime = field(default_factory=utc_now)
    role: Literal["system"] = field(default="system", init=False)


@dataclass(frozen=True, slots=True)
class AssistantMessage:
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    model: str | None = None
    usage: dict[str, JsonValue] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utc_now)
    role: Literal["assistant"] = field(default="assistant", init=False)


@dataclass(frozen=True, slots=True)
class ToolMessage:
    tool_call_id: str
    name: str
    content: str
    created_at: datetime = field(default_factory=utc_now)
    role: Literal["tool"] = field(default="tool", init=False)


type Message = UserMessage | SystemMessage | AssistantMessage | ToolMessage
