"""Chat Completions 适配器：HTTP、协议解析和资源所有权集中于此。"""

import math
from types import TracebackType
from typing import Any, Literal, Self

import httpx
from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError, model_validator

from core.messages import AssistantMessage, Message, ToolCall, ToolMessage
from core.model import ModelError, ModelRequest


class _WireModel(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)


class _Function(_WireModel):
    name: str = Field(min_length=1)
    arguments: str


class _Call(_WireModel):
    id: str = Field(min_length=1)
    type: Literal["function"]
    function: _Function


class _Message(_WireModel):
    role: Literal["assistant"]
    content: str | None = None
    refusal: str | None = None
    tool_calls: list[_Call] | None = None

    @model_validator(mode="after")
    def check_content(self) -> Self:
        if self.content is None and not self.tool_calls and self.refusal is None:
            raise ValueError("Assistant message has neither content nor tool calls")
        ids = [call.id for call in self.tool_calls or ()]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate tool call IDs")
        return self


class _Choice(_WireModel):
    message: _Message


class _Response(_WireModel):
    choices: list[_Choice] = Field(min_length=1)
    model: str | None = None
    usage: dict[str, JsonValue] | None = None


def to_wire_messages(messages: tuple[Message, ...]) -> list[dict[str, Any]]:
    """严格检查调用与结果配对，不把损坏的历史悄悄发给远端。"""
    result: list[dict[str, Any]] = []
    pending: set[str] = set()
    for message in messages:
        if isinstance(message, ToolMessage):
            if message.tool_call_id not in pending:
                raise ModelError("Conversation contains an orphan or duplicate tool result")
            pending.remove(message.tool_call_id)
            result.append(
                {
                    "role": "tool",
                    "tool_call_id": message.tool_call_id,
                    "name": message.name,
                    "content": message.content,
                }
            )
            continue
        if pending:
            raise ModelError("Conversation contains tool calls without results")
        item: dict[str, Any] = {"role": message.role, "content": message.content}
        if isinstance(message, AssistantMessage) and message.tool_calls:
            pending = {call.id for call in message.tool_calls}
            if len(pending) != len(message.tool_calls) or "" in pending:
                raise ModelError("Conversation contains invalid tool call IDs")
            item["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": call.arguments,
                    },
                }
                for call in message.tool_calls
            ]
        result.append(item)
    if pending:
        raise ModelError("Conversation contains tool calls without results")
    return result


def chat_endpoint(base_url: str) -> str:
    try:
        url = httpx.URL(base_url)
    except httpx.InvalidURL as error:
        raise ValueError("base_url is not a valid HTTP(S) URL") from error
    if url.scheme not in {"http", "https"} or not url.host or url.query or url.fragment:
        raise ValueError("base_url must be an HTTP(S) URL without query or fragment")
    path = url.raw_path.rstrip(b"/")
    if not path.endswith(b"/chat/completions"):
        path += b"/chat/completions"
    return str(url.copy_with(raw_path=path))


class OpenAIChatModel:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        temperature: float | None = None,
        timeout: float = 30.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not api_key.strip() or not model.strip():
            raise ValueError("api_key and model must be non-empty")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        if temperature is not None and not math.isfinite(temperature):
            raise ValueError("temperature must be finite")
        self._endpoint = chat_endpoint(base_url)
        self._api_key = api_key
        self._model = model
        self._temperature = temperature
        self._timeout = timeout
        self._owns_client = client is None
        self._client = client if client is not None else httpx.AsyncClient()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def complete(self, request: ModelRequest) -> AssistantMessage:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": to_wire_messages(request.messages),
        }
        if self._temperature is not None:
            payload["temperature"] = self._temperature
        if request.tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    },
                }
                for tool in request.tools
            ]
            payload["tool_choice"] = request.tool_choice
        try:
            response = await self._client.post(
                self._endpoint,
                headers={"Authorization": f"Bearer {self._api_key}"},
                json=payload,
                timeout=self._timeout,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            # 不把响应正文和认证信息带到模型历史或用户界面。
            raise ModelError(f"Model HTTP error: {error.response.status_code}") from error
        except httpx.TimeoutException as error:
            raise ModelError("Model request timed out") from error
        except httpx.RequestError as error:
            raise ModelError("Model transport error") from error
        try:
            parsed = _Response.model_validate_json(response.content)
        except (ValidationError, ValueError) as error:
            raise ModelError("Invalid model response") from error
        message = parsed.choices[0].message
        return AssistantMessage(
            content=message.content or message.refusal or "",
            tool_calls=tuple(
                ToolCall(call.id, call.function.name, call.function.arguments)
                for call in message.tool_calls or ()
            ),
            model=parsed.model or self._model,
            usage=parsed.usage or {},
        )
