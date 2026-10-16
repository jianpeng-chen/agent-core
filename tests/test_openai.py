import asyncio
import json

import httpx
import pytest

from core.messages import AssistantMessage, ToolCall, ToolMessage, UserMessage
from core.model import ModelError, ModelRequest
from core.openai import OpenAIChatModel, chat_endpoint, to_wire_messages
from core.tools import ToolDefinition


def completion(content="done"):
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


async def test_http_payload_and_tool_response():
    calls = []

    def handle(request):
        calls.append(request)
        return httpx.Response(
            200,
            json={
                "model": "server-model",
                "usage": {"prompt_tokens": 3},
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "c1",
                                    "type": "function",
                                    "function": {"name": "echo", "arguments": '{"text":"hi"}'},
                                }
                            ],
                        }
                    }
                ],
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        async with OpenAIChatModel(api_key="secret", model="test", client=client) as model:
            result = await model.complete(
                ModelRequest(
                    (UserMessage("hello"),), (ToolDefinition("echo", "Echo", {"type": "object"}),)
                )
            )
        assert not client.is_closed
    payload = json.loads(calls[0].content)
    assert payload == {
        "model": "test",
        "messages": [{"role": "user", "content": "hello"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "echo",
                    "description": "Echo",
                    "parameters": {"type": "object"},
                },
            }
        ],
        "tool_choice": "auto",
    }
    assert calls[0].headers["authorization"] == "Bearer secret"
    assert calls[0].extensions["timeout"]["read"] == 30
    assert result.tool_calls == (ToolCall("c1", "echo", '{"text":"hi"}'),)
    assert result.model == "server-model"
    assert result.usage == {"prompt_tokens": 3}


async def test_explicit_temperature_and_forced_summary_payload():
    payloads = []

    def handle(request):
        payloads.append(json.loads(request.content))
        return httpx.Response(200, json=completion())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        model = OpenAIChatModel(api_key="key", model="any-model", temperature=0.2, client=client)
        await model.complete(
            ModelRequest((UserMessage("hi"),), (ToolDefinition("echo", "Echo", {}),), "none")
        )
        await model.complete(ModelRequest((UserMessage("hi"),)))
    assert payloads[0]["temperature"] == 0.2
    assert payloads[0]["tool_choice"] == "none"
    assert "tools" not in payloads[1]
    assert "tool_choice" not in payloads[1]


@pytest.mark.parametrize(
    "body",
    [
        "not json",
        {},
        {"choices": []},
        {"choices": [{"message": {"role": "assistant"}}]},
        {"choices": [{"message": {"role": "assistant", "content": 42}}]},
        {"choices": [{"message": {"content": "missing role"}}]},
        {"choices": [{"message": {"role": "user", "content": "wrong role"}}]},
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "",
                                "type": "function",
                                "function": {"name": "echo", "arguments": "{}"},
                            }
                        ],
                    }
                }
            ]
        },
    ],
)
async def test_invalid_responses_raise_model_error(body):
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200, content=body if isinstance(body, str) else json.dumps(body)
            )
        )
    ) as client:
        model = OpenAIChatModel(api_key="key", model="test", client=client)
        with pytest.raises(ModelError, match="Invalid model response"):
            await model.complete(ModelRequest((UserMessage("hi"),)))


async def test_http_error_does_not_leak_body():
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(401, text="secret internal body")
        )
    ) as client:
        model = OpenAIChatModel(api_key="key", model="test", client=client)
        with pytest.raises(ModelError, match="401") as error:
            await model.complete(ModelRequest((UserMessage("hi"),)))
    assert "secret" not in str(error.value)


@pytest.mark.parametrize(
    "exception,expected",
    [
        (httpx.ReadTimeout("timeout"), "timed out"),
        (httpx.ConnectError("connection"), "transport error"),
    ],
)
async def test_transport_errors(exception, expected):
    def fail(request):
        raise exception

    async with httpx.AsyncClient(transport=httpx.MockTransport(fail)) as client:
        model = OpenAIChatModel(api_key="key", model="test", client=client)
        with pytest.raises(ModelError, match=expected):
            await model.complete(ModelRequest((UserMessage("hi"),)))


async def test_cancellation_propagates_and_owned_client_closes():
    async def cancel(request):
        raise asyncio.CancelledError

    async with httpx.AsyncClient(transport=httpx.MockTransport(cancel)) as client:
        model = OpenAIChatModel(api_key="key", model="test", client=client)
        with pytest.raises(asyncio.CancelledError):
            await model.complete(ModelRequest((UserMessage("hi"),)))
    model = OpenAIChatModel(api_key="key", model="test")
    async with model:
        assert not model._client.is_closed
    assert model._client.is_closed


def test_complete_tool_exchange_maps_to_wire():
    messages = to_wire_messages(
        (
            AssistantMessage(tool_calls=(ToolCall("c1", "echo"),)),
            ToolMessage("c1", "echo", '{"ok":true}'),
            UserMessage("next"),
        )
    )
    assert messages[0]["tool_calls"][0]["function"]["name"] == "echo"
    assert messages[1]["tool_call_id"] == "c1"


@pytest.mark.parametrize(
    "messages",
    [
        (ToolMessage("missing", "echo", "{}"),),
        (AssistantMessage(tool_calls=(ToolCall("c1", "echo"),)),),
        (AssistantMessage(tool_calls=(ToolCall("c1", "echo"),)), UserMessage("next")),
    ],
)
def test_malformed_history_rejected(messages):
    with pytest.raises(ModelError, match="Conversation"):
        to_wire_messages(messages)


@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://example.test/v1/", "https://example.test/v1/chat/completions"),
        ("https://example.test/v1/chat/completions", "https://example.test/v1/chat/completions"),
    ],
)
def test_endpoint_normalization(url, expected):
    assert chat_endpoint(url) == expected


@pytest.mark.parametrize(
    "kwargs",
    [
        {"timeout": 0},
        {"timeout": float("inf")},
        {"temperature": float("nan")},
        {"base_url": "file:///tmp/model"},
    ],
)
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        OpenAIChatModel(api_key="key", model="test", **kwargs)


async def test_null_tool_calls_are_an_ordinary_text_response():
    body = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "done",
                    "tool_calls": None,
                }
            }
        ]
    }
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=body))
    ) as client:
        model = OpenAIChatModel(api_key="key", model="test", client=client)
        result = await model.complete(ModelRequest((UserMessage("hi"),)))
    assert result.content == "done"
    assert result.tool_calls == ()


@pytest.mark.parametrize("segment", ["a%2Fb", "a%3Fb"])
def test_endpoint_preserves_encoded_path(segment):
    base = f"https://example.test/proxy/{segment}/v1"
    endpoint = base + "/chat/completions"
    assert chat_endpoint(base) == endpoint
    assert chat_endpoint(endpoint) == endpoint
