import asyncio
import json
import threading
from datetime import date, datetime

import pytest
from pydantic import Field, JsonValue, create_model

from core.session import Session
from core.tools import Permission, Tool, ToolContext, ToolInput, ToolRegistry, ToolResult


class EchoInput(ToolInput):
    text: str = Field(description="需要回传的文本")


def echo(args: EchoInput, context: ToolContext) -> ToolResult:
    return ToolResult.success({"text": args.text})


def make_tool(**kwargs):
    return Tool("echo", "回传文本", EchoInput, echo, **kwargs)


@pytest.fixture
def context():
    return ToolContext(Session("u"))


def test_schema_is_generated_from_input_model():
    definition = make_tool().definition
    assert definition.parameters["properties"]["text"]["type"] == "string"
    assert definition.parameters["additionalProperties"] is False
    assert definition.parameters["required"] == ["text"]


@pytest.mark.parametrize(
    "arguments", ["broken", "[]", "null", "{}", '{"text":1}', '{"text":"hi","extra":true}']
)
async def test_bad_arguments_are_results_not_exceptions(arguments, context):
    result = await make_tool().invoke(arguments, context)
    assert not result.ok
    assert "Invalid tool input" in result.error


async def test_disabled_tools_hidden_and_execution_checked_again(context):
    context.session.state["enabled"] = True
    registry = ToolRegistry([make_tool(enabled=lambda ctx: bool(ctx.session.state["enabled"]))])
    assert len(registry.definitions(context)) == 1
    context.session.state["enabled"] = False
    assert registry.definitions(context) == ()
    result = await registry.invoke("echo", '{"text":"hi"}', context)
    assert result.error == "Tool disabled: echo"


async def test_authorization_receives_validated_input_and_blocks_handler(context):
    async def authorize(args, ctx):
        assert isinstance(args, EchoInput)
        return Permission(False, "denied")

    def forbidden(*args):
        pytest.fail("handler should not execute")

    tool = Tool("echo", "echo", EchoInput, forbidden, authorize=authorize)
    assert (await tool.invoke('{"text":"hi"}', context)).error == "denied"


async def test_sync_handler_runs_off_event_loop(context):
    caller_thread = threading.get_ident()

    def handler(args, ctx):
        assert threading.get_ident() != caller_thread
        return ToolResult.success(args.text)

    result = await Tool("echo", "echo", EchoInput, handler).invoke('{"text":"hi"}', context)
    assert result.data == "hi"


async def test_async_handler_failure_and_cancellation(context):
    async def broken(args, ctx):
        raise RuntimeError("internal details")

    result = await Tool("echo", "echo", EchoInput, broken).invoke('{"text":"hi"}', context)
    assert result.error == "Tool execution failed: echo"

    async def cancelled(args, ctx):
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await Tool("echo", "echo", EchoInput, cancelled).invoke('{"text":"hi"}', context)


async def test_unknown_and_duplicate_tools(context):
    with pytest.raises(ValueError, match="Duplicate"):
        ToolRegistry([make_tool(), make_tool()])
    result = await ToolRegistry().invoke("missing", "{}", context)
    assert result.error == "Unknown tool: missing"


async def test_non_json_output_becomes_tool_error(context):
    def invalid(args, ctx):
        return ToolResult.success(float("nan"))

    result = await Tool("bad", "bad", EchoInput, invalid).invoke('{"text":"hi"}', context)
    assert not result.ok


def test_large_results_remain_valid_bounded_json():
    content = ToolResult.success({"text": '你好"\\' * 1000}).to_json(max_chars=256)
    assert len(content) <= 256
    assert json.loads(content)["truncated"] is True
    assert json.loads(content)["ok"] is True


def test_results_enforce_error_contract():
    with pytest.raises(ValueError):
        ToolResult(ok=False)
    with pytest.raises(ValueError):
        ToolResult(ok=True, error="broken")


async def test_disabled_during_authorization_never_executes(context):
    entered = asyncio.Event()
    release = asyncio.Event()
    context.session.state["enabled"] = True

    async def authorize(args, ctx):
        entered.set()
        await release.wait()
        return Permission()

    def forbidden(args, ctx):
        pytest.fail("disabled tool must not execute")

    tool = Tool(
        "echo",
        "echo",
        EchoInput,
        forbidden,
        enabled=lambda ctx: bool(ctx.session.state["enabled"]),
        authorize=authorize,
    )
    task = asyncio.create_task(tool.invoke('{"text":"hi"}', context))
    await asyncio.wait_for(entered.wait(), 1)
    context.session.state["enabled"] = False
    release.set()
    assert (await task).error == "Tool disabled: echo"


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity", "1e999"])
async def test_non_finite_input_never_reaches_authorization(value, context):
    class NumberInput(ToolInput):
        value: float

    def forbidden(*args):
        pytest.fail("invalid input must not reach authorization or handler")

    tool = Tool("number", "number", NumberInput, forbidden, authorize=forbidden)
    result = await tool.invoke('{"value":' + value + "}", context)
    assert not result.ok
    assert "Invalid tool input" in result.error


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity", "1e999"])
async def test_non_finite_numbers_in_nested_json_are_rejected(value, context):
    class NestedInput(ToolInput):
        payload: JsonValue

    def forbidden(*args):
        pytest.fail("invalid nested input must not execute")

    tool = Tool("nested", "nested", NestedInput, forbidden, authorize=forbidden)
    result = await tool.invoke('{"payload":{"values":[' + value + "]}}", context)
    assert not result.ok
    assert "Invalid tool input" in result.error


@pytest.mark.parametrize(
    "annotation,raw,expected",
    [
        (tuple[float, ...], "[1.25]", (1.25,)),
        (date, '"2026-10-16"', date(2026, 10, 16)),
        (datetime, '"2026-10-16T12:00:00"', datetime(2026, 10, 16, 12)),
    ],
)
async def test_preflight_preserves_native_json_validation(annotation, raw, expected, context):
    input_model = create_model("NativeInput", __base__=ToolInput, value=(annotation, ...))

    def handler(args, ctx):
        assert args.value == expected
        return ToolResult.success("valid")

    result = await Tool("native", "native", input_model, handler).invoke(
        '{"value":' + raw + "}", context
    )
    assert result.ok
    assert result.data == "valid"
