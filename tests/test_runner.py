import asyncio
import json

import pytest

from core.messages import AssistantMessage, ToolCall, ToolMessage
from core.model import ModelError
from core.openai import to_wire_messages
from core.runner import Agent, FinishReason
from core.session import Session, SessionBusyError
from core.tools import Permission, Tool, ToolInput, ToolResult


class EchoInput(ToolInput):
    text: str


class EmptyInput(ToolInput):
    pass


async def echo(args, ctx):
    return ToolResult.success({"text": args.text})


def echo_tool(**kwargs):
    return Tool("echo", "Echo", EchoInput, echo, **kwargs)


class ScriptedModel:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.requests = []

    async def complete(self, request):
        self.requests.append(request)
        response = next(self.responses)
        if isinstance(response, BaseException):
            raise response
        return response


def request_tool(name="echo", arguments='{"text":"hi"}', id="c1"):
    return AssistantMessage(tool_calls=(ToolCall(id, name, arguments),), usage={"total_tokens": 2})


async def test_complete_tool_roundtrip():
    model = ScriptedModel(request_tool(), AssistantMessage("done", usage={"total_tokens": 3}))
    session = Session("u", session_id="s")
    result = await Agent(model, tools=[echo_tool()], system_prompt="policy").run(session, "hello")
    assert result.content == "done"
    assert result.finish_reason == FinishReason.COMPLETED
    assert result.model_calls == 2
    assert result.tool_rounds == 1
    assert result.usage == ({"total_tokens": 2}, {"total_tokens": 3})
    assert [m.role for m in session.messages] == ["user", "assistant", "tool", "assistant"]
    assert json.loads(session.messages[2].content)["data"] == {"text": "hi"}
    assert model.requests[0].messages[0].content.startswith("policy")
    assert [m.role for m in model.requests[1].messages] == ["system", "user", "assistant", "tool"]


async def test_disabled_tool_is_neither_advertised_nor_executed():
    model = ScriptedModel(request_tool(), AssistantMessage("done"))
    session = Session("u")
    await Agent(model, tools=[echo_tool(enabled=lambda ctx: False)]).run(session, "run")
    assert model.requests[0].tools == ()
    assert "not offered" in session.messages[2].content


async def test_enablement_rechecked_after_model_returns():
    session = Session("u", state={"enabled": True})

    class ChangingModel(ScriptedModel):
        async def complete(self, request):
            session.state["enabled"] = False
            return await super().complete(request)

    model = ChangingModel(request_tool(), AssistantMessage("done"))
    tool = echo_tool(enabled=lambda ctx: bool(ctx.session.state["enabled"]))
    await Agent(model, tools=[tool]).run(session, "run")
    assert len(model.requests[0].tools) == 1
    assert "disabled" in session.messages[2].content
    assert model.requests[1].tools == ()


@pytest.mark.parametrize(
    "tool,call,error",
    [
        (echo_tool(), request_tool("missing"), "not offered"),
        (echo_tool(), request_tool(arguments="{bad"), "Invalid tool input"),
        (
            echo_tool(authorize=lambda args, ctx: Permission(False, "denied")),
            request_tool(),
            "denied",
        ),
    ],
)
async def test_tool_errors_are_returned_to_model(tool, call, error):
    model = ScriptedModel(call, AssistantMessage("recovered"))
    session = Session("u")
    result = await Agent(model, tools=[tool]).run(session, "run")
    assert result.content == "recovered"
    assert error in model.requests[1].messages[-1].content


@pytest.mark.parametrize("parallel", [False, True])
async def test_execution_order_and_parallel_opt_in(parallel):
    release = asyncio.Event()
    started = []

    async def first(args, ctx):
        started.append("first")
        if parallel:
            await asyncio.wait_for(release.wait(), 1)
        return ToolResult.success("first")

    async def second(args, ctx):
        started.append("second")
        release.set()
        return ToolResult.success("second")

    tools = [
        Tool("first", "first", EmptyInput, first),
        Tool("second", "second", EmptyInput, second),
    ]
    calls = (ToolCall("c1", "first"), ToolCall("c2", "second"))
    model = ScriptedModel(AssistantMessage(tool_calls=calls), AssistantMessage("done"))
    session = Session("u")
    await Agent(model, tools=tools, parallel_tools=parallel).run(session, "run")
    assert started == ["first", "second"]
    assert [m.name for m in session.messages if isinstance(m, ToolMessage)] == ["first", "second"]


async def test_limit_forces_exactly_one_summary_call():
    model = ScriptedModel(request_tool(), AssistantMessage("summary"))
    result = await Agent(model, tools=[echo_tool()], max_tool_rounds=1).run(Session("u"), "run")
    assert result.finish_reason == FinishReason.FORCED_SUMMARY
    assert result.tool_rounds == 1
    assert len(model.requests) == 2
    assert model.requests[-1].tool_choice == "none"


async def test_zero_limit_never_executes_even_if_model_ignores_none():
    model = ScriptedModel(request_tool())
    session = Session("u")
    result = await Agent(model, tools=[echo_tool()], max_tool_rounds=0).run(session, "run")
    assert result.finish_reason == FinishReason.MAX_STEPS
    assert result.tool_rounds == 0
    assert [m.role for m in session.messages] == ["user", "assistant"]
    assert to_wire_messages(session.messages)


@pytest.mark.parametrize(
    "limit,responses",
    [
        (1, (ModelError("upstream failed"),)),
        (0, (ModelError("summary failed"),)),
        (1, (request_tool(), ModelError("summary failed"))),
    ],
)
async def test_model_errors_include_summary_path(limit, responses):
    model = ScriptedModel(*responses)
    result = await Agent(model, tools=[echo_tool()], max_tool_rounds=limit).run(Session("u"), "run")
    assert result.finish_reason == FinishReason.MODEL_ERROR
    assert "failed" in result.content
    assert result.model_calls == len(responses)


async def test_duplicate_call_ids_are_model_error_before_history_is_written():
    model = ScriptedModel(
        AssistantMessage(tool_calls=(ToolCall("c", "echo"), ToolCall("c", "echo")))
    )
    session = Session("u")
    result = await Agent(model, tools=[echo_tool()]).run(session, "run")
    assert result.finish_reason == FinishReason.MODEL_ERROR
    assert to_wire_messages(session.messages)


@pytest.mark.parametrize("parallel", [False, True])
async def test_cancellation_closes_tool_history_and_releases_guard(parallel):
    started = asyncio.Event()

    async def blocking(args, ctx):
        started.set()
        await asyncio.Event().wait()

    model = ScriptedModel(request_tool(), AssistantMessage("recovered"))
    agent = Agent(model, tools=[Tool("echo", "Echo", EchoInput, blocking)], parallel_tools=parallel)
    session = Session("u")
    task = asyncio.create_task(agent.run(session, "run"))
    await asyncio.wait_for(started.wait(), 1)
    with pytest.raises(SessionBusyError):
        await agent.run(session, "must not append")
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert "interrupted" in session.messages[-1].content
    assert to_wire_messages(session.messages)
    assert (await agent.run(session, "continue")).content == "recovered"
    assert all(m.content != "must not append" for m in session.messages)


async def test_context_window_keeps_tool_exchange_in_current_turn():
    model = ScriptedModel(AssistantMessage("old"), request_tool(), AssistantMessage("new"))
    agent = Agent(model, tools=[echo_tool()], context_turns=1)
    session = Session("u")
    await agent.run(session, "old question")
    await agent.run(session, "new question")
    assert [m.role for m in model.requests[-1].messages] == ["system", "user", "assistant", "tool"]
    assert to_wire_messages(model.requests[-1].messages)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_tool_rounds": -1},
        {"max_tool_rounds": True},
        {"context_turns": 0},
        {"max_tool_result_chars": 1},
    ],
)
def test_invalid_limits(kwargs):
    with pytest.raises(ValueError):
        Agent(ScriptedModel(), **kwargs)


@pytest.mark.parametrize("limit", [1, 2])
async def test_broken_enabled_predicate_does_not_block_next_response(limit):
    session = Session("u", state={"allowed": True})

    async def remove_state(args, context):
        del context.session.state["allowed"]
        return ToolResult.success()

    healthy = Tool("remove", "Remove state", EmptyInput, remove_state)
    broken = echo_tool(enabled=lambda context: bool(context.session.state["allowed"]))
    model = ScriptedModel(request_tool("remove", "{}"), AssistantMessage("done"))
    result = await Agent(model, tools=[healthy, broken], max_tool_rounds=limit).run(session, "go")
    assert result.content == "done"
    assert all(tool.name != "echo" for tool in model.requests[-1].tools)
    if limit == 1:
        assert model.requests[-1].tools == ()
        assert result.finish_reason == FinishReason.FORCED_SUMMARY


async def test_repeated_cancellation_during_tool_cleanup_preserves_history():
    ready = asyncio.Event()
    cleaning = asyncio.Event()
    release = asyncio.Event()

    async def cancel_self(args, context):
        await ready.wait()
        raise asyncio.CancelledError

    async def needs_cleanup(args, context):
        ready.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cleaning.set()
            await release.wait()
            raise

    tools = [
        Tool("first", "first", EmptyInput, cancel_self),
        Tool("second", "second", EmptyInput, needs_cleanup),
    ]
    model = ScriptedModel(
        AssistantMessage(
            tool_calls=(
                ToolCall("c1", "first"),
                ToolCall("c2", "second"),
            )
        ),
        AssistantMessage("recovered"),
    )
    agent = Agent(model, tools=tools, parallel_tools=True)
    session = Session("u")
    task = asyncio.create_task(agent.run(session, "go"))
    await asyncio.wait_for(cleaning.wait(), 1)
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 1)
    assert [m.role for m in session.messages] == ["user", "assistant", "tool", "tool"]
    assert to_wire_messages(session.messages)
    assert (await agent.run(session, "continue")).content == "recovered"
