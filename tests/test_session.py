import pytest

from core.messages import AssistantMessage, SystemMessage, ToolCall, ToolMessage, UserMessage
from core.session import Session, SessionBusyError, build_context


def test_context_keeps_complete_turns_and_leading_system_messages():
    session = Session("u1", session_id="s1")
    for message in (
        SystemMessage("policy"),
        UserMessage("old"),
        AssistantMessage("old answer"),
        UserMessage("new"),
        AssistantMessage(tool_calls=(ToolCall("c1", "echo"),)),
        ToolMessage("c1", "echo", "ok"),
        AssistantMessage("new answer"),
    ):
        session.append(message)
    context = build_context(session, max_turns=1)
    assert [message.role for message in context.messages] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert context.messages[1].content == "new"
    assert context.session_id == "s1"
    assert context.user_id == "u1"
    assert len(session.messages) == 7


def test_session_and_context_isolate_nested_state_and_usage():
    state = {"nested": {"value": 1}}
    usage = {"details": {"tokens": 2}}
    session = Session("u", state=state)
    session.append(AssistantMessage(usage=usage))
    state["nested"]["value"] = 9
    usage["details"]["tokens"] = 9
    context = build_context(session)
    context.state["nested"]["value"] = 8
    context.messages[0].usage["details"]["tokens"] = 8
    assert session.state == {"nested": {"value": 1}}
    assert session.messages[0].usage == {"details": {"tokens": 2}}


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_context_rejects_invalid_window(value):
    with pytest.raises(ValueError, match="positive integer"):
        build_context(Session("u"), max_turns=value)


def test_empty_session_and_timezone_aware_messages():
    session = Session("u")
    assert session.session_id.startswith("sess_")
    assert build_context(session).messages == ()
    assert UserMessage("hello").created_at.utcoffset().total_seconds() == 0


def test_guard_rejects_reentry_and_releases_after_failure():
    session = Session("u")
    with pytest.raises(RuntimeError, match="test"):
        with session.run_guard():
            with pytest.raises(SessionBusyError):
                with session.run_guard():
                    pass
            raise RuntimeError("test")
    with session.run_guard():
        session.append(UserMessage("recovered"))
