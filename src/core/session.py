"""会话拥有历史记录；上下文按完整用户轮次生成独立快照。"""

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from uuid import uuid4

from pydantic import JsonValue

from core.messages import Message, SystemMessage, UserMessage


class SessionBusyError(RuntimeError):
    """同一会话已有未结束的运行，不能交错写入消息。"""


class Session:
    def __init__(
        self,
        user_id: str,
        *,
        session_id: str | None = None,
        messages: Sequence[Message] = (),
        state: Mapping[str, JsonValue] | None = None,
    ) -> None:
        self.user_id = user_id
        self.session_id = session_id if session_id is not None else f"sess_{uuid4().hex}"
        self.state: dict[str, JsonValue] = deepcopy(dict(state or {}))
        self._messages = deepcopy(list(messages))
        self._running = False

    @property
    def messages(self) -> tuple[Message, ...]:
        return tuple(deepcopy(self._messages))

    def append(self, message: Message) -> None:
        self._messages.append(deepcopy(message))

    @contextmanager
    def run_guard(self) -> Iterator[None]:
        """防止同一事件循环中的重入；Session 不用于跨线程共享。"""
        if self._running:
            raise SessionBusyError("Session already has an active run")
        self._running = True
        try:
            yield
        finally:
            self._running = False


@dataclass(frozen=True, slots=True)
class Context:
    session_id: str
    user_id: str
    state: dict[str, JsonValue]
    messages: tuple[Message, ...]


def build_context(session: Session, *, max_turns: int = 12) -> Context:
    """保留完整用户轮次和开头的系统消息，避免切断工具调用与结果。"""
    if isinstance(max_turns, bool) or not isinstance(max_turns, int) or max_turns < 1:
        raise ValueError("max_turns must be a positive integer")
    messages = session.messages
    starts = [index for index, message in enumerate(messages) if isinstance(message, UserMessage)]
    if len(starts) > max_turns:
        prefix = tuple(
            message for message in messages[: starts[0]] if isinstance(message, SystemMessage)
        )
        messages = prefix + messages[starts[-max_turns] :]
    return Context(
        session_id=session.session_id,
        user_id=session.user_id,
        state=deepcopy(session.state),
        messages=messages,
    )
