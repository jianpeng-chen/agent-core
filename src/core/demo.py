"""无需网络的完整工具调用示例，也用于命令行冒烟验证。"""

import json
from uuid import uuid4

from pydantic import Field

from core.messages import AssistantMessage, ToolCall, ToolMessage, UserMessage
from core.model import ModelRequest
from core.tools import Tool, ToolContext, ToolInput, ToolResult


class EchoInput(ToolInput):
    text: str = Field(description="需要回传的文本")


def echo_tool() -> Tool[EchoInput]:
    def echo(args: EchoInput, context: ToolContext) -> ToolResult:
        return ToolResult.success({"text": args.text})

    return Tool("echo", "Echo the supplied text.", EchoInput, echo)


class DemoModel:
    """固定请求 echo 工具，展示真实执行链路；不模拟智能推理。"""

    async def complete(self, request: ModelRequest) -> AssistantMessage:
        last = request.messages[-1]
        if isinstance(last, ToolMessage):
            result = json.loads(last.content)
            if result.get("truncated"):
                content = "工具结果过长，已生成摘要。"
            elif result["ok"]:
                content = f"工具返回：{result['data']['text']}"
            else:
                content = f"工具失败：{result['error']}"
            return AssistantMessage(content, model="offline-demo")
        if request.tool_choice == "none":
            return AssistantMessage("已达到工具轮次上限。", model="offline-demo")
        text = last.content if isinstance(last, UserMessage) else "hello"
        return AssistantMessage(
            tool_calls=(
                ToolCall(
                    f"demo_{uuid4().hex}", "echo", json.dumps({"text": text}, ensure_ascii=False)
                ),
            ),
            model="offline-demo",
        )
