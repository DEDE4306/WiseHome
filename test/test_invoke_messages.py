import asyncio
import json
from typing import List, Dict, Any

from langchain_core.messages import ToolMessage, AIMessage, HumanMessage
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from core.model import create_model, Mongodb_checkpointer
from langchain.agents import create_agent

llm = create_model()


async def load_mcp_tools() -> List[BaseTool]:
    """从服务器加载 MCP 工具，如果获取失败则抛出异常"""
    try:
        client = MultiServerMCPClient(
            {
                "wisehome": {
                    "url": "http://localhost:8001/sse",
                    "transport": "sse",
                }
            }
        )
        tools = await client.get_tools()
        if tools is None:
            raise RuntimeError("警告: 获取到的工具为空")
        return tools
    except Exception as e:
        raise RuntimeError(f"获取MCP工具失败: {e}") from e


async def build_agent():
    tools = await load_mcp_tools()
    agent = create_agent(
        model=llm,
        checkpointer=Mongodb_checkpointer,
        tools=tools,
    )
    return agent


def extract_tool_text(tool_content: str) -> str:
    """解析工具输出内容，只提取核心的text字段

    Args:
        tool_content: ToolMessage的content内容（JSON字符串/列表）

    Returns:
        纯文本内容，解析失败则返回原始内容
    """
    try:
        # 处理嵌套列表的情况（如 [[{'type':'text', 'text':'xxx'}]]）
        content = json.loads(tool_content) if isinstance(tool_content, str) else tool_content

        # 递归提取所有text字段
        text_list = []

        def recursive_extract_text(data):
            if isinstance(data, list):
                for item in data:
                    recursive_extract_text(item)
            elif isinstance(data, dict):
                if "text" in data:
                    text_list.append(data["text"])

        recursive_extract_text(content)

        # 拼接所有text内容
        return "；".join(text_list) if text_list else tool_content
    except (json.JSONDecodeError, TypeError):
        # 解析失败则返回原始内容（兼容非JSON格式）
        return tool_content


async def run_workflow_with_new_msgs(agent, message):
    """运行workflow并只获取本次新增的消息，Tool只输出text内容"""
    new_msgs = []

    # 用astream模式运行，只捕获增量更新
    async for chunk in agent.astream(
            {"messages": [HumanMessage(content=message)]},
            {"configurable": {"thread_id": "1"}},
    ):
        for node_output in chunk.values():
            if "messages" in node_output:
                new_msgs.extend(node_output["messages"])

    # ========== 核心优化：只提取Tool的text + 过滤无效AI回复 ==========
    # 1. 提取所有工具的纯文本内容
    tool_texts = []
    for msg in new_msgs:
        if isinstance(msg, ToolMessage) and msg.content:
            pure_text = extract_tool_text(msg.content)
            if pure_text:  # 过滤空文本
                tool_texts.append(pure_text)

    # 2. 提取所有有效AI回复
    ai_outputs = []
    for msg in new_msgs:
        if isinstance(msg, AIMessage) and msg.content and msg.content != "无AI回复":
            ai_outputs.append(msg.content)

    # 3. 格式化输出（去掉列表，只显示纯文本）
    tool_str = "；".join(tool_texts) if tool_texts else "无工具输出"
    ai_str = "\n".join(ai_outputs) if ai_outputs else "无AI回复"

    print("Tool: ", tool_str)
    print("AI: ", ai_str)
    return new_msgs


async def main():
    agent = await build_agent()
    while True:
        message = input("用户: ")
        if not message.strip():
            continue
        await run_workflow_with_new_msgs(agent, message)


if __name__ == "__main__":
    asyncio.run(main())