import asyncio
import json
import re
import os
from typing import TypedDict, Annotated, Literal, List, Callable, Coroutine, Any
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import add_messages, StateGraph, END

from db.chats import get_session_history
from config.prompts import system_template, router_template

# ========== 模型初始化 ==========
# 加载环境变量
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# 模型初始化
model = init_chat_model(
    "qwen3-32b",
    model_provider="openai",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=api_key,
    temperature=0.3,
    model_kwargs={"extra_body": {"enable_thinking": False}}
)

# ========== 工具加载层 ==========
_tools_cache = None

async def load_mcp_tools() -> List[BaseTool]:
    """从服务器加载 MCP 工具，如果获取失败则抛出异常"""
    global _tools_cache
    if _tools_cache is not None:
        return _tools_cache

    try:
        client = MultiServerMCPClient(
            {
                "wisehome": {
                    "url": "http://localhost:8001/sse",
                    "transport": "sse",
                }
            }
        )
        _tools_cache = await client.get_tools()
        if _tools_cache is None:
            raise RuntimeError("警告: 获取到的工具为空")
        return _tools_cache
    except Exception as e:
        raise RuntimeError(f"获取MCP工具失败: {e}") from e

# ========== 主循环 ==========
async def loop():
    tools = await load_mcp_tools()
    print(f"当前加载的所有工具: {tools}")
    print("MCP 智能家居系统已启动！示例: '打开客厅灯'，'查询天气'")

    agent = create_agent(
        model,
        tools=tools,
        system_prompt=system_template
    )

    while True:
        user_input = input("\n你: ").strip()
        if user_input.lower() in {"exit", "quit", "退出"}:
            print("AI: 再见！")
            break
        response = await agent.ainvoke({"messages": [HumanMessage(user_input)]})
        msgs = response.get("messages", [])

        tool_output = None
        ai_output = None

        for msg in msgs:
            if isinstance(msg, ToolMessage):
                # 工具调用的结果
                tool_output = msg.content[0]['text'] if msg.content and len(msg.content) > 0 else "无工具输出"
            elif isinstance(msg, AIMessage):
                # AI 的回复
                ai_output = msg.content if msg.content else "无AI回复"

        print("Tool: ", tool_output)
        print("AI: ", ai_output)
