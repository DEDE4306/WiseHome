from typing import TypedDict, Annotated, Literal, List, Callable, Coroutine, Any


from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from core.model import create_model, Mongodb_checkpointer
from core.workflow import smart_home_workflow


# ========== 主循环 ==========
async def loop():
    print("MCP 智能家居系统已启动！示例: '打开客厅灯'，'查询天气'")

    while True:
        user_input = input("\n你: ").strip()
        if user_input.lower() in {"exit", "quit", "退出"}:
            print("AI: 再见！")
            break

        response = await smart_home_workflow.ainvoke(
            {"messages": [HumanMessage(user_input)]},
            {"configurable": {"thread_id": "1"}},
        )

        msgs = response.get("messages", [])
        print(msgs)

        tool_output = None
        ai_output = None

        for msg in msgs:
            if isinstance(msg, ToolMessage):
                # 工具调用的结果
                tool_output = msg.content if msg.content else "无工具输出"
                print("Tool: ", tool_output)    # 如果工具有输出，打印出来
            elif isinstance(msg, AIMessage):
                # AI 的回复
                ai_output = msg.content if msg.content else "无AI回复"

        print("AI: ", ai_output)
