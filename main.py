import asyncio
from dotenv import load_dotenv
import os
import logging

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

from db.chats import get_session_history


# 初始化模型
load_dotenv()
api_key = os.getenv("BAILIAN_API_KEY")

# 屏蔽没用的错误输出
logging.getLogger("langchain").setLevel(logging.ERROR)
logging.getLogger("langchain_core").setLevel(logging.ERROR)

# 创建模型
model = init_chat_model(
    "qwen3-0.6b",
    model_provider="openai",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=api_key,
    temperature=1,
    model_kwargs={"extra_body": {"enable_thinking": False}}
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个智能家居助手，请调用一个或多个工具完成用户需要的操作."),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    MessagesPlaceholder(variable_name="messages"),
])

async def load_mcp_tools():
    """加载 MCP 工具"""
    client = MultiServerMCPClient(
        {
            "wisehome": {
                "url": "http://localhost:8001/sse",
                "transport": "sse",
            }
        }
    )
    tools = await client.get_tools()
    return tools


async def main():
    try:
        tools = await load_mcp_tools()
        print("MCP 智能家居系统已连接！可以开始对话，比如：'打开客厅灯'、'播放音乐'。")

        agent = create_react_agent(
            model,
            tools=tools,
            prompt=prompt,
        )

        agent_with_memory = RunnableWithMessageHistory(
            agent,
            get_session_history,
            input_messages_key="messages",
            history_messages_key="chat_history",
        )

        while True:
            user_input = input("\n你: ").strip()
            if user_input.lower() in {"exit", "quit"}:
                print("再见！")
                break

            response = await agent_with_memory.ainvoke(
                {
                    "messages": [HumanMessage(content=user_input)],
                },
                config={"configurable": {"thread_id": "2", "session_id": "user_1"}}
            )

            for msg in response["messages"]:
                if hasattr(msg, "content") and msg.content:
                    print(f"reasoning: {msg.type}: {msg.content}")
            ai_response = response["messages"][-1].text()
            print(f"AI: {ai_response}")

    except Exception as e:
        print(f"发生错误：{e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n已退出。")
