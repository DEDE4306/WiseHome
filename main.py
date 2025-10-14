import asyncio
from dotenv import load_dotenv
import os

from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from langchain_mcp_adapters.client import MultiServerMCPClient

# 初始化模型
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

memory = InMemorySaver()

# 创建模型
model = init_chat_model(
    "deepseek-ai/DeepSeek-V3",
    model_provider="openai",
    base_url="https://api.siliconflow.cn/v1",
    api_key=api_key,
)

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
            model=model,
            tools=tools,
            checkpointer=memory,
        )

        while True:
            user_input = input("\n你: ").strip()
            if user_input.lower() in {"exit", "quit"}:
                print("再见！")
                break

            response = await agent.ainvoke(
                {
                    "messages": [{"role": "user", "content": user_input}],
                },
                config={"configurable": {"thread_id": "2"}}
            )

            for msg in response["messages"]:
                if hasattr(msg, "content") and msg.content:
                    print(f"reasoning: {msg.type}: {msg.content}")
            ai_response = response["messages"][-1].text()
            print(f"AI: {ai_response}")

    except Exception as e:
        print(f"MCP 服务器连接失败：{e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n已退出。")
