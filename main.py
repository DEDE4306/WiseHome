import asyncio
from dotenv import load_dotenv
import os

from langchain import hub
from langchain.agents import AgentExecutor,create_structured_chat_agent
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_mcp_adapters.client import MultiServerMCPClient
from db.chats import get_session_history

# 初始化模型
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
api_key_2 = os.getenv("BAILIAN_API_KEY")

# # 创建模型
model = init_chat_model(
    "deepseek-ai/DeepSeek-R1",
    model_provider="openai",
    base_url="https://api.siliconflow.cn/v1",
    api_key=api_key,
)

prompt = hub.pull("hwchase17/structured-chat-agent")

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

        agent = create_structured_chat_agent(
            model,
            tools=tools,
            prompt=prompt
        )

        agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
        )

        agent_with_memory = RunnableWithMessageHistory(
            agent_executor,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
        )

        session_id = "user_1"


        while True:
            user_input = input("\n你: ").strip()
            if user_input.lower() in {"exit", "quit"}:
                print("再见！")
                break

            response = await agent_with_memory.ainvoke(
                {"input": user_input},
                config={"configurable": {"thread_id": "2","session_id": session_id}}
            )

            ai_response = response["output"]
            print(f"AI: {ai_response}")

    except Exception as e:
        print(f"发生错误：{e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n已退出。")
