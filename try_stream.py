import asyncio
from dotenv import load_dotenv
import os
import logging
import json

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage

from db.chats import get_session_history


# 初始化模型
load_dotenv()
api_key = os.getenv("BAILIAN_API_KEY")

logging.getLogger("langchain").setLevel(logging.ERROR)
logging.getLogger("langchain_core").setLevel(logging.ERROR)

# 创建 prompt 模板
prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are an intelligent home assistant. "
        "Help users control their smart home devices. "
        "Always respond in Chinese unless the user speaks English. "
        "Be concise and direct."
    )),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    MessagesPlaceholder(variable_name="messages"),  # 用户当前输入
])

# 创建模型（添加回调）
model = init_chat_model(
    "qwen3-0.6b",
    model_provider="openai",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=api_key,
    model_kwargs={"extra_body": {"enable_thinking": False}},
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



        # 创建 agent（不使用 prompt 参数，使用 state_modifier）
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

        session_id = "user_1"

        while True:
            user_input = input("\n你: ").strip()
            if user_input.lower() in {"exit", "quit"}:
                print("再见！")
                break

            # 特殊命令：查看历史
            if user_input == "!history":
                history = get_session_history(session_id)
                print("\n📚 当前会话历史:")
                for i, msg in enumerate(history.messages, 1):
                    print(f"{i}. [{msg.__class__.__name__}]: {msg.content[:100]}...")
                continue

            # 特殊命令：清除历史
            if user_input == "!clear":
                history = get_session_history(session_id)
                history.clear()
                print("✅ 历史已清除")
                continue

            try:
                response = await agent_with_memory.ainvoke(
                    {
                        "messages": [HumanMessage(content=user_input)],  # 使用 HumanMessage
                    },
                    config={"configurable": {"session_id": session_id}}
                )

                # 输出最终回复
                ai_response = response["messages"][-1].content
                print(f"\n🤖 AI: {ai_response}\n")

            except Exception as e:
                print(f"❌ 调用出错：{e}")
                import traceback
                traceback.print_exc()

    except Exception as e:
        print(f"❌ 发生错误：{e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n已退出。")