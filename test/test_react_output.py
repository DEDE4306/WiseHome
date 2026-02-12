from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

import os
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model

# ========== 创建模型 ==========
# 加载环境变量
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# 创建模型
def create_model():
    model = init_chat_model(
        "qwen3-4b",
        model_provider="openai",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=api_key,
        temperature=0.3,
        model_kwargs={"extra_body": {"enable_thinking": False}}
    )
    return model

@tool
def now_search(query: str) -> str:
    """Search for information."""
    return f"Search results for '{query}': Found relevant info."

@tool
def open_light(brightness: int = 100) -> str:
    """Calculate math expressions."""
    return f"灯已开启，亮度为 {brightness}"




tools = [now_search, open_light]

model = create_model()

agent = create_agent(model, tools)


for chunk in agent.stream(
    {"messages": [("user", "开灯")]},
    stream_mode="updates"  # 显示每个节点的更新
):
    for node, data in chunk.items():
        if "messages" in data and data["messages"]:
            # 获取最新消息
            new_msg = data["messages"][-1]

            if isinstance(new_msg, AIMessage):
                if new_msg.tool_calls:
                    # 🟢 AI 决定调用工具
                    print("🤔 Thought & Action:")
                    for tc in new_msg.tool_calls:
                        print(f"   📞 Tool: {tc['name']}(**{tc['args']}**)  [ID: {tc['id']}]")
                elif new_msg.content:
                    # ✅ AI 最终回复
                    print(f"\n💬 AI 回复: {new_msg.content.strip()}")

            elif isinstance(new_msg, ToolMessage):
                # 🔧 工具执行结果
                print(f"🔧 Tool '{new_msg.name}' 结果: {new_msg.content.strip()}")
                print(f"   [Call ID: {new_msg.tool_call_id}]")

    print("-" * 60)