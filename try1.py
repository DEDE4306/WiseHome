import asyncio
from dotenv import load_dotenv
import os

from langchain import hub
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain.chat_models import init_chat_model
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_mcp_adapters.client import MultiServerMCPClient
from db.chats import get_session_history

# 初始化模型
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
api_key_2 = os.getenv("BAILIAN_API_KEY")

# # 创建模型
# model = init_chat_model(
#     "deepseek-ai/DeepSeek-R1",
#     model_provider="openai",
#     base_url="https://api.siliconflow.cn/v1",
#     api_key=api_key,
# )

model = init_chat_model(
    "qwen3-0.6b",
    model_provider="openai",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=api_key_2,
)

# prompt = hub.pull("hwchase17/structured-chat-agent")

prompt = ChatPromptTemplate.from_template("""
    你是一个智能家居助手。请调用工具完成用户请求。

    用户输入：{input}
    可用工具：{tool_names} {tools}

    规则：
    1. 每次请求只调用一次动作。
    2. 如果动作成功执行，不要重复调用工具。
    3. 输出格式必须是 JSON。
    4. **动作完成后，直接输出 Final Answer，包含动作结果信息**。

    示例输出(json)：
    {{
      "action": "Final Answer",
      "action_input": "卧室空调已开启，温度为25℃"
    }}

    推理过程：
    {agent_scratchpad}
    """
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
            max_iterations=3,  # 最多3次迭代
            max_execution_time=10,  # 最多10秒
            early_stopping_method="generate"
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
                config={"configurable": {"thread_id": "2", "session_id": session_id}}
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
