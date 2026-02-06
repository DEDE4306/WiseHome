import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage


@tool
def search_database(query: str, limit: int = 10) -> str:
    """Search the customer database for records matching the query.

    Args:
        query: Search terms to look for
        limit: Maximum number of results to return
    """
    return f"Found {limit} results for '{query}'"

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

agent = create_agent(
    model,
    tools=[search_database],
    system_prompt="你是一个数据库管理员，需要查询数据库"
)

query = "ice-cream"

tool_call = {
    "type": "tool_call",
    "id": "1",
    "args": {"query": query, "limit": 10}
}

ans = search_database.invoke(tool_call)
print(ans)

response = agent.invoke({"messages": [HumanMessage("查询冰激凌库存")]})
print(response)