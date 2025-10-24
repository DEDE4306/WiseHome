import asyncio
import json
import re
import os

from typing import TypedDict, Annotated, Optional, Literal, List
from dotenv import load_dotenv

from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from langgraph.graph import add_messages, StateGraph, END
from langgraph.prebuilt import create_react_agent

from db.chats import get_session_history
from template import system_template, router_template

# 初始化模型
load_dotenv()
api_key = os.getenv("BAILIAN_API_KEY")

model = init_chat_model(
    "qwen3-0.6b",
    model_provider="openai",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=api_key,
    temperature=1,
    model_kwargs={"extra_body": {"enable_thinking": False}}
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_template),
    MessagesPlaceholder("chat_history"),
    ("human", "{input} {agent_scratchpad}"),
])

# 改进的路由模板



class AgentState(TypedDict):
    """状态定义"""
    messages: Annotated[list, add_messages]
    sub_tasks: List[dict]  # [{"task": "xxx", "category": "xxx"}]
    current_idx: int  # 当前执行到第几个任务
    input: str  # 原始用户输入
    is_chat: bool  # 是否为普通对话


_tools_cache = None


async def load_mcp_tools():
    """加载 MCP 工具"""
    global _tools_cache
    if _tools_cache is None:
        client = MultiServerMCPClient(
            {
                "wisehome": {
                    "url": "http://localhost:8001/sse",
                    "transport": "sse",
                }
            }
        )
        _tools_cache = await client.get_tools()
    return _tools_cache


def extract_json(text: str) -> dict:
    """从文本中提取 JSON"""
    text = text.strip()

    # 尝试直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 尝试提取 JSON 块
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception as e:
            print(f"⚠️  JSON 解析失败：{e}")

    return {}


async def llm_route(user_input: str) -> dict:
    """使用 LLM 判断意图并拆分任务"""
    route_prompt = ChatPromptTemplate.from_messages([
        ("system", router_template),
        ("user", "{input}")
    ])

    response = await (route_prompt | model).ainvoke({"input": user_input})
    result = extract_json(response.content)

    # 验证结果
    if not result:
        print("⚠️  路由解析失败，默认为普通对话")
        return {"type": "chat", "response": "抱歉，我没理解你的意思。"}

    result_type = result.get("type", "chat")

    # 如果是任务但 sub_tasks 为空，降级为对话
    if result_type == "task" and not result.get("sub_tasks"):
        print("⚠️  任务拆分为空，降级为对话")
        return {"type": "chat", "response": "请告诉我你需要什么帮助？"}

    return result


async def agent_router(state: AgentState) -> AgentState:
    """路由节点：判断意图并初始化状态"""
    user_input = state["input"]
    route_result = await llm_route(user_input)

    result_type = route_result.get("type", "chat")

    print(f"📋 意图识别：{result_type}")

    if result_type == "chat":
        # 普通对话，直接返回
        response = route_result.get("response", "你好！")
        print(f"💬 对话回复：{response}")
        return {
            **state,
            "is_chat": True,
            "messages": [AIMessage(content=response)],
            "sub_tasks": [],
            "current_idx": 0
        }
    else:
        # 任务指令
        sub_tasks = route_result.get("sub_tasks", [])
        print(f"📋 任务拆分：{json.dumps(sub_tasks, ensure_ascii=False)}")
        return {
            **state,
            "is_chat": False,
            "sub_tasks": sub_tasks,
            "current_idx": 0,
            "messages": []
        }


def route_decision(state: AgentState) -> Literal["smart_home_control", "query_info", "end"]:
    """决策函数：根据当前任务的 category 路由"""
    # 如果是普通对话，直接结束
    if state.get("is_chat", False):
        return "end"

    current_idx = state.get("current_idx", 0)
    sub_tasks = state.get("sub_tasks", [])

    # 所有任务执行完毕
    if current_idx >= len(sub_tasks):
        return "end"

    # 获取当前任务的类别
    current_task = sub_tasks[current_idx]
    category = current_task.get("category", "query_info")

    print(f"🔀 路由到: {category} (任务 {current_idx + 1}/{len(sub_tasks)})")

    return category


async def filter_tools_by_category(all_tools: List[BaseTool], category: str) -> List[BaseTool]:
    """根据类别筛选工具"""
    if category == "smart_home_control":
        keywords = ["turn", "set", "open", "close", "switch", "调节", "开", "关", "设置", "play", "stop"]
    else:  # query_info
        keywords = ["get", "query", "status", "weather", "list", "查询", "获取", "状态", "信息"]

    filtered_tools = [
        tool for tool in all_tools
        if any(kw in tool.name.lower() or kw in tool.description.lower() for kw in keywords)
    ]

    # 兜底：若没筛到工具，返回所有工具
    return filtered_tools if filtered_tools else all_tools


async def execute_current_task(state: AgentState, category: str) -> AgentState:
    """执行当前子任务"""
    current_idx = state["current_idx"]
    sub_tasks = state["sub_tasks"]
    current_task = sub_tasks[current_idx]
    task_content = current_task["task"]

    print(f"⚙️  执行任务 {current_idx + 1}: {task_content} [{category}]")

    # 加载并筛选工具
    all_tools = await load_mcp_tools()
    filtered_tools = await filter_tools_by_category(all_tools, category)

    # 显示工具名称（调试用）
    tool_names = [t.name for t in filtered_tools]
    print(f"🔧 使用工具: {', '.join(tool_names)}")

    # 创建 Agent
    agent = create_structured_chat_agent(model, tools=filtered_tools, prompt=prompt)
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=filtered_tools,
        verbose=True,
        handle_parsing_errors=True,
    )

    agent_with_memory = RunnableWithMessageHistory(
        agent_executor,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    try:
        response = await agent_with_memory.ainvoke(
            {"input": task_content},
            config={"configurable": {"thread_id": "2", "session_id": "user_1"}}
        )
        result = response.get("output", "执行完成")

        print(f"✅ 任务完成: {result}")

        return {
            **state,
            "messages": [*state["messages"], AIMessage(content=f"[任务{current_idx + 1}] {result}")],
            "current_idx": current_idx + 1
        }
    except Exception as e:
        print(f"❌ 任务执行出错: {e}")
        return {
            **state,
            "messages": [*state["messages"], AIMessage(content=f"[任务{current_idx + 1}] 失败: {str(e)}")],
            "current_idx": current_idx + 1
        }


async def smart_home_agent(state: AgentState) -> AgentState:
    """智能家居控制执行器"""
    return await execute_current_task(state, "smart_home_control")


async def query_info_agent(state: AgentState) -> AgentState:
    """信息查询执行器"""
    return await execute_current_task(state, "query_info")


async def create_workflow():
    """构建 Workflow"""
    workflow = StateGraph(AgentState)

    workflow.add_node("router", agent_router)
    workflow.add_node("smart_home_control", smart_home_agent)
    workflow.add_node("query_info", query_info_agent)

    workflow.set_entry_point("router")

    # 从 router 到第一个任务或结束
    workflow.add_conditional_edges(
        "router",
        route_decision,
        {
            "smart_home_control": "smart_home_control",
            "query_info": "query_info",
            "end": END
        }
    )

    # 每个任务执行完后，继续路由到下一个任务或结束
    workflow.add_conditional_edges(
        "smart_home_control",
        route_decision,
        {
            "smart_home_control": "smart_home_control",
            "query_info": "query_info",
            "end": END
        }
    )

    workflow.add_conditional_edges(
        "query_info",
        route_decision,
        {
            "smart_home_control": "smart_home_control",
            "query_info": "query_info",
            "end": END
        }
    )

    return workflow.compile()


async def main():
    try:
        print("🏠 MCP 智能家居系统已连接！")
        print("💡 示例：'打开客厅灯并查询天气' 或 '你好'")

        workflow = await create_workflow()

        while True:
            user_input = input("\n你: ").strip()

            if user_input.lower() in {"exit", "quit", "退出"}:
                print("👋 再见！")
                break

            if not user_input:
                continue

            response = await workflow.ainvoke(
                {"input": user_input},
                config={"configurable": {"thread_id": "2", "session_id": "user_1"}}
            )

            messages = response.get("messages", [])
            if messages:
                print("\n🤖 AI:")
                for msg in messages:
                    if hasattr(msg, "content"):
                        print(f"  {msg.content}")
            else:
                print("🤖 AI: (无回复)")

    except Exception as e:
        print(f"❌ 发生错误：{e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 已退出。")