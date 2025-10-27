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

load_dotenv()
api_key = os.getenv("BAILIAN_API_KEY")

# 初始化模型
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


class AgentState(TypedDict):
    """状态定义"""
    messages: Annotated[list, add_messages]
    sub_tasks: List[dict]  # [{"task": "xxx", "category": "xxx"}]
    current_idx: int  # 当前执行到第几个任务
    is_chat: bool  # 是否为普通对话
    context_info: str  # 上下文信息（如查询结果）
    input: str  # 原始用户输入

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

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception as e:
            print(f"JSON 解析失败：{e}")

    return {}


async def llm_route(user_input: str, context_info: str = "") -> dict:
    """使用 LLM 判断意图并拆分任务"""
    full_input = f"{user_input}\n{context_info}" if context_info else user_input   # 如果有上下文信息，附加到输入中

    route_prompt = ChatPromptTemplate.from_messages([
        ("system", router_template),
        ("user", "{input}")
    ])

    response = await (route_prompt | model).ainvoke({"input": full_input})
    result = extract_json(response.content)

    if not result:
        print("路由解析失败，默认为普通对话")
        return {"type": "chat", "response": "抱歉，我没理解你的意思。"}

    result_type = result.get("type", "chat")

    if result_type == "task" and not result.get("sub_tasks"):
        print("任务拆分为空，降级为对话")
        return {"type": "chat", "response": "请告诉我你需要什么帮助？"}

    return result


async def agent_router(state: AgentState) -> AgentState:
    """路由节点：判断意图并初始化状态"""
    user_input = state["input"]
    context_info = state.get("context_info", "")

    route_result = await llm_route(user_input, context_info)

    result_type = route_result.get("type", "chat")

    print(f"意图识别：{result_type}")

    if result_type == "chat":
        response = route_result.get("response", "你好！")
        print(f"对话回复：{response}")
        return {
            **state,
            "is_chat": True,
            "sub_tasks": [],
            "current_idx": 0,
            "messages": [AIMessage(content=response)],
        }
    else:
        sub_tasks = route_result.get("sub_tasks", [])
        print(f"任务拆分：{json.dumps(sub_tasks, ensure_ascii=False)}")
        return {
            **state,
            "is_chat": False,
            "sub_tasks": sub_tasks,
            "current_idx": 0,
            "messages": state.get("messages", [])  # 保留已有消息
        }


def route_decision(state: AgentState) -> Literal["smart_home_control", "query_info", "mixed", "re_route", "end"]:
    """决策函数：根据当前任务的 category 路由"""
    if state.get("is_chat", False):
        return "end"

    current_idx = state.get("current_idx", 0)
    sub_tasks = state.get("sub_tasks", [])

    if current_idx >= len(sub_tasks):
        return "end"

    current_task = sub_tasks[current_idx]
    category = current_task.get("category", "query_info")

    # 检查是否需要重新路由
    if current_task.get("needs_re_route", False):
        print(f"需要重新路由 (任务 {current_idx + 1}/{len(sub_tasks)})")
        return "re_route"

    print(f"路由到: {category} (任务 {current_idx + 1}/{len(sub_tasks)})")

    return category


async def filter_tools_by_category(all_tools: List[BaseTool], category: str) -> List[BaseTool]:
    """根据类别筛选工具"""
    if category == "smart_home_control":
        keywords = ["turn", "set", "open", "close", "switch", "调节", "开", "关", "设置", "播放", "play", "stop", "音乐"]
    elif category == "query_info":
        keywords = ["get", "query", "status", "weather", "list", "查询", "获取", "状态", "信息", "rooms", "room"]
    else:
        return all_tools

    filtered_tools = [
        tool for tool in all_tools
        if any(kw in tool.name.lower() or kw in tool.description.lower() for kw in keywords)
    ]

    return filtered_tools if filtered_tools else all_tools


# 全局缓存：category -> (agent, agent_executor)
_agent_cache = {}
_agent_cache_lock = asyncio.Lock()

async def get_agent_executor_for_category(category: str, model, tools, prompt):
    """获取指定类别的 agent executor（带缓存）"""
    global _agent_cache

    if category in _agent_cache:
        return _agent_cache[category]

    async with _agent_cache_lock:
        # 双重检查，防止竞态
        if category in _agent_cache:
            return _agent_cache[category]

        # 创建 agent 和 executor
        agent = create_structured_chat_agent(model, tools=tools, prompt=prompt)
        agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
        )

        # 包装成带记忆的 Runnable
        agent_with_memory = RunnableWithMessageHistory(
            agent_executor,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
        )

        # 缓存
        _agent_cache[category] = agent_with_memory
        return agent_with_memory


async def execute_simple_task(state: AgentState, category: str) -> AgentState:
    """执行简单任务（单一查询或控制）"""
    current_idx = state["current_idx"]
    sub_tasks = state["sub_tasks"]
    current_task = sub_tasks[current_idx]
    task_content = current_task["task"]

    print(f"执行任务 {current_idx + 1}: {task_content} [{category}]")

    all_tools = await load_mcp_tools()
    filtered_tools = await filter_tools_by_category(all_tools, category)

    tool_names = [t.name for t in filtered_tools]
    print(f"使用工具: {', '.join(tool_names)}")

    agent_with_memory = await get_agent_executor_for_category(category, model, filtered_tools, prompt)

    try:
        response = await agent_with_memory.ainvoke(
            {"input": task_content},
            config={"configurable": {"thread_id": "2", "session_id": "user_1"}}
        )
        result = response.get("output", "执行完成")

        print(f"任务完成: {result}")

        return {
            **state,
            "messages": [AIMessage(content=result)],
            "current_idx": current_idx + 1
        }
    except Exception as e:
        print(f"任务执行出错: {e}")
        return {
            **state,
            "messages": [AIMessage(content=f"失败: {str(e)}")],
            "current_idx": current_idx + 1
        }


def extract_query_from_mixed_task(task: str) -> str:
    """从混合任务中提取查询部分"""
    if "所有房间" in task or "全部房间" in task:
        return "获取所有房间列表"
    elif "所有设备" in task or "全部设备" in task:
        return "获取所有设备列表"
    else:
        return task

async def mixed_task_agent(state: AgentState) -> AgentState:
    """混合任务执行器：先查询，将结果附加到原始任务，然后标记需要重新路由"""
    current_idx = state["current_idx"]
    sub_tasks = state["sub_tasks"]
    current_task = sub_tasks[current_idx]
    task_content = current_task["task"]

    print(f"执行混合任务 {current_idx + 1}: {task_content}")
    print(f"策略：先查询信息，再基于结果重新拆分任务")

    # 第一步：执行查询（获取房间列表等）
    all_tools = await load_mcp_tools()
    query_tools = await filter_tools_by_category(all_tools, "query_info")

    tool_names = [t.name for t in query_tools]
    print(f"用查询工具: {', '.join(tool_names)}")

    agent_with_memory = await get_agent_executor_for_category("query_info", model, query_tools, prompt)

    try:
        # 构造查询指令（提取查询部分）
        query_instruction = extract_query_from_mixed_task(task_content)
        print(f"查询指令: {query_instruction}")

        response = await agent_with_memory.ainvoke(
            {"input": query_instruction},
            config={"configurable": {"thread_id": "2", "session_id": "user_1"}}
        )
        query_result = response.get("output", "")

        print(f"查询完成: {query_result}")

        # 将查询结果作为上下文，标记需要重新路由
        updated_task = {
            **current_task,
            "needs_re_route": True,
            "query_result": query_result
        }

        # 更新任务列表
        updated_sub_tasks = sub_tasks.copy()
        updated_sub_tasks[current_idx] = updated_task

        # 构造完整上下文信息
        context_info = f"补充信息：{query_result}"

        return {
            **state,
            "sub_tasks": updated_sub_tasks,
            "context_info": context_info,
            "messages": [AIMessage(content=f"[查询结果] {query_result}")]
        }

    except Exception as e:
        print(f"查询执行出错: {e}")
        return {
            **state,
            "messages": [AIMessage(content=f"查询失败: {str(e)}")],
            "current_idx": current_idx + 1
        }


async def re_route_agent(state: AgentState) -> AgentState:
    """重新路由节点：基于查询结果，重新拆分任务"""
    current_idx = state["current_idx"]
    sub_tasks = state["sub_tasks"]
    current_task = sub_tasks[current_idx]

    original_task = current_task["task"]
    query_result = current_task.get("query_result", "")

    print(f"重新路由任务: {original_task}")
    print(f"基于查询结果: {query_result}")

    # 使用 LLM 重新拆分任务（附加查询结果作为上下文）
    context_info = f"补充信息：{query_result}"
    route_result = await llm_route(original_task, context_info)

    if route_result.get("type") == "task":
        new_sub_tasks = route_result.get("sub_tasks", [])
        print(f"拆分结果：{json.dumps(new_sub_tasks, ensure_ascii=False)}")

        # 替换当前任务为新拆分的任务列表
        updated_sub_tasks = sub_tasks[:current_idx] + new_sub_tasks + sub_tasks[current_idx + 1:]

        return {
            **state,
            "sub_tasks": updated_sub_tasks,
            "context_info": "",
        }
    else:
        # 拆分失败，跳过此任务
        print("拆分失败，跳过此任务")
        return {
            **state,
            "current_idx": current_idx + 1,
            "context_info": ""
        }


async def smart_home_agent(state: AgentState) -> AgentState:
    """智能家居控制执行器"""
    return await execute_simple_task(state, "smart_home_control")


async def query_info_agent(state: AgentState) -> AgentState:
    """信息查询执行器"""
    return await execute_simple_task(state, "query_info")

async def create_workflow():
    """构建 Workflow"""
    workflow = StateGraph(AgentState)

    workflow.add_node("router", agent_router)
    workflow.add_node("smart_home_control", smart_home_agent)
    workflow.add_node("query_info", query_info_agent)
    workflow.add_node("mixed", mixed_task_agent)
    workflow.add_node("re_route", re_route_agent)

    workflow.set_entry_point("router")

    workflow.add_conditional_edges(
        "router",
        route_decision,
        {
            "smart_home_control": "smart_home_control",
            "query_info": "query_info",
            "mixed": "mixed",
            "re_route": "re_route",
            "end": END
        }
    )

    workflow.add_conditional_edges(
        "smart_home_control",
        route_decision,
        {
            "smart_home_control": "smart_home_control",
            "query_info": "query_info",
            "mixed": "mixed",
            "re_route": "re_route",
            "end": END
        }
    )

    workflow.add_conditional_edges(
        "query_info",
        route_decision,
        {
            "smart_home_control": "smart_home_control",
            "query_info": "query_info",
            "mixed": "mixed",
            "re_route": "re_route",
            "end": END
        }
    )

    workflow.add_conditional_edges(
        "mixed",
        route_decision,
        {
            "smart_home_control": "smart_home_control",
            "query_info": "query_info",
            "mixed": "mixed",
            "re_route": "re_route",
            "end": END
        }
    )

    workflow.add_conditional_edges(
        "re_route",
        route_decision,
        {
            "smart_home_control": "smart_home_control",
            "query_info": "query_info",
            "mixed": "mixed",
            "re_route": "re_route",
            "end": END
        }
    )

    return workflow.compile()


async def main():
    try:
        print("MCP 智能家居系统已连接！")
        print("示例：'打开客厅灯'，'查询天气'")

        workflow = await create_workflow()

        while True:
            user_input = input("\n你: ").strip()

            if user_input.lower() in {"exit", "quit", "退出"}:
                print("👋 再见！")
                break

            if not user_input:
                continue

            response = await workflow.ainvoke(
                {"input": user_input, "context_info": ""},
                config={"configurable": {"thread_id": "2", "session_id": "user_1"}}
            )

            messages = response.get("messages", [])
            if messages:
                print("\nAI:")
                for msg in messages:
                    if hasattr(msg, "content"):
                        print(f"{msg.content}")
            else:
                print("AI: (无回复)")

    except Exception as e:
        print(f"发生错误：{e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n已退出。")