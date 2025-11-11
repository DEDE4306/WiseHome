import asyncio
import json
import re
import os
from typing import TypedDict, Annotated, Literal, List, Callable, Coroutine, Any
from dotenv import load_dotenv

from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import add_messages, StateGraph, END

from db.chats import get_session_history
from template import system_template, router_template

# ========== 环境与模型初始化 ==========
# 加载环境变量
load_dotenv()
api_key = os.getenv("BAILIAN_API_KEY")

# 模型初始化
model = init_chat_model(
    "qwen3-1.7b",
    model_provider="openai",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=api_key,
    temperature=0.3,
    model_kwargs={"extra_body": {"enable_thinking": False}}
)

# 构造 prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", system_template),
    MessagesPlaceholder("chat_history"),
    ("human", "{input} {agent_scratchpad}"),
])

# ========== 类型定义 ==========
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]     # 自动保存历史信息
    sub_tasks: List[dict]   # 拆分后的子任务，形式为：[{"task": "xxx", "category": "xxx"}]
    current_idx: int        # 当前执行的任务编号
    is_chat: bool           # 是否为普通聊天
    context_info: str       # 额外的上下文查询信息
    input: str              # 用户原始输入

# ========== 通用工具函数 ==========
def safe_async(func: Callable[..., Coroutine[Any, Any, AgentState]]):
    """装饰器：捕获 agent 异常并保持状态"""
    async def wrapper(state: AgentState) -> AgentState:
        try:
            return await func(state)
        except Exception as e:
            print(f"[Error] {func.__name__} -> {e}")
            return {
                **state,
                "messages": [AIMessage(content=f"执行失败：{e}")],
                "current_idx": state.get("current_idx", 0) + 1
            }
    return wrapper

def extract_json(text: str) -> dict:
    """从文本中提取 JSON"""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception as e:
            print(f"JSON解析失败: {e}")
    return {}

# ========== 工具加载层 ==========
_tools_cache = None
_tools_filter_cache = {}

async def load_mcp_tools() -> List[BaseTool]:
    """加载 MCP 工具，如果获取失败则抛出异常"""
    global _tools_cache
    if _tools_cache is not None:
        return _tools_cache

    try:
        client = MultiServerMCPClient(
            {
                "wisehome": {
                    "url": "http://localhost:8001/sse",
                    "transport": "sse",
                }
            }
        )
        _tools_cache = await client.get_tools()
        if _tools_cache is None:
            raise RuntimeError("获取到的工具为空")
        return _tools_cache
    except Exception as e:
        # 捕获任何异常并抛出自定义错误
        raise RuntimeError(f"获取MCP工具失败: {e}") from e

async def filter_tools_by_category(category: str) -> List[BaseTool]:
    global _tools_filter_cache
    if category in _tools_filter_cache:
        return _tools_filter_cache[category]

    all_tools = await load_mcp_tools()
    keyword_map = {
        "smart_home_control": ["turn", "set", "open", "close", "switch", "调节", "开", "关", "设置", "播放", "play", "stop", "add", "minus"],
        "query_info": ["get", "query", "status", "weather", "list", "查询", "获取", "状态", "信息", "rooms", "room"]
    }

    keywords = keyword_map.get(category, [])
    filtered = [t for t in all_tools if any(kw in t.name.lower() or kw in t.description.lower() for kw in keywords)]
    _tools_filter_cache[category] = filtered or all_tools
    return _tools_filter_cache[category]

# ========== Agent 执行层 ==========
_agent_cache = {}
_agent_cache_lock = asyncio.Lock()

async def get_agent_executor(category: str, tools: List[BaseTool]):
    """获取指定类别的 agent executor（带缓存）"""
    if category in _agent_cache:
        return _agent_cache[category]

    async with _agent_cache_lock:
        if category in _agent_cache:
            return _agent_cache[category]

        agent = create_structured_chat_agent(model, tools=tools, prompt=prompt)
        executor = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5,
            max_execution_time=30,
        )

        agent_with_memory = RunnableWithMessageHistory(
            executor,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history"
        )
        _agent_cache[category] = agent_with_memory
        return agent_with_memory

async def llm_route(user_input: str, context_info: str = "") -> dict:
    """使用 LLM 判断意图并拆分任务"""
    history = get_session_history("user_1", 2)
    recent_msgs = await history.aget_messages()
    history_text = ""
    for msg in recent_msgs:
        if hasattr(msg, "content"):
            if isinstance(msg, HumanMessage):
                history_text += f"user: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                history_text += f"ai: {msg.content}\n"

    full_input = f"{user_input}\n{context_info}\n历史消息：\n{history}" if context_info or history else user_input
    # router 的提示词
    route_prompt = ChatPromptTemplate.from_messages([
        ("system", router_template),
        ("user", "{input}")
    ])
    response = await (route_prompt | model).ainvoke({"input": full_input})
    result = extract_json(response.content)
    if not result:
        result = {"type": "chat", "sub_tasks": [{"task": user_input, "category": "chat"}]}
    return result

# ========== Agent 节点实现 ==========
@safe_async
async def agent_router(state: AgentState) -> AgentState:
    """路由节点：判断意图并初始化状态"""
    result = await llm_route(state["input"], state.get("context_info", ""))
    # 获取 router 的任务拆分
    if result.get("type") == "task" and result.get("sub_tasks"):
        print(f"任务拆分: {json.dumps(result['sub_tasks'], ensure_ascii=False)}")
        return {
            **state,
            "is_chat": False,
            "sub_tasks": result["sub_tasks"],
            "current_idx": 0
        }
    # 未获取到任务拆分，当作普通聊天
    print(f"普通聊天: {json.dumps(result['sub_tasks'], ensure_ascii=False)}")
    return {
        **state,
        "is_chat": True,
        "sub_tasks": result["sub_tasks"],
        "current_idx": 0
    }

def route_decision(state: AgentState) -> Literal["smart_home_control", "query_info", "mixed", "re_route", "end", "chat"]:
    """路由决策函数，负责判断下一步路由"""
    idx, tasks = state.get("current_idx", 0), state.get("sub_tasks", [])
    if idx >= len(tasks):
        return "end"
    task = tasks[idx]
    if state.get("is_chat"):
        return "chat"
    if task.get("needs_re_route"):
        return "re_route"
    return task.get("category", "query_info")

async def execute_task(state: AgentState, category: str) -> AgentState:
    """执行具体任务"""
    idx, tasks = state["current_idx"], state["sub_tasks"]
    task = tasks[idx]["task"]

    print(f"执行任务 {idx + 1}: {task} [{category}]")

    # 获取所需工具
    tools = await filter_tools_by_category(category)
    tool_names = [t.name for t in tools]
    print(f"使用工具: {', '.join(tool_names)}")
    # 构造 agent
    agent = await get_agent_executor(category, tools)
    resp = await agent.ainvoke(
        {"input": task},
        config={"configurable": {"thread_id": "2", "session_id": "user_1"}}
    )

    return {
        **state,
        "messages": [AIMessage(content=resp.get("output", "执行完成"))],
        "current_idx": idx + 1
    }

@safe_async
async def mixed_task_agent(state: AgentState) -> AgentState:
    """混合任务执行器：执行查询，将结果附加到当前任务并标记重新路由"""
    current_idx = state["current_idx"]
    sub_tasks = state["sub_tasks"]
    current_task = sub_tasks[current_idx]
    task_content = current_task["task"]

    print(f"执行混合任务 {current_idx + 1}: {task_content}")
    print(f"策略：先查询信息，再基于结果重新拆分任务")

    # 执行查询
    query_tools = await filter_tools_by_category("query_info")

    tool_names = [t.name for t in query_tools]
    print(f"用查询工具: {', '.join(tool_names)}")

    # 获取查询执行 Agent
    agent_with_memory = await get_agent_executor("query_info", query_tools)

    try:
        # 执行查询
        response = await agent_with_memory.ainvoke(
            {"input": task_content},
            config={"configurable": {"thread_id": "2", "session_id": "user_1"}}
        )
        query_result = response.get("output", "").strip()
        print(f"查询完成: {query_result}")

        # 更新任务状态：附加查询结果并标记需要重新路由
        updated_task = {
            **current_task,
            "needs_re_route": True,
            "query_result": query_result
        }
        updated_sub_tasks = sub_tasks.copy()
        updated_sub_tasks[current_idx] = updated_task

        # 更新上下文，用于 re_route 阶段
        context_info = f"补充信息：{query_result}"

        return {
            **state,
            "sub_tasks": updated_sub_tasks,
            "context_info": context_info,
        }

    except Exception as e:
        print(f"查询执行出错: {e}")
        return {
            **state,
            "messages": [AIMessage(content=f"查询失败: {str(e)}")],
            "current_idx": current_idx + 1
        }


async def re_route_agent(state: AgentState) -> AgentState:
    """重新路由节点：基于查询结果重新拆分任务"""
    # 获取当前子任务
    current_idx = state["current_idx"]
    sub_tasks = state["sub_tasks"]
    current_task = sub_tasks[current_idx]

    original_task = current_task["task"]
    query_result = current_task.get("query_result", "")
    original_input = state.get("input", "")

    print(f"原始输入: '{original_input}'")
    print(f"重新路由任务: {original_task}")
    print(f"基于查询结果: {query_result}")

    context_info = f"补充信息：{query_result}" if query_result else ""

    # 调用 LLM 拆分任务
    route_result = await llm_route(original_task, context_info)

    if route_result.get("type") == "task":
        new_sub_tasks = route_result.get("sub_tasks", [])

        filtered_tasks = []
        for t in new_sub_tasks:
            if t.get("category") == "mixed":
                t = {**t, "category": "smart_home_control"}
            filtered_tasks.append(t)

        print(f"拆分结果：{json.dumps(new_sub_tasks, ensure_ascii=False)}")

        # 替换当前 mixed 任务为新拆分结果
        updated_sub_tasks = sub_tasks[:current_idx] + filtered_tasks + sub_tasks[current_idx + 1:]


        return {
            **state,
            "sub_tasks": updated_sub_tasks,
            "context_info": ""
        }

    # 如果 LLM 无法正确拆分，则跳过
    print("拆分失败，跳过此任务")
    return {
        **state,
        "current_idx": current_idx + 1,
        "context_info": ""
    }

@safe_async
async def smart_home_agent(state: AgentState) -> AgentState:
    return await execute_task(state, "smart_home_control")

@safe_async
async def query_info_agent(state: AgentState) -> AgentState:
    return await execute_task(state, "query_info")

@safe_async
async def chat_agent(state: AgentState) -> AgentState:
    idx, tasks = state["current_idx"], state["sub_tasks"]
    chat_content = tasks[idx]["task"]
    tools = []  # 聊天不需要工具，或者你可以加上某些辅助工具
    agent = await get_agent_executor("chat", tools)
    resp = await agent.ainvoke(
        {"input": chat_content},
        config={"configurable": {"thread_id": "2", "session_id": "user_1"}}
    )
    output = resp.get("output", "执行完成")
    return {
        **state,
        "messages": [AIMessage(content=output)],
        "current_idx": idx + 1
    }

# ========== 工作流层 ==========
async def create_workflow():
    wf = StateGraph(AgentState)
    nodes = {
        "router": agent_router,
        "smart_home_control": smart_home_agent,
        "query_info": query_info_agent,
        "chat": chat_agent,
        "mixed": mixed_task_agent,
        "re_route": re_route_agent,
    }
    for name, func in nodes.items():
        wf.add_node(name, func)
    wf.set_entry_point("router")

    edges = {k: route_decision for k in nodes}
    for node in edges:
        wf.add_conditional_edges(node, route_decision, {
            "smart_home_control": "smart_home_control",
            "query_info": "query_info",
            "mixed": "mixed",
            "re_route": "re_route",
            "chat":"chat",
            "end": END,
        })
    return wf.compile()

# ========== 主循环 ==========
async def main():
    print("MCP 智能家居系统已连接！示例：'打开客厅灯'，'查询天气'")
    wf = await create_workflow()
    while True:
        user_input = input("\n你: ").strip()
        if user_input.lower() in {"exit", "quit", "退出"}:
            print("再见！"); break
        response = await wf.ainvoke({"input": user_input, "context_info": ""})
        msgs = response.get("messages", [])
        print("\nAI:", end=" ")
        for msg in msgs:
            text = msg.content if hasattr(msg, "content") else str(msg)
            print(text.strip())

if __name__ == "__main__":
    asyncio.run(main())
