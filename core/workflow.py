import asyncio
from typing import List, Dict, Any, Callable

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, ModelResponse, wrap_model_call
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.constants import START, END
from langgraph.graph import add_messages, StateGraph
from pydantic import BaseModel, Field
from typing_extensions import Annotated, TypedDict, Literal

from core.model import create_model, Mongodb_checkpointer
from config.prompts import type_router_template, category_router_template, system_template, task_splitter_template, complex_task_template

# ============== 全局缓存 ==============
_tools_cache = None
_tools_filter_cache = {}
_agent_cache = {}
_agent_cache_lock = asyncio.Lock()

# ============= State 定义 ============
class AgentState(TypedDict):
    """Agent 状态定义，用于在 workflow 节点间传递数据"""
    messages: Annotated[list[AnyMessage], add_messages]     # 会话消息列表
    input: str      # 直接输入
    task_type: Literal["simple", "complex", "mixed"]        # 任务类型
    task_category: Literal["chat", "query_info", "smart_home_control"]    # 子任务类型
    sub_tasks: List[Dict[str, Any]]     # 处理后的子任务列表
    current_idx: int        # 当前执行到的子任务索引
    current_task: str        # 当前正在执行的任务

# ============== 路由模型定义 ==============
class RouteTaskType(BaseModel):
    """判断任务类型"""
    task_type: Literal["simple", "complex", "mixed"] = Field(
        description="任务类型：simple（单一操作），complex（多步协同），mixed（多个独立任务）"
    )

class RouteCategory(BaseModel):
    """判断子任务类别"""
    task_category: Literal["chat", "query_info", "smart_home_control"] = Field(
        description="子任务类别：chat（聊天）、query_info（查询信息）、smart_home_control（控制设备）"
    )

class TaskResult(BaseModel):
    """混合任务拆分结果，严格匹配指定的输出格式"""
    sub_tasks: List[Dict[str, str]] = Field(
        None,
        description="拆分后的简单任务列表，每个元素是仅包含'task'键的字典",
    )

# ============== 加载工具 =============
async def load_mcp_tools() -> List[BaseTool]:
    """从服务器加载 MCP 工具，如果获取失败则抛出异常"""
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
            raise RuntimeError("警告: 获取到的工具为空")
        return _tools_cache
    except Exception as e:
        raise RuntimeError(f"获取MCP工具失败: {e}") from e

# ============== llm 初始化 ==============
model = create_model()

type_router_llm = model.with_structured_output(RouteTaskType)

category_router_llm = model.with_structured_output(RouteCategory)

task_splitter_llm = model.with_structured_output(TaskResult)

complex_task_llm = model.with_structured_output(TaskResult)

# ============== 入口函数 ==============
def entry(state: AgentState):
    return {
        "input": state["messages"][-1].content
    }


# ============== 路由函数 ==============
def type_router(state: AgentState) -> dict:
    """路由，意图识别，判断任务类型"""

    result = type_router_llm.invoke([
        SystemMessage(content=type_router_template),
        HumanMessage(content=state["input"])
    ])

    print("判断当前任务类型: ", result.task_type)

    return {
        "task_type": result.task_type,
        "current_idx": 0,
        "sub_tasks": []
    }

def category_router(state: AgentState) -> dict:
    """路由，子任务类型识别"""
    message = state["current_task"]

    result = category_router_llm.invoke([
        SystemMessage(content=category_router_template),
        HumanMessage(content=f"任务：{message}")
    ])

    return {"task_category": result.task_category}

# ============= 处理函数 ==============
def simple_tasks(state: AgentState) -> dict:
    """任务列表处理
    如果有 sub_tasks，从 sub_tasks 中提取当前任务并更新索引
    如果没有，直接取用户消息作为 current_task
    
    Args:
        state: Agent 状态
        
    Returns:
        更新后的状态，包含 current_task 和 current_idx
    """
    sub_tasks = state.get("sub_tasks", [])
    current_idx = state.get("current_idx", 0)
    
    if sub_tasks:
        if current_idx < len(sub_tasks):
            current_task = sub_tasks[current_idx].get("task", "")
            return {
                "current_task": current_task,
                "current_idx": current_idx + 1
            }
        return {
            "current_task": "",
            "current_idx": current_idx + 1
        }
    else:
        return {
            "current_task": state["input"],
            "current_idx": current_idx + 1
        }


def task_splitter(state: AgentState) -> dict:
    """拆分混合任务为多个子任务"""
    result = task_splitter_llm.invoke([
        SystemMessage(content=task_splitter_template),
        HumanMessage(content=state["input"])
    ])

    print("任务拆分结果: ", "; ".join(f"{i+1}. {item['task']}" for i, item in enumerate(result.sub_tasks)))

    return {
        "sub_tasks": result.sub_tasks,
        "current_idx": 0
    }

def complex_tasks(state: AgentState) -> dict:
    result = complex_task_llm.invoke([
        SystemMessage(content=complex_task_template),
        HumanMessage(content=state["input"])
    ])

    print("查询任务: ", result.sub_tasks[0].get("task", ""))
    return {
        "sub_tasks": result.sub_tasks,
        "current_task": result.sub_tasks[0].get("task", ""),
        "current_idx": 0
    }

async def get_more_info(state: AgentState) -> dict:
    task = state["current_task"]
    tools = await filter_tools("query_info")

    agent = await get_agent_executor("query_info", tools)

    resp = await agent.ainvoke(
        {"messages": [HumanMessage(task)]},
        config={"configurable": {"thread_id": "1", "session_id": "user_1"}}
    )
    message = resp.get("messages", [])[-1].content
    input = state["input"]
    input = "用户需求: " + input + "。额外信息: " + message

    return {"input": input}

def generate_new_tasks(state: AgentState) -> dict:
    """拆分混合任务为多个子任务"""
    print("未拆分的混合任务: ", state["input"])

    result = task_splitter_llm.invoke([
        SystemMessage(content=task_splitter_template),
        HumanMessage(content=state["input"])
    ])

    print("任务拆分结果: ", "; ".join(f"{i+1}. {item['task']}" for i, item in enumerate(result.sub_tasks)))

    return {
        "sub_tasks": result.sub_tasks,
        "current_idx": 0
    }

# ============== 工具过滤 ==============
async def filter_tools(category: str) -> List[BaseTool]:
    """根据任务类型过滤工具
    
    Args:
        category: 任务类别，可选值为 'chat'、'query_info'、'smart_home_control'
        
    Returns:
        过滤后的工具列表
        
    规则:
        - chat: 返回空列表
        - query_info: 返回查询类工具
        - smart_home_control: 返回控制类工具
    """
    global _tools_filter_cache
    
    keyword_map = {
        "smart_home_control": [
            "turn_on_ac", "turn_off_ac", "set_ac_temperature",
            "add_ac_temperature", "minus_ac_temperature",
            "turn_on_light", "turn_off_light", "set_light_brightness",
            "add_light_brightness", "minus_light_brightness",
            "play_music", "stop_music",
        ],
        "query_info": [
            "get_ac_temperature", "get_light_brightness", "get_music_device",
            "get_user_rooms", "get_room_devices", "get_user_preferences",
            "get_time", "get_weather"
        ],
        "chat": []
    }

    all_tools = await load_mcp_tools()
    keywords = keyword_map.get(category, [])
    filtered = [t for t in all_tools if any(kw in t.name.lower() for kw in keywords)]
    _tools_filter_cache[category] = filtered or all_tools

    return _tools_filter_cache[category]

# ============== Agent 创建 =============
async def get_agent_executor(category: str, tools: List[BaseTool]):
    """获取指定类别的 agent，使用缓存避免重复创建
    
    Args:
        category: 任务类别
        tools: 该类别使用的工具列表
        
    Returns:
        Agent executor 实例
    """
    global _agent_cache
    
    if category in _agent_cache:
        return _agent_cache[category]
    
    async with _agent_cache_lock:
        if category in _agent_cache:
            return _agent_cache[category]
        
        from langchain.agents import create_agent
        
        agent = create_agent(model, tools=tools, system_prompt=system_template, checkpointer=Mongodb_checkpointer)
        _agent_cache[category] = agent
        return agent

# ============== Agent 执行节点 =============
async def smart_home_agent(state: AgentState):
    """智能家居控制 agent，负责执行设备控制操作"""
    task = state.get("current_task", {})
    print("正在执行任务: ", task)

    tools = await filter_tools("smart_home_control")

    agent = await get_agent_executor("smart_home_control", tools)

    resp = await agent.ainvoke(
        {"messages": [HumanMessage(task)]},
        config={"configurable": {"thread_id": "1", "session_id": "user_1"}}
    )

    return resp

async def query_info_agent(state: AgentState):
    """信息查询 agent，负责执行状态查询操作"""
    task = state.get("current_task", {})
    print("正在执行任务", task)

    tools = await filter_tools("query_info")

    agent = await get_agent_executor("query_info", tools)

    resp = await agent.ainvoke(
        {"messages": [HumanMessage(task)]},
        config={"configurable": {"thread_id": "1", "session_id": "user_1"}}
    )

    return resp

async def chat_agent(state: AgentState) -> AgentState:
    """聊天 agent，负责处理普通对话"""
    task = state.get("current_task", {})
    print("正在执行任务: ", task)
    
    tools = await filter_tools("chat")

    agent = await get_agent_executor("chat", tools)

    resp = await agent.ainvoke(
        {"messages": [HumanMessage(task)]},
        config={"configurable": {"thread_id": "1", "session_id": "user_1"}}
    )

    return resp

# ============= 路由决策函数 ================
def type_decision(state: AgentState):
    """"""
    # simple（单一操作），complex（多步协同），mixed（多个独立任务）"
    if state["task_type"] == "simple":
        return "category_router"
    elif state["task_type"] == "mixed":
        return "task_splitter"
    elif state["task_type"] == "complex":
        return "complex_tasks"

def category_decision(state: AgentState):
    """路由决策函数，根据任务类别决定下一步执行的 agent"""
    print("开始路由决策，当前 category:", state.get("task_category"))
    if state["task_category"] == "chat":
        return "chat_agent"
    elif state["task_category"] == "query_info":
        return "query_info_agent"
    elif state["task_category"] == "smart_home_control":
        return "smart_home_agent"

def build_workflow():
    """构建 workflow 图，包含路由节点和各个 agent 节点
    
    Returns:
        编译后的 workflow 实例
    """
    builder = StateGraph(AgentState)

    builder.add_node("entry", entry)
    builder.add_node("type_router", type_router)
    builder.add_node("task_splitter", task_splitter)
    builder.add_node("simple_tasks", simple_tasks)
    builder.add_node("complex_tasks", complex_tasks)
    builder.add_node("get_more_info", get_more_info)
    builder.add_node("generate_new_tasks", generate_new_tasks)
    builder.add_node("category_router", category_router)
    builder.add_node("smart_home_agent", smart_home_agent)
    builder.add_node("query_info_agent", query_info_agent)
    builder.add_node("chat_agent", chat_agent)

    builder.set_entry_point("entry")
    builder.add_edge("entry", "type_router")

    builder.add_conditional_edges(
        "type_router",
        type_decision,
        {
            "category_router": "simple_tasks",
            "task_splitter": "task_splitter",
            "complex_tasks": "complex_tasks"
        },
    )

    builder.add_edge("task_splitter", "simple_tasks")

    builder.add_conditional_edges(
        "simple_tasks",
        should_continue,
        {
            "continue": "category_router",
            "end": END,
        },
    )

    builder.add_edge("complex_tasks", "get_more_info")
    builder.add_edge("get_more_info", "generate_new_tasks")
    builder.add_edge("generate_new_tasks", "simple_tasks")
    
    # 从 category_router 到各个 agent
    builder.add_conditional_edges(
        "category_router",
        category_decision,
        {
            "smart_home_agent": "smart_home_agent",
            "query_info_agent": "query_info_agent",
            "chat_agent": "chat_agent",
        },
    )

    builder.add_edge("smart_home_agent", "simple_tasks")
    builder.add_edge("query_info_agent", "simple_tasks")
    builder.add_edge("chat_agent", "simple_tasks")

    workflow = builder.compile(checkpointer=Mongodb_checkpointer)
    return workflow

def should_continue(state: AgentState):
    """判断是否还有任务需要执行

    Returns:
        "continue" 或 "end"
    """
    sub_tasks = state.get("sub_tasks", [])
    current_idx = state.get("current_idx", 0)
    tasks_len = max(len(sub_tasks), 1)

    if current_idx <= tasks_len:
        print(f"当前执行任务: {current_idx}/{tasks_len}", )
        return "continue"
    return "end"

smart_home_workflow = build_workflow()