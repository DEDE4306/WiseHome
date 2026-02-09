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
from config.prompts import type_router_template, category_router_template, system_template

# ============== 全局缓存 ==============
_tools_cache = None
_tools_filter_cache = {}
_agent_cache = {}
_agent_cache_lock = asyncio.Lock()

# ============= State 定义 ============
class AgentState(TypedDict):
    """Agent 状态定义，用于在 workflow 节点间传递数据"""
    messages: Annotated[list[AnyMessage], add_messages]     # 会话消息列表
    task_type: Literal["simple", "complex", "mixed"]        # 任务类型
    task_category: Literal["chat", "query_info", "smart_home_control"]    # 子任务类型
    sub_tasks: List[Dict[str, Any]]     # 处理后的子任务列表
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

class TaskSplitResult(BaseModel):
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

# ============== 模型初始化 ==============
model = create_model()

type_router_llm = model.with_structured_output(RouteTaskType)

category_router_llm = model.with_structured_output(RouteCategory)

task_spliter_llm = model.with_structured_output(TaskSplitResult)

# ============== 路由函数 ==============
def type_router(state: AgentState) -> dict:
    """路由，意图识别，判断任务类型"""
    message = state["messages"][-1]

    result = type_router_llm.invoke([
        SystemMessage(content=type_router_template),
        HumanMessage(content=message.content)
    ])

    return {"task_type": result.task_type, "current_task": message.content}

def category_router(state: AgentState) -> dict:
    """路由，子任务类型识别"""
    message = state["messages"][-1]

    result = category_router_llm.invoke([
        SystemMessage(content=category_router_template),
        HumanMessage(content=f"任务：{message}")
    ])

    return {"task_category": result.task_category, "current_task": message.content}

# ============= 处理函数 ==============
def task_splitter(state: AgentState) -> dict:
    message = state["messages"][-1]

    result = task_spliter_llm.invoke([
        SystemMessage(content=type_router_template),
        HumanMessage(content=message.content)
    ])

    print("任务拆分原始输出: ", result)

    return result

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
        
        agent = create_agent(model, tools=tools, system_prompt=system_template)
        _agent_cache[category] = agent
        return agent

# ============== Agent 执行节点 =============
async def smart_home_agent(state: AgentState):
    """智能家居控制 agent，负责执行设备控制操作"""
    task = state.get("current_task", {})
    print("当前 category: ", state.get("task_category"))

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
    print("当前 category: ", state.get("task_category"))

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
    print("当前 category: ", state.get("task_category"))
    
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
        return "chat_agent"
    elif state["task_type"] == "mixed":
        return "task_spliter"
    # elif state["task_type"] == "smart_home_control":
    #     return "smart_home_agent"

def simple_tasks(state: AgentState):
    """
    这个函数的作用是，从
    sub_tasks: List[Dict[str, Any]]     # 处理后的子任务列表
    中提取 current_task
    然后送给 category_decision
    但是这样的话，是不是要在 state 中写一个 current_idx？但是我觉得就算写了 current_idx 也不太对；想象一下
    如果 这个节点在 graph 中走了两遍。。那应该怎么办，，
    current_task: str        # 当前正在执行的任务
    反正就是要在 state 中写入 current_task
    """
    pass


def category_decision(state: AgentState):
    """路由决策函数，根据任务类别决定下一步执行的 agent"""
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

    builder.add_node("type_router", type_router)
    builder.add_node()
    builder.add_node("category_router", category_router)
    builder.add_node("smart_home_agent", smart_home_agent)
    builder.add_node("query_info_agent", query_info_agent)
    builder.add_node("chat_agent", chat_agent)

    builder.add_edge(START, "category_router")
    builder.add_conditional_edges(
        "category_router",
        category_decision,
        {  # Name returned by route_decision : Name of next node to visit
            "smart_home_agent": "smart_home_agent",
            "query_info_agent": "query_info_agent",
            "chat_agent": "chat_agent",
        },
    )
    builder.add_edge("smart_home_agent", END)
    builder.add_edge("query_info_agent", END)
    builder.add_edge("chat_agent", END)

    workflow = builder.compile(checkpointer=Mongodb_checkpointer)
    return workflow

smart_home_workflow = build_workflow()


