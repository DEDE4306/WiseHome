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
    task_type: Literal["simple", "complex", "mixed"]        # 第一层任务类型
    task_category: Literal["chat", "query_info", "smart_home_control"]    # 子任务类型
    sub_tasks: List[Dict[str, Any]]     # 处理后的子任务列表
    current_task: Dict[str, Any]        # 当前正在执行的任务

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

# ============== 路由函数 ==============
def type_router(state: AgentState) -> dict:
    """路由，意图识别，判断任务类型"""
    message = state["messages"][-1]

    result = type_router_llm.invoke([
        SystemMessage(content=type_router_template),
        HumanMessage(content=message.content)
    ])

    return {"task_type": result.task_type, "current_task": message}

def category_router(state: AgentState) -> dict:
    """路由，子任务类型识别"""
    message = state["messages"][-1]


    result = category_router_llm.invoke([
        SystemMessage(content=category_router_template),
        HumanMessage(content=f"任务：{message}")
    ])

    return {"task_category": result.task_category, "current_task": message}

# ============== 工具过滤 ==============
async def filter_tools(category: str) -> List[BaseTool]:
    """根据任务类型过滤工具
    
    Args:
        category: 任务类别，可选值为 'chat'、'query_info'、'smart_home_control'
        
    Returns:
        过滤后的工具列表
        
    规则:
        - chat: 返回空列表（聊天不需要工具）
        - query_info: 返回查询类工具（包含 get、query、status、weather、list 等关键词）
        - smart_home_control: 返回控制类工具（包含 turn、set、open、close、switch、调节、开、关、设置、播放、play、stop、add、minus 等关键词）
    """
    global _tools_filter_cache
    
    if category in _tools_filter_cache:
        return _tools_filter_cache[category]
    
    keyword_map = {
        "smart_home_control": ["turn", "set", "open", "close", "switch", "调节", "开", "关", "设置", "播放", "play", "stop", "add", "minus"],
        "query_info": ["get", "query", "status", "weather", "list", "查询", "获取", "状态", "信息", "rooms", "room"],
        "chat": []
    }
    
    keywords = keyword_map.get(category, [])
    
    if category == "chat":
        _tools_filter_cache[category] = []
        return []
    
    all_tools = await load_mcp_tools()
    filtered = [t for t in all_tools if any(kw in t.name.lower() or kw in t.description.lower() for kw in keywords)]
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

# ============== Agent 节点 =============
async def smart_home_agent(state: AgentState):
    """智能家居控制 agent，负责执行设备控制操作"""
    task = state.get("current_task", {})
    print("当前 State: ", state)

    tools = await filter_tools("smart_home_control")
    print("当前使用工具: ", tools)

    agent = await get_agent_executor("smart_home_control", tools)

    resp = await agent.ainvoke(
        {"messages": task},
        config={"configurable": {"thread_id": "1", "session_id": "user_1"}}
    )

    return resp

async def query_info_agent(state: AgentState):
    """信息查询 agent，负责执行状态查询操作"""
    task = state.get("current_task", {})
    print("当前 State: ", state)

    tools = await filter_tools("query_info")
    print("当前使用工具: ", tools)

    agent = await get_agent_executor("query_info", tools)

    resp = await agent.ainvoke(
        {"messages": task},
        config={"configurable": {"thread_id": "1", "session_id": "user_1"}}
    )

    return resp


async def chat_agent(state: AgentState) -> AgentState:
    """聊天 agent，负责处理普通对话"""
    task = state.get("current_task", {})
    print("当前 State: ", state)
    
    tools = await filter_tools("chat")
    print("当前使用工具: ", tools)

    agent = await get_agent_executor("chat", tools)

    resp = await agent.ainvoke(
        {"messages": task},
        config={"configurable": {"thread_id": "1", "session_id": "user_1"}}
    )

    return resp

def route_decision(state: AgentState):
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

    builder.add_node("category_router", category_router)
    builder.add_node("smart_home_agent", smart_home_agent)
    builder.add_node("query_info_agent", query_info_agent)
    builder.add_node("chat_agent", chat_agent)

    builder.add_edge(START, "category_router")
    builder.add_conditional_edges(
        "category_router",
        route_decision,
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


