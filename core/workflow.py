import asyncio
from typing import List, Dict, Any, Callable

from langchain.agents.middleware import ModelRequest, ModelResponse, wrap_model_call
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage
from langgraph.graph import add_messages, StateGraph
from pydantic import BaseModel, Field
from typing_extensions import Annotated, TypedDict, Literal

from core.model import create_model
from config.prompts import type_router_template, category_router_template


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    task_type: Literal["simple", "complex", "mixed"]     # 第一层任务类型
    task_category: Literal["chat", "query_info", "smart_home_control"]  # 子任务类型
    sub_tasks: List[Dict[str, Any]]  # 处理后的子任务列表
    current_task: Dict[str, Any]  # 当前正在执行的任务


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

model = create_model()
type_router_llm = model.with_structured_output(RouteTaskType)
category_router_llm = model.with_structured_output(RouteCategory)

def type_router(state: AgentState) -> dict:
    """路由，意图识别，判断任务类型"""
    message = state["messages"][-1]

    result = type_router_llm.invoke([
        SystemMessage(content=type_router_template),
        HumanMessage(content=message.content)
    ])

    return {"task_type": result.task_type}

def category_router(state: AgentState) -> dict:
    """路由，子任务类型识别"""
    task = state["current_task"]

    result = category_router_llm.invoke([
        SystemMessage(content=category_router_template),
        HumanMessage(content=f"任务：{task}")
    ])

    return {"task_category": result.task_category}

async def smart_home_agent(state: AgentState) -> AgentState:
    pass

async def query_info_agent(state: AgentState) -> AgentState:
    pass

async def chat_agent(state: AgentState) -> AgentState:
    pass

def build_workflow():
    builder = StateGraph(AgentState)
    builder.add_node("category_router", category_router)
    builder.add_node("smart_home_agent", smart_home_agent)
    builder.add_node("query_info_agent", query_info_agent)
    builder.add_node("chat_agent", chat_agent)


