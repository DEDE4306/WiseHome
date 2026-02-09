from typing import TypedDict, Annotated, Literal, List, Dict, Any
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langgraph.graph import add_messages, StateGraph, START, END
from langgraph.checkpoint.mongodb import MongoDBSaver
from pydantic import BaseModel, Field
from typing_extensions import Literal
from core.model import create_model

# ================== 1. 状态定义 ==================
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]  # 对话历史
    task_type: Literal["simple", "complex", "mixed"]     # 第一层任务类型
    sub_tasks: List[Dict[str, Any]]                    # 拆分后的子任务列表
    current_task: Dict[str, Any]                       # 当前正在处理的任务
    task_category: Literal["chat", "query_info", "smart_home_control"]  # 子任务类型


class RouteTaskType(BaseModel):
    task_type: Literal["simple", "complex", "mixed"] = Field(
        description="任务类型：simple（单一操作），complex（多步协同），mixed（多个独立任务）"
    )


# 创建带结构化输出的 LLM
model = create_model()
router_llm = model.with_structured_output(RouteTaskType)


def router_1(state: AgentState) -> dict:
    """第一层路由：判断任务是 simple、complex 还是 mixed"""
    last_msg = state["messages"][-1]

    system_prompt = (
        "你是一个智能家居任务分类器。请根据用户请求判断任务类型：\n"
        "- 如果是单一操作（如'开灯'），返回 'simple'\n"
        "- 如果是多个操作需协同完成（如'开灯并调暗'），返回 'complex'\n"
        "- 如果是多个独立任务（如'开灯并查天气'），返回 'mixed'\n"
        "请以 JSON 格式输出，只包含 task_type 字段。"
    )

    result = router_llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=last_msg.content)
    ])

    return {"task_type": result.task_type}

class RouteCategory(BaseModel):
    task_category: Literal["chat", "query_info", "smart_home_control"] = Field(
        description="子任务类别：chat（聊天）、query_info（查询信息）、smart_home_control（控制设备）"
    )

def router_2(state: AgentState) -> dict:
    """第二层路由：判断子任务属于哪一类"""
    task = state["current_task"]
    prompt = (
        "分析以下任务的意图：\n"
        f"任务：{task['description']}\n"
        "请从 chat, query_info, smart_home_control 中选择最合适的类别。\n"
        "请以 JSON 格式输出，只包含 task_category 字段。"
    )

    result = router_llm.invoke([
        SystemMessage(content=prompt),
    ])

    return {"task_category": result.task_category}

import re

def split_tasks(state: AgentState) -> dict:
    """将混合任务拆分成多个独立子任务"""
    tasks = []
    for task in state["sub_tasks"]:
        # 使用正则提取独立动作（简化版）
        if "并" in task["description"]:
            parts = re.split(r'[，。；并]', task["description"])
            for part in [p.strip() for p in parts if p.strip()]:
                tasks.append({
                    "description": part,
                    "category": "unknown",
                    "original": task["description"]
                })
        else:
            tasks.append(task)

    return {"sub_tasks": tasks}

async def handle_complex_task(state: AgentState) -> dict:
    """处理复杂任务，例如 '打开所有房间的灯'"""
    task = state["current_task"]
    description = task["description"]

    # 示例：如果包含“所有房间”，先查询用户房间
    if "所有房间" in description:
        # 假设我们有一个工具可以获取房间列表
        room_list = await get_user_room_list()  # 你需要实现这个工具
        new_task = {
            "description": f"打开{room_list}的所有灯",
            "category": "smart_home_control",
            "original": description
        }
        return {"sub_tasks": [new_task]}
    else:
        return {"sub_tasks": [task]}  # 不做修改

def handle_simple_task(state: AgentState) -> dict:
    """处理简单任务，直接执行"""
    task = state["current_task"]
    # 可以调用工具或返回响应
    response = f"已执行任务：{task['description']}"
    return {"messages": [HumanMessage(content=response)]}

builder = StateGraph(AgentState)

# 添加节点
builder.add_node("router_1", router_1)
builder.add_node("handle_complex_task", handle_complex_task)
builder.add_node("split_tasks", split_tasks)
builder.add_node("router_2", router_2)
builder.add_node("handle_simple_task", handle_simple_task)

# 添加边
builder.add_edge(START, "router_1")

# 条件边：根据 task_type 决定下一步
builder.add_conditional_edges(
    "router_1",
    lambda state: state["task_type"],
    {
        "simple": "handle_simple_task",
        "complex": "handle_complex_task",
        "mixed": "split_tasks"
    }
)

# 混合任务 → 拆分 → 处理
builder.add_edge("split_tasks", "router_2")
builder.add_edge("router_2", "handle_simple_task")  # 每个子任务都走一遍

# 复杂任务 → 生成新任务 → 拆分（可选）
builder.add_edge("handle_complex_task", "split_tasks")

# 结束
builder.add_edge("handle_simple_task", END)

# 编译图
graph = builder.compile(checkpointer=checkpointer)