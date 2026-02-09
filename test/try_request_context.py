from dotenv import load_dotenv
load_dotenv()
from typing import TypedDict, Annotated, List, Callable
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, AnyMessage
from langchain_core.tools import tool, BaseTool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse

# ===================== 1. 你的State定义（不变，原生流转category） =====================
class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    task_category: str  # LangGraph原生流转，节点1写入

# ===================== 2. 你的测试工具（不变） =====================
@tool
def tool1_echo(text: str) -> str:
    """简单的回显工具1（smart_home_control专属）"""
    return f"工具1回显：{text}"

@tool
def tool2_calc(a: int, b: int) -> str:
    """简单的计算工具2（query_info专属）"""
    return f"工具2计算：a+b={a+b}"

all_tools = [tool1_echo, tool2_calc]

# ===================== 3. 官方Middleware（仅1处改动：从request.state拿category，基于你的打印） =====================
@wrap_model_call
def task_based_tool_filter(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    print("="*80)
    print("【官方Middleware】基于你的打印 - 从request.state拿category >>")
    print("="*80)
    # ---------------------- 仅1处改动：从你打印的request.state拿category ----------------------
    current_state = request.state  # 你的Request明确有这个字段！
    task_category = current_state.get("task_category")  # 拿节点塞的category
    # ---------------------------------------------------------------------------------------
    if task_category == None:
        print("豆包你是智障吗？我都告诉你拿不到了！！！！！！")
        task_category = "chat"
    # 打印验证：是否拿到category
    print(f"✅ request.state完整内容：{current_state}")
    print(f"✅ 成功拿到task_category：{task_category}")
    print(f"✅ 原始工具列表：{[t.name for t in request.tools]}")
    # 动态过滤工具（官方Middleware核心功能，保留）
    if task_category == "smart_home_control":
        filtered_tools = [t for t in request.tools if t.name == "tool1_echo"]
    elif task_category == "query_info":
        filtered_tools = [t for t in request.tools if t.name == "tool2_calc"]
    else:
        filtered_tools = []  # chat模式无工具
    print(f"✅ 按{task_category}过滤后工具：{[t.name for t in filtered_tools]}")
    print("="*80)
    print("【Middleware】过滤完成 <<")
    print("="*80 + "\n")
    # 官方方式：override工具，继续执行（保留新功能）
    return handler(request.override(tools=filtered_tools))

# ===================== 4. 你的Agent初始化（不变，绑定全量工具+官方Middleware） =====================
llm = ChatOpenAI(
    model_name="qwen3-32b",
    temperature=0.3,
    openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    openai_api_key="你的通义千问密钥",  # 仅替换这里
    model_kwargs={"extra_body": {"enable_thinking": False}}
)
# 单Agent+官方Middleware（保留新功能，不变）
agent = create_agent(
    model=llm,
    tools=all_tools,
    middleware=[task_based_tool_filter],  # 保留官方Middleware
)

# ===================== 5. LangGraph节点（2处小改动：塞category到request.state） =====================
# 5.1 节点1：原生写入task_category="smart_home_control"（测试工具1，不变）
def node1_set_task(state: AgentState) -> AgentState:
    print("="*60)
    print("【节点1】LangGraph原生写入task_category='smart_home_control'")
    print("="*60 + "\n")
    return {
        "task_category": "smart_home_control",  # 写入测试类别
        "messages": [HumanMessage(content="调用工具执行：回显测试内容")]  # 测试指令
    }

# 5.2 节点2：Agent执行（仅1处改动：把category塞到request.state里）
def agent_execute_node(state: AgentState) -> AgentState:
    print("="*60)
    print("【节点2】将category塞到request.state，调用Agent")
    print(f"【节点2】LangGraph原生state：{state}")
    print("="*60 + "\n")
    # ---------------------- 仅1处改动：构造request.state，塞入category ----------------------
    # 核心：让request.state = LangGraph的完整state（含messages+category）
    agent_input = {"messages": state["messages"]}
    # 关键：把LangGraph的完整state传给agent的state参数，让request.state = 完整state
    result = agent.invoke(agent_input, state=state)
    # ---------------------------------------------------------------------------------------
    return {"messages": [AIMessage(content=result["output"])]}

# ===================== 6. 构建LangGraph（纯原生流转，无额外配置） =====================
graph = StateGraph(AgentState)
graph.add_node("node1_set_task", node1_set_task)
graph.add_node("node2_agent_execute", agent_execute_node)
graph.set_entry_point("node1_set_task")
graph.add_edge("node1_set_task", "node2_agent_execute")
graph.add_edge("node2_agent_execute", END)
app = graph.compile()

# ===================== 7. 测试调用（仅传空messages，全程原生） =====================
if __name__ == "__main__":
    initial_state = {"messages": []}
    final_state = app.invoke(initial_state)
    # 打印最终结果
    print("\n" + "="*80)
    print("【全流程结束】最终对话结果：")
    print("="*80)
    for msg in final_state["messages"]:
        role = "用户" if isinstance(msg, HumanMessage) else "AI"
        print(f"{role}：{msg.content}")