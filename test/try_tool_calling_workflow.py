from langchain_core.tools import tool
from langchain.chat_models import init_chat_model

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

# 初始化模型
model = init_chat_model(model="claude-3-5-haiku-latest")
# 绑定工具
model_with_tools = model.bind_tools([multiply])

# 调用模型
response_message = model_with_tools.invoke("what's 42 x 7?")
# 获取工具调用信息
tool_call = response_message.tool_calls[0]

# 执行工具
multiply.invoke(tool_call)
# 输出 ToolMessage(content='294', name='multiply', tool_call_id=...)