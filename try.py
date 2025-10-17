import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from pymongo import MongoClient
from typing import List, Dict

# ==================== MongoDB 配置 ====================
MONGODB_URI = "mongodb://localhost:27017/"
DATABASE_NAME = "smart_home"
COLLECTION_NAME = "chat_history"

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
api_key_2 = os.getenv("BAILIAN_API_KEY")

def get_message_history(session_id: str):
    """获取会话历史"""
    return MongoDBChatMessageHistory(
        connection_string=MONGODB_URI,
        session_id=session_id,
        database_name=DATABASE_NAME,
        collection_name=COLLECTION_NAME,
    )

# ==================== 工具定义示例 ====================
@tool
def control_light(room: str, action: str) -> str:
    """控制灯光

    Args:
        room: 房间名称，如"客厅"、"卧室"
        action: 操作，"开"或"关"
    """
    return f"已{action}启{room}的灯"

@tool
def control_ac(room: str, action: str, temperature: int = 26) -> str:
    """控制空调

    Args:
        room: 房间名称
        action: 操作，"开"或"关"
        temperature: 温度（开启时有效）
    """
    if action == "开":
        return f"已开启{room}空调，温度设置为{temperature}度"
    return f"已关闭{room}空调"

@tool
def control_curtain(room: str, action: str) -> str:
    """控制窗帘

    Args:
        room: 房间名称
        action: 操作，"开"或"关"
    """
    return f"已{action}启{room}的窗帘"


# ==================== 核心优化策略 ====================

class SmartHomeOptimizer:
    """针对小模型的优化器"""

    def __init__(self):
        self.last_actions = []  # 记录最近的操作
        self.max_history = 3

    def parse_intent(self, user_input: str) -> List[Dict]:
        """
        预解析用户意图，识别多个操作
        这是关键优化：在调用 LLM 前先做规则提取
        """
        intents = []

        # 定义关键词映射
        device_keywords = {
            "灯": "light",
            "空调": "ac",
            "窗帘": "curtain"
        }

        room_keywords = {
            "客厅": "客厅",
            "卧室": "卧室",
            "厨房": "厨房",
            "书房": "书房"
        }

        action_keywords = {
            "开": "开",
            "关": "关",
            "打开": "开",
            "关闭": "关"
        }

        # 提取所有匹配项
        for device, device_type in device_keywords.items():
            if device in user_input:
                for room, room_name in room_keywords.items():
                    if room in user_input:
                        for action_text, action in action_keywords.items():
                            if action_text in user_input:
                                intents.append({
                                    "device": device_type,
                                    "room": room_name,
                                    "action": action,
                                    "original": f"{action_text}{room}{device}"
                                })

        return intents

    def check_duplicate_action(self, current_action: str) -> bool:
        """检查是否是重复操作"""
        if current_action in self.last_actions:
            return True

        self.last_actions.append(current_action)
        if len(self.last_actions) > self.max_history:
            self.last_actions.pop(0)

        return False

    def create_structured_prompt(self, intents: List[Dict]) -> str:
        """为小模型创建结构化提示"""
        if not intents:
            return ""

        prompt_parts = ["请执行以下操作："]
        for idx, intent in enumerate(intents, 1):
            prompt_parts.append(
                f"{idx}. 在{intent['room']}{intent['action']}启{intent['device']}"
            )

        return "\n".join(prompt_parts)

# ==================== Agent 配置 ====================

def create_smart_home_agent():
    """创建优化后的智能家居 Agent"""

    # 初始化 LLM（使用 Qwen3-0.6b）
    model = init_chat_model(
        "qwen3-0.6b",
        model_provider="openai",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=api_key_2,
        model_kwargs={"extra_body": {"enable_thinking": False}}
    )

    # 工具列表
    tools = [control_light, control_ac, control_curtain]

    # 关键：优化后的 Prompt 模板
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个智能家居助手。

重要规则：
1. 每次只执行用户当前请求的操作
2. 不要重复执行已经完成的操作
3. 简洁回复，不要多余解释
4. 如果用户提到多个设备，分别调用对应工具
5. 执行完操作后直接告知结果，不要询问是否需要其他帮助

当前可用设备：灯、空调、窗帘
可用房间：客厅、卧室、厨房、书房"""),

        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # 创建 Agent
    agent = create_tool_calling_agent(model, tools, prompt)

    # 创建执行器
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=5,  # 限制最大迭代次数
        early_stopping_method="generate",
        handle_parsing_errors=True,
    )

    return agent_executor


# ==================== 主控制逻辑 ====================

class SmartHomeController:
    """智能家居控制器"""

    def __init__(self, session_id: str = "default_user"):
        self.session_id = session_id
        self.optimizer = SmartHomeOptimizer()
        self.agent_executor = create_smart_home_agent()

        # 配置带历史记录的执行器
        self.agent_with_history = RunnableWithMessageHistory(
            self.agent_executor,
            get_message_history,
            input_messages_key="input",
            history_messages_key="chat_history",
        )

    def process_command(self, user_input: str) -> str:
        """处理用户命令（核心优化方法）"""

        # 步骤1：预解析意图
        intents = self.optimizer.parse_intent(user_input)

        # 步骤2：如果识别到多个操作，重构输入
        if len(intents) > 1:
            structured_input = self.optimizer.create_structured_prompt(intents)
            print(f"[优化] 检测到多个操作，重构为: {structured_input}")
            user_input = structured_input

        # 步骤3：调用 Agent
        try:
            response = self.agent_with_history.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": self.session_id}}
            )

            result = response.get("output", "操作失败")

            # 步骤4：记录操作，防止重复
            action_key = f"{user_input}_{result}"
            if self.optimizer.check_duplicate_action(action_key):
                return "该操作刚刚已经执行过了"

            return result

        except Exception as e:
            return f"执行出错: {str(e)}"

    def clear_history(self):
        """清除历史记录"""
        client = MongoClient(MONGODB_URI)
        db = client[DATABASE_NAME]
        collection = db[COLLECTION_NAME]
        collection.delete_many({"SessionId": self.session_id})
        print(f"已清除会话 {self.session_id} 的历史记录")


# ==================== 使用示例 ====================

if __name__ == "__main__":
    controller = SmartHomeController(session_id="user_001")

    print("智能家居控制系统启动")
    print("=" * 50)

    test_commands = [
        "打开客厅的灯",
        "打开客厅空调，温度26度",
        "同时打开客厅的灯和空调",  # 测试多操作
        "关闭客厅所有设备",
        "打开客厅的灯",  # 测试重复检测
    ]

    for cmd in test_commands:
        print(f"\n用户: {cmd}")
        result = controller.process_command(cmd)
        print(f"系统: {result}")
        print("-" * 50)

    # 清除历史（可选）
    # controller.clear_history()