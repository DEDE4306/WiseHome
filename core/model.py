import os
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langgraph.checkpoint.mongodb import MongoDBSaver
from db.connection import MongoConnection, URI, DB_NAME, CHAT_COLLECTION_NAME

# ========== 创建模型 ==========
# 加载环境变量
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# 创建模型
def create_model():
    model = init_chat_model(
        "qwen3-32b",
        model_provider="openai",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=api_key,
        temperature=0.3,
        model_kwargs={"extra_body": {"enable_thinking": False}}
    )
    return model

# =========== 短期记忆 ===========
Mongodb_checkpointer = MongoDBSaver(
    client=MongoConnection.get_client(),
    db_name=DB_NAME,
)

