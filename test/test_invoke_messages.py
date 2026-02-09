from core.model import create_model, Mongodb_checkpointer
from langchain.agents import create_agent

llm = create_model()


agent = create_agent(
    model=llm,
    checkpointer=Mongodb_checkpointer
)

response = agent.invoke(
    {"messages": "你好"},
    {"configurable":{"thread_id": "1"}}
)

