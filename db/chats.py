from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
from db.connection import URI, DB_NAME

def get_session_history(session_id: str, history_size = 5) -> MongoDBChatMessageHistory:
    return MongoDBChatMessageHistory(
        connection_string=URI,
        session_id=session_id,
        database_name=DB_NAME,
        collection_name="chat_logs",
        history_size=history_size
    )

