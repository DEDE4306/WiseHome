from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory

from config.constants import MONGODB_URI, DB_NAME, CHAT_COLLECTION_NAME    

def get_session_history(session_id: str, history_size = 5) -> MongoDBChatMessageHistory:
    return MongoDBChatMessageHistory(
        session_id=session_id,
        connection_string=MONGODB_URI,
        database_name=DB_NAME,
        collection_name=CHAT_COLLECTION_NAME,
        history_size=history_size
    )

