from pymongo import MongoClient

# MongoDB 连接配置
URI = "mongodb://localhost:27017/"  # MongoDB 服务器地址
DB_NAME = "wisehome_db"  # MongoDB 数据库名称

# 集合名称
CHAT_COLLECTION_NAME = "chat_logs"  # 聊天日志集合名称


class MongoConnection:
    """单例 MongoDB 客户端"""
    _client = None

    @classmethod
    def get_client(cls) -> MongoClient:
        """获取 MongoDB 客户端实例"""
        if cls._client is None:
            cls._client = MongoClient(URI)
        return cls._client

    @classmethod
    def get_db(cls, db_name: str = DB_NAME):
        """获取 MongoDB 数据库实例"""
        return cls.get_client()[db_name]