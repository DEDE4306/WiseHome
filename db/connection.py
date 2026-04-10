from pymongo import MongoClient

from config.constants import MONGODB_URI, DB_NAME

class MongoConnection:
    """单例 MongoDB 客户端"""
    _client = None

    @classmethod
    def get_client(cls) -> MongoClient:
        """获取 MongoDB 客户端实例"""
        if cls._client is None:
            cls._client = MongoClient(MONGODB_URI)
        return cls._client

    @classmethod
    def get_db(cls, db_name: str = DB_NAME):
        """获取 MongoDB 数据库实例"""
        return cls.get_client()[db_name]