from pymongo import MongoClient

URI = "mongodb://localhost:27017/"
DB_NAME = "wisehome_db"

class MongoConnection:
    """单例 MongoDB 客户端"""
    _client = None

    @classmethod
    def get_client(cls) -> MongoClient:
        if cls._client is None:
            cls._client = MongoClient(URI)
        return cls._client

    @classmethod
    def get_db(cls, db_name: str = DB_NAME):
        return cls.get_client()[db_name]