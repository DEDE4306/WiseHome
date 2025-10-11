from langgraph.store.memory import InMemoryStore
from typing import TypedDict

store = InMemoryStore()

class RoomInfo(TypedDict):
    aircon: bool

class UserInfo(TypedDict):
    rooms: dict[str, RoomInfo]
    preferences: dict[str, str]  # 如默认音乐、温度单位

