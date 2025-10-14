import json
import os
import uuid
from typing import Dict

from models.user_model import UserInfo, RoomInfo, DeviceState
from models.device_definitions import DEVICE_DEFINITIONS

class PersistentStore:
    """JSON 持久化存储"""
    def __init__(self, path: str = "../data/store.json"):
        self.path = path
        self._data: Dict[str, UserInfo] = {}
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self._data = data
        else:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)

    def _persist(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False,indent=2)

    def get_user(self, user_id: str) -> UserInfo:
        if user_id not in self._data:
            raise ValueError(f"用户 {user_id} 不存在")
        return self._data[user_id]

    def get_room(self, user_id: str, room_name: str) -> RoomInfo:
        user = self.get_user(user_id)
        for room in user["rooms"]:
            if room["name"] == room_name:
                return room
        raise ValueError(f"房间 {room_name} 不存在")

    def get_device(self, room: RoomInfo, device_type: str) -> DeviceState:
        for dev in room["devices"]:
            if dev["type"] == device_type:
                return dev
        raise ValueError(f"{room['name']} 没有 {device_type} 设备")

    def save(self):
        """公有方法：将内存数据保存到 JSON"""
        self._persist()

store = PersistentStore()

def get_user(user_id: str) -> UserInfo:
    return store.get_user(user_id)

def get_room(user_id: str, room_name: str) -> RoomInfo:
    return store.get_room(user_id, room_name)

def get_device(room: RoomInfo, device_type: str) -> DeviceState:
    return store.get_device(room, device_type)
