from db.connection import MongoConnection
from models.user_model import UserInfo, RoomInfo, DeviceState

class UserStore:
    def __init__(self, db_name="wisehome_db"):
        self.db = MongoConnection.get_db(db_name=db_name)
        self.users = self.db["users"]

    def get_user(self, user_id: str) -> UserInfo:
        """获取用户信息"""
        user = self.users.find_one({"user_id": user_id}, {"_id": False})
        if not user:
            raise ValueError(f"用户 {user_id} 不存在")
        return user

    def save_user(self, user: UserInfo):
        print("[DEBUG] 保存用户:", user["user_id"])
        result = self.users.update_one(
            {"user_id": user["user_id"]},
            {"$set": user},
            upsert=True
        )
        print("[DEBUG] Mongo 更新结果:", result.raw_result)

    def delete_user(self, user_id: str):
        """删除用户"""
        self.users.delete_one({"user_id": user_id})

    def get_room(self, user_id: str, room_name: str) -> RoomInfo:
        """查找房间"""
        user = self.get_user(user_id)
        for i, room in enumerate(user["rooms"]):
            if room["name"] == room_name:
                return user["rooms"][i]
        raise ValueError(f"房间 {room_name} 不存在")

    def get_device(self, room: RoomInfo, device_type: str) -> DeviceState:
        """查找设备"""
        for i, dev in enumerate(room["devices"]):
            if dev["type"] == device_type:
                return room["devices"][i]
        raise ValueError(f"{room['name']} 没有 {device_type} 设备")

    def update_device_state(self, user_id: str, room_name: str, device_type: str, new_state: dict):
        """原子更新设备状态"""
        result = self.users.update_one(
            {
                "user_id": user_id,
                "rooms.name": room_name,
                "rooms.devices.type": device_type
            },
            {"$set": {"rooms.$[r].devices.$[d]": new_state}},
            array_filters=[
                {"r.name": room_name},
                {"d.type": device_type}
            ]
        )
        print(f"[DEBUG] Mongo 更新结果: {result.raw_result}")

    def update_preference(self, user: UserInfo):
        self.users.update_one(
            {"user_id": user["user_id"]},
            {"$set": user},
            upsert=True
        )

# 创建全局 store 实例
store = UserStore()

# 兼容旧接口（可直接使用）
def get_user(user_id: str) -> UserInfo:
    return store.get_user(user_id)

def get_room(user_id: str, room_name: str) -> RoomInfo:
    return store.get_room(user_id, room_name)

def get_device(room: RoomInfo, device_type: str) -> DeviceState:
    return store.get_device(room, device_type)

def update_device_state(user_id: str, room_name: str, device_type: str, new_state: dict):
    return store.update_device_state(user_id, room_name, device_type, new_state)

def update_preference(user:UserInfo):
    return store.update_preference(user)

if __name__ == "__main__":
    """开启指定房间的空调"""
    try:
        r = get_room("user_123", "卧室")
        ac = get_device(r, "ac")

        ac_temp = ac["meta"].get("temperature", 15)
        ac["state"] = "on"
        ac["meta"]["temperature"] = ac_temp

        update_device_state("user_123", "卧室", "ac", ac)
    except Exception as e:
        print(str(e))
