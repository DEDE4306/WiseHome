from typing import TypedDict, List, Dict

class DeviceState(TypedDict):
    id: str             # 设备唯一ID
    type: str           # 设备类型：light/ac/music...
    state: str          # 当前状态：on/off/playing/paused...
    meta: Dict          # 可选配置：音量、温度、亮度等

class RoomInfo(TypedDict):
    name: str                     # 房间名称
    devices: List[DeviceState]    # 房间内所有设备状态列表
    meta: Dict                    # 其他信息：面积、楼层、备注等

class UserInfo(TypedDict):
    user_id: str                  # 用户唯一ID
    name: str                     # 用户姓名
    rooms: List[RoomInfo]         # 用户房间信息列表
    preferences: Dict             # 用户偏好设置

