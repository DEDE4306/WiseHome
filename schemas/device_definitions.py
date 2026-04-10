from typing import TypedDict, Dict, List


class Device(TypedDict):
    type: str           # 设备类型
    default_state: str  # 默认状态
    meta: Dict          # 可选配置，如亮度、温度、音量范围

DEVICE_DEFINITIONS: List[Device] = [
    {"type": "light", "default_state": "off", "meta": {"brightness": [0, 100]}},
    {"type": "ac", "default_state": "off", "meta": {"temperature": [16, 30]}},
    {"type": "music", "default_state": "stopped", "meta": {"volume": [0, 100]}},
]