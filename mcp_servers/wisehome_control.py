import atexit
from datetime import datetime
from fastmcp import FastMCP

from db.users import get_room, get_user, get_device, update_device_state,update_preference

mcp = FastMCP("WiseHomeControl")

# 暂时先写死吧
DEFAULT_USER_ID = "user_123"

@mcp.tool()
def turn_on_ac(room: str, temperature: int = 25,user_id: str = DEFAULT_USER_ID) -> str:
    """开启指定房间的空调"""
    try:
        r = get_room(user_id, room)
        ac = get_device(r, "ac")

        if ac["state"] == "on":
            return f"{room}空调已开启，当前温度 {ac['meta'].get('temperature')}℃"
        ac_temp = ac["meta"].get("temperature", temperature)
        ac["state"] = "on"
        ac["meta"]["temperature"] = ac_temp

        update_device_state(user_id, room, "ac", ac)

        return f"{room}空调已开启（温度 {ac_temp}℃）"
    except ValueError as e:
        return str(e)

@mcp.tool()
def turn_off_ac(room: str, user_id: str = DEFAULT_USER_ID) -> str:
    """关闭指定房间的空调"""
    try:
        r = get_room(user_id, room)
        ac = get_device(r, "ac")

        if ac["state"] == "off":
            return f"{room}空调已关闭"
        ac["state"] = "off"

        update_device_state(user_id, room, "ac", ac)

        return f"{room}空调已关闭"
    except ValueError as e:
        return str(e)

@mcp.tool()
def set_temperature(room: str, temperature: int, user_id: str = DEFAULT_USER_ID) -> str:
    """设置房间空调温度"""
    if not 16 <= temperature <= 30:
        return f"设置的温度 {temperature}℃ 超出温度范围 16~30℃"
    try:
        r = get_room(user_id, room)
        ac = get_device(r, "ac")
        if ac["state"] == "off":
            ac["state"] = "on"  # 未开空调，先开启
        ac["meta"]["temperature"] = temperature

        update_device_state(user_id, room, "ac", ac)

        return f"{room}空调温度已设置为 {temperature}℃"
    except ValueError as e:
        return str(e)

@mcp.tool()
def turn_on_light(room: str, brightness: int = 100, user_id: str = DEFAULT_USER_ID) -> str:
    """打开房间灯光"""
    try:
        r = get_room(user_id, room)
        light = get_device(r, "light")
        if light["state"] == "on":
            return f"{room}灯已开启，亮度为 {light['meta'].get('brightness')}"
        light["state"] = "on"
        light["meta"]["brightness"] = brightness

        update_device_state(user_id, room, "light", light)

        return f"{room}灯已打开，亮度为 {light['meta'].get('brightness')}"
    except ValueError as e:
        return str(e)

@mcp.tool()
def turn_off_light(room: str, user_id: str = DEFAULT_USER_ID) -> str:
    """关闭房间灯光"""
    try:
        r = get_room(user_id, room)
        light = get_device(r, "light")
        if light["state"] == "off":
            return f"{room}灯已关闭"
        light["state"] = "off"

        update_device_state(user_id, room, "light", light)

        return f"{room}灯已关闭"
    except ValueError as e:
        return str(e)

@mcp.tool()
def set_brightness(room: str, brightness: int, user_id: str = DEFAULT_USER_ID) -> str:
    """设置房间灯光亮度"""
    try:
        r = get_room(user_id, room)
        light = get_device(r, "light")
        if not (0 <= brightness <= 100):
            return f"亮度必须在 0 到 100 之间"
        if light["state"] == "off":
            light["state"] = "on"
        light["meta"]["brightness"] = brightness
        if brightness == 0:
            light["state"] = "off"
            return f"{room}灯亮度设置为 0，灯已关闭"

        update_device_state(user_id, room, "light", light)

        return f"{room}灯亮度已设置为 {brightness} "
    except ValueError as e:
        return str(e)

@mcp.tool()
def play_music(room: str, song: str, user_id: str = DEFAULT_USER_ID) -> str:
    """播放音乐"""
    try:
        r = get_room(user_id, room)
        music = get_device(r, "music")
        music["state"] = "playing"
        music["meta"]["song"] = song

        update_device_state(user_id, room, "music", music)

        return f"{room}的音箱正在播放《{song}》"
    except ValueError:
        return f"{room}没有音乐设备"

@mcp.tool()
def stop_music(room: str, user_id: str = DEFAULT_USER_ID) -> str:
    """停止播放音乐，传入房间名称，停止该房间的音乐"""
    try:
        r = get_room(user_id, room)
        music = get_device(r, "music")
        music["state"] = "stopped"

        update_device_state(user_id, room, "music", music)

        return f"{room} 音乐已停止"
    except ValueError:
        return f"{room} 没有音乐设备"

@mcp.tool()
def get_user_rooms(user_id: str = DEFAULT_USER_ID) -> str:
    """获取用户的房间信息"""
    try:
        user = get_user(user_id)
        rooms = [r["name"] for r in user["rooms"]]
        return f"用户 {user['name']} 拥有的房间：{', '.join(rooms)}"
    except ValueError as e:
        return str(e)

@mcp.tool()
def get_room_devices(room: str, user_id: str = DEFAULT_USER_ID) -> str:
    """查询房间设备"""
    try:
        r = get_room(user_id, room)
        devices = [f"{d['type']}({d['state']}, {d['meta']})" for d in r["devices"]]
        return f"{room} 的设备：{', '.join(devices)}"
    except ValueError as e:
        return str(e)

@mcp.tool()
def get_user_preferences(user_id: str = DEFAULT_USER_ID) -> str:
    """查询用户偏好"""
    try:
        user = get_user(user_id)
        return f"用户偏好：{user['preferences']}"
    except ValueError as e:
        return str(e)

@mcp.tool()
def store_user_preferences(preferences: dict, user_id: str = DEFAULT_USER_ID) -> str:
    """
    获取或更新用户偏好。如果传入 preferences，则更新；
    """
    try:
        user = get_user(user_id)
        user.setdefault("preferences", {}).update(preferences)

        update_preference(user)

        return f"已更新偏好：{preferences}"
    except ValueError as e:
        return str(e)

@mcp.tool()
def get_time() -> str:
    """获取当前系统时间"""
    now = datetime.now()
    return f"当前时间：{now.strftime('%Y-%m-%d %H:%M:%S')}"

@mcp.tool()
def get_weather(city: str = "上海") -> str:
    """获取天气"""
    return f"{city} 当前晴，气温 26℃，湿度 60%"

if __name__ == "__main__":
    mcp.run(transport="sse", port=8001)