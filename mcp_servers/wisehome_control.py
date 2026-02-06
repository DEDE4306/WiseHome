from datetime import datetime
from fastmcp import FastMCP

from db.users import get_room, get_user, get_device, update_device_state, update_preference

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
    """
    支持设置房间空调温度到指定值（16-30）。
    适用场景：设置温度到特定温度，如"将客厅空调温度设置为 25 度"。
    如果用户说“调高/调低 XX 空调温度”，应使用 add_temperature 或 minus_temperature。
    """
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
def add_temperature(room: str, temperature: int = 1, user_id: str = DEFAULT_USER_ID) -> str:
    """
    在当前温度基础上调高空调温度，输入 temperature 为需要增加的温度大小
    适用场景："将客厅空调温度调高 2 度" → 当前温度 +2，temperature 为 2
    如果不传递参数，默认调整 1 度
    """
    try:
        r = get_room(user_id, room)
        ac = get_device(r, "ac")
        if ac["state"] == "off":
            ac["state"] = "on"  # 未开空调，先开启
        cur_temperature = ac["meta"]["temperature"]
        final_temperature = min(cur_temperature + temperature, 30)
        ac["meta"]["temperature"] = final_temperature

        update_device_state(user_id, room, "ac", ac)

        return f"{room}空调温度已从 {cur_temperature} 调整为 {final_temperature}℃"
    except ValueError as e:
        return str(e)

@mcp.tool()
def minus_temperature(room: str, temperature: int = 1, user_id: str = DEFAULT_USER_ID) -> str:
    """
    在当前温度基础上调低空调温度，输入 temperature 为需要调低的温度大小
    适用场景："将客厅空调温度调低 2 度" → 当前温度 -2，tempeature 为 2
    如果不传递参数，默认调整 1 度
    """
    try:
        r = get_room(user_id, room)
        ac = get_device(r, "ac")
        if ac["state"] == "off":
            ac["state"] = "on"  # 未开空调，先开启
        cur_temperature = ac["meta"]["temperature"]
        final_temperature = max(cur_temperature - temperature, 16)
        ac["meta"]["temperature"] = final_temperature

        update_device_state(user_id, room, "ac", ac)

        return f"{room}空调温度已从 {cur_temperature} 调整为 {final_temperature}℃"
    except ValueError as e:
        return str(e)

@mcp.tool()
def get_temperature(room: str, user_id: str = DEFAULT_USER_ID) -> str:
    """查询指定房间中空调的温度"""
    try:
        r = get_room(user_id, room)
        ac = get_device(r, "ac")

        if ac["state"] == "off":
            return f"{room}空调未打开。"

        temperature = ac["meta"]["temperature"]
        return f"{room}空调当前温度为 {temperature}"

    except ValueError as e:
        return str(e)

@mcp.tool()
def turn_on_light(room: str, brightness: int = 100, user_id: str = DEFAULT_USER_ID) -> str:
    """
    打开房间灯光并设置初始亮度。
    仅用于开灯操作，请不要用这个函数来调节已开启的灯的亮度。
    如需调节亮度，请使用 set_brightness、add_brightness 或 minus_brightness。
    """
    try:
        r = get_room(user_id, room)
        light = get_device(r, "light")
        if light["state"] == "on":
            return f"{room}灯已开启，当前亮度为 {light['meta'].get('brightness')}，"
        light["state"] = "on"
        light["meta"]["brightness"] = brightness

        update_device_state(user_id, room, "light", light)

        return f"{room}灯已打开，亮度为 {light['meta'].get('brightness')}"
    except ValueError as e:
        return str(e)

@mcp.tool()
def turn_off_light(room: str, user_id: str = DEFAULT_USER_ID) -> str:
    """、
    关闭房间灯光
    仅用于关灯操作，不用于调节灯光的亮度，如果要调暗灯光亮度请用 minus_brightness。
    如需调节亮度，请使用 set_brightness、add_brightness 或 minus_brightness。
    """
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
    """
    设置房间灯光亮度到指定值（0-100）。
    适用场景：设置灯光到特定亮度，如"将灯光设置为50"，"将灯光调整到 50"
    如果用户说"调亮/调暗XX"，应使用 add_brightness 或 minus_brightness。
    """
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
def add_brightness(room: str, brightness: int = 10, user_id: str = DEFAULT_USER_ID) -> str:
    """
    在当前亮度基础上调亮灯光。
    适用场景：
    - "将客厅灯调亮20" → 当前亮度+20
    - "把灯调亮一点" → 当前亮度+10（默认值）
    - "增加亮度30" → 当前亮度+30
    如果灯是关闭的，会先开灯（亮度设为0），然后增加指定亮度。
    """
    try:
        r = get_room(user_id, room)
        light = get_device(r, "light")
        cur_brightness = light["meta"]["brightness"]
        total_brightness = min(cur_brightness + brightness, 100)
        if light["state"] == "off":
            light["state"] = "on"
        light["meta"]["brightness"] = total_brightness
        if brightness == 0:
            light["state"] = "off"
            return f"{room}灯亮度设置为 0，灯已关闭"

        update_device_state(user_id, room, "light", light)

        return f"{room}灯亮度已从 {cur_brightness} 调至 {total_brightness}"
    except ValueError as e:
        return str(e)

@mcp.tool()
def minus_brightness(room: str, brightness: int = 10, user_id: str = DEFAULT_USER_ID) -> str:
    """
    在当前亮度基础上调暗灯光。
    适用场景：
    - "将客厅灯调暗20" → 当前亮度-20
    - "把灯调暗一点" → 当前亮度-10（默认值）
    - "降低亮度30" → 当前亮度-30
    如果调暗后亮度为 0，会自动关闭灯光，不需要手动关闭。
    """
    try:
        r = get_room(user_id, room)
        light = get_device(r, "light")

        if light["state"] == "off":
            return f"{room}灯已关闭，无法调暗亮度。请先开灯。"

        cur_brightness = light["meta"]["brightness"]
        total_brightness = max(cur_brightness - brightness, 0)
        if light["state"] == "off":
            light["state"] = "on"
        light["meta"]["brightness"] = total_brightness
        if total_brightness == 0:
            light["state"] = "off"
            return f"{room}灯亮度设置为 0，灯已关闭"

        update_device_state(user_id, room, "light", light)

        return f"{room}灯亮度已从 {cur_brightness} 调至 {total_brightness}"
    except ValueError as e:
        return str(e)

@mcp.tool()
def get_brightness(room: str, user_id: str = DEFAULT_USER_ID) -> str:
    """查询指定房间中灯的亮度"""
    try:
        r = get_room(user_id, room)
        light = get_device(r, "light")

        if light["state"] == "off":
            return f"{room}灯未打开。"

        brightness = light["meta"]["brightness"]
        return f"{room}灯现在是开启状态，亮度为 {brightness}"

    except ValueError as e:
        return str(e)

@mcp.tool()
def play_music(room: str, song: str, user_id: str = DEFAULT_USER_ID) -> str:
    """播放音乐，传入房间名称和歌曲名，在指定房间播放音乐"""
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
    """停止播放音乐，传入房间名称，关闭该房间的音乐"""
    try:
        r = get_room(user_id, room)
        music = get_device(r, "music")
        music["state"] = "stopped"

        update_device_state(user_id, room, "music", music)

        return f"{room} 音乐已停止"
    except ValueError:
        return f"{room} 没有音乐设备"


@mcp.tool()
def get_music_device(user_id: str = DEFAULT_USER_ID) -> str:
    """获取用户拥有音乐设备的房间列表"""
    try:
        user = get_user(user_id)
        rooms_with_music = []

        # 遍历所有房间
        for room in user["rooms"]:
            room_name = room["name"]
            devices = room["devices"]

            # 检查该房间是否有音乐设备
            has_music = any(device["type"] == "music" for device in devices)

            if has_music:
                rooms_with_music.append(room_name)

        # 格式化输出
        if not rooms_with_music:
            return "用户没有音乐设备"
        elif len(rooms_with_music) == 1:
            return f"用户的音乐设备位于{rooms_with_music[0]}"
        else:
            return f"用户的音乐设备位于{'、'.join(rooms_with_music)}"

    except ValueError as e:
        return str(e)



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
    """查询房间中的设备信息"""
    try:
        r = get_room(user_id, room)
        devices = [f"{d['type']}({d['state']}, {d['meta']})" for d in r["devices"]]
        return f"{room} 的设备：{', '.join(devices)}"
    except ValueError as e:
        return str(e)

@mcp.tool()
def get_user_preferences(user_id: str = DEFAULT_USER_ID) -> str:
    """查询用户偏好，包括喜欢的音乐，喜欢的空调温度、灯的亮度等"""
    try:
        user = get_user(user_id)
        return f"用户偏好：{user['preferences']}"
    except ValueError as e:
        return str(e)

@mcp.tool()
def store_user_preferences(preferences: dict, user_id: str = DEFAULT_USER_ID) -> str:
    """更新用户偏好。传入 preferences 更新；"""
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