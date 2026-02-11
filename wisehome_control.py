from datetime import datetime
from fastmcp import FastMCP

from db.users import get_room, get_user, get_device, update_device_state, update_preference

mcp = FastMCP("WiseHomeControl")

# 暂时先写死吧
DEFAULT_USER_ID = "user_123"

@mcp.tool()
def turn_on_ac(room: str, temperature: int = 25,user_id: str = DEFAULT_USER_ID) -> str:
    """开启指定房间的空调
    
    适用场景：用户要求"打开XX空调"、"开启XX空调"、"XX空调开"等
    Args:
        room: 房间名称，如"客厅"、"卧室"等
        temperature: 可选，开启时设置的温度，默认为25度
        user_id: 用户ID，默认为当前用户
    Returns:
        操作结果字符串
    """
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
    """关闭指定房间的空调
    
    适用场景：用户要求"关闭XX空调"、"XX空调关"、"关掉XX空调"等
    Args:
        room: 房间名称，如"客厅"、"卧室"等
        user_id: 用户ID，默认为当前用户
    Returns:
        操作结果字符串
    """
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
def set_ac_temperature(room: str, temperature: int, user_id: str = DEFAULT_USER_ID) -> str:
    """设置房间空调温度到指定值（16-30度）
    
    适用场景：用户要求"将XX空调温度设置为XX度"、"XX空调调到XX度"等
    Args:
        room: 房间名称，如"客厅"、"卧室"等
        temperature: 目标温度，范围16-30度
        user_id: 用户ID，默认为当前用户
    Returns:
        操作结果字符串
    注意：如果用户说"调高/调低XX空调温度"，应使用 add_temperature 或 minus_temperature
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
def add_ac_temperature(room: str, temperature: int = 1, user_id: str = DEFAULT_USER_ID) -> str:
    """在当前温度基础上调高空调温度
    
    适用场景：用户要求"将XX空调温度调高XX度"、"XX空调温度加XX"、"调高XX空调"等
    Args:
        room: 房间名称，如"客厅"、"卧室"等
        temperature: 需要增加的温度数值，默认为1度
        user_id: 用户ID，默认为当前用户
    Returns:
        操作结果字符串
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
def minus_ac_temperature(room: str, temperature: int = 1, user_id: str = DEFAULT_USER_ID) -> str:
    """在当前温度基础上调低空调温度
    
    适用场景：用户要求"将XX空调温度调低XX度"、"XX空调温度减XX"、"调低XX空调"等
    Args:
        room: 房间名称，如"客厅"、"卧室"等
        temperature: 需要调低的温度数值，默认为1度
        user_id: 用户ID，默认为当前用户
    Returns:
        操作结果字符串
    示例："将客厅空调温度调低2度" → temperature参数为2
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
def get_ac_temperature(room: str, user_id: str = DEFAULT_USER_ID) -> str:
    """查询指定房间中空调的温度
    
    适用场景：用户要求"查询XX空调温度"、"XX空调多少度"、"XX空调温度"等
    Args:
        room: 房间名称，如"客厅"、"卧室"等
        user_id: 用户ID，默认为当前用户
    Returns:
        当前空调温度信息
    """
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
    """打开房间灯光并设置初始亮度
    
    适用场景：用户要求"打开XX灯"、"开XX灯"、"XX灯开"等
    Args:
        room: 房间名称，如"客厅"、"卧室"等
        brightness: 可选，开灯时的初始亮度，默认为100（最大亮度）
        user_id: 用户ID，默认为当前用户
    Returns:
        操作结果字符串
    注意：仅用于开灯操作，不要用此函数调节已开启灯的亮度。如需调节亮度，请使用 set_brightness、add_brightness 或 minus_brightness
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
    """关闭房间灯光
    
    适用场景：用户要求"关闭XX灯"、"关XX灯"、"XX灯关"、"关掉XX灯"等
    Args:
        room: 房间名称，如"客厅"、"卧室"等
        user_id: 用户ID，默认为当前用户
    Returns:
        操作结果字符串
    注意：仅用于关灯操作，不用于调节灯光亮度。如需调暗灯光亮度，请使用 minus_brightness
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
def set_light_brightness(room: str, brightness: int, user_id: str = DEFAULT_USER_ID) -> str:
    """设置房间灯光亮度到指定值（0-100）
    
    适用场景：用户要求"将XX灯亮度设置为XX"、"XX灯调到XX"、"XX灯亮度XX"等
    Args:
        room: 房间名称，如"客厅"、"卧室"等
        brightness: 目标亮度值，范围0-100，0表示关闭
        user_id: 用户ID，默认为当前用户
    Returns:
        操作结果字符串
    注意：如果用户说"调亮/调暗XX灯"，应使用 add_brightness 或 minus_brightness
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
def add_light_brightness(room: str, brightness: int = 10, user_id: str = DEFAULT_USER_ID) -> str:
    """在当前亮度基础上调亮灯光
    
    适用场景：用户要求"将XX灯调亮XX"、"XX灯亮度加XX"、"调亮XX灯"、"XX灯亮一点"等
    Args:
        room: 房间名称，如"客厅"、"卧室"等
        brightness: 需要增加的亮度数值，默认为10
        user_id: 用户ID，默认为当前用户
    Returns:
        操作结果字符串
    示例："将客厅灯调亮20" → brightness参数为20；"把灯调亮一点" → 使用默认值10
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
def minus_light_brightness(room: str, brightness: int = 10, user_id: str = DEFAULT_USER_ID) -> str:
    """在当前亮度基础上调暗灯光
    
    适用场景：用户要求"将XX灯调暗XX"、"XX灯亮度减XX"、"调暗XX灯"、"XX灯暗一点"等
    Args:
        room: 房间名称，如"客厅"、"卧室"等
        brightness: 需要调低的亮度数值，默认为10
        user_id: 用户ID，默认为当前用户
    Returns:
        操作结果字符串
    示例："将客厅灯调暗20" → brightness参数为20；"把灯调暗一点" → 使用默认值10
    注意：如果调暗后亮度为0，会自动关闭灯光
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
def get_light_brightness(room: str, user_id: str = DEFAULT_USER_ID) -> str:
    """查询指定房间中灯的亮度
    
    适用场景：用户要求"查询XX灯亮度"、"XX灯亮度多少"、"XX灯多亮"等
    Args:
        room: 房间名称，如"客厅"、"卧室"等
        user_id: 用户ID，默认为当前用户
    Returns:
        灯光亮度信息字符串
    """
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
    """在指定房间播放音乐
    
    适用场景：用户要求"在XX播放XX"、"XX放歌"、"播放XX音乐"等
    Args:
        room: 房间名称，如"客厅"、"卧室"等
        song: 歌曲名称
        user_id: 用户ID，默认为当前用户
    Returns:
        操作结果字符串
    """
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
    """停止指定房间的音乐播放
    
    适用场景：用户要求"停止XX音乐"、"XX音乐停"、"关掉XX音乐"等
    Args:
        room: 房间名称，如"客厅"、"卧室"等
        user_id: 用户ID，默认为当前用户
    Returns:
        操作结果字符串
    """
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
    """获取用户拥有音乐设备的房间列表
    
    适用场景：用户要求"哪些房间有音箱"、"哪里可以放歌"、"音乐设备在哪里"等
    Args:
        user_id: 用户ID，默认为当前用户
    Returns:
        包含音乐设备的房间列表字符串
    """
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
    """获取用户的房间信息
    
    适用场景：用户要求"有哪些房间"、"我的房间"、"房间列表"等
    Args:
        user_id: 用户ID，默认为当前用户
    Returns:
        用户拥有的房间列表字符串
    """
    try:
        user = get_user(user_id)
        rooms = [r["name"] for r in user["rooms"]]
        return f"用户 {user['name']} 拥有的房间：{', '.join(rooms)}"
    except ValueError as e:
        return str(e)

@mcp.tool()
def get_room_devices(room: str, user_id: str = DEFAULT_USER_ID) -> str:
    """查询指定房间中的设备信息
    
    适用场景：用户要求"XX房间有什么设备"、"XX房间设备"、"查看XX设备"等
    Args:
        room: 房间名称，如"客厅"、"卧室"等
        user_id: 用户ID，默认为当前用户
    Returns:
        房间设备信息字符串
    """
    try:
        r = get_room(user_id, room)
        devices = [f"{d['type']}({d['state']}, {d['meta']})" for d in r["devices"]]
        return f"{room} 的设备：{', '.join(devices)}"
    except ValueError as e:
        return str(e)

@mcp.tool()
def get_user_preferences(user_id: str = DEFAULT_USER_ID) -> str:
    """查询用户偏好设置
    
    适用场景：用户要求"我的偏好"、"用户设置"、"喜欢的音乐/温度/亮度"等
    Args:
        user_id: 用户ID，默认为当前用户
    Returns:
        用户偏好信息字符串，包括喜欢的音乐、空调温度、灯的亮度等
    """
    try:
        user = get_user(user_id)
        return f"用户偏好：{user['preferences']}"
    except ValueError as e:
        return str(e)

@mcp.tool()
def store_user_preferences(preferences: dict, user_id: str = DEFAULT_USER_ID) -> str:
    """更新用户偏好设置
    
    适用场景：用户要求"设置我的偏好"、"更新偏好"、"保存偏好"等
    Args:
        preferences: 偏好设置字典，包含要更新的偏好项，如喜欢的音乐、空调温度、灯的亮度等
        user_id: 用户ID，默认为当前用户
    Returns:
        操作结果字符串
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
    """获取当前系统时间
    
    适用场景：用户要求"现在几点"、"当前时间"、"几点了"等
    Args:无
    Returns:
        当前时间字符串，格式为"YYYY-MM-DD HH:MM:SS"
    """
    now = datetime.now()
    return f"当前时间：{now.strftime('%Y-%m-%d %H:%M:%S')}"

@mcp.tool()
def get_weather(city: str = "上海") -> str:
    """获取指定城市的天气信息
    
    适用场景：用户要求"天气如何"、"XX天气"、"查询天气"等
    Args:
        city: 城市名称，默认为"上海"
    Returns:
        天气信息字符串
    """
    return f"{city} 当前晴，气温 26℃，湿度 60%"

if __name__ == "__main__":
    mcp.run(transport="sse", port=8001)