from datetime import datetime
from fastmcp import FastMCP

mcp = FastMCP("WiseHomeControl")

@mcp.tool()
def turn_on_ac(room: str = "客厅") -> str:
    """开启指定房间的空调"""
    return f"{room}空调已开启（{datetime.now().strftime('%H:%M:%S')}）"

@mcp.tool()
def turn_off_ac(room: str = "客厅") -> str:
    """关闭指定房间的空调"""
    return f"{room}空调已关闭（{datetime.now().strftime('%H:%M:%S')}）"

@mcp.tool()
def turn_on_light(room: str = "客厅") -> str:
    """开启指定房间的灯光"""
    return f"{room}灯光已打开"

@mcp.tool()
def turn_off_light(room: str = "卧室") -> str:
    """关闭指定房间的灯光"""
    return f"{room}灯光已关闭"

@mcp.tool()
def play_music(song: str = "轻音乐") -> str:
    """播放指定音乐"""
    return f"正在播放《{song}》"

# 停止音乐
@mcp.tool()
def stop_music() -> str:
    """停止播放音乐"""
    return "音乐已停止播放"

# 设置温度
@mcp.tool()
def set_temperature(temp: float = 24.0) -> str:
    """设置指定房间的空调温度"""
    return f"室内温度已设置为 {temp}℃"

# 查询天气
@mcp.tool()
def get_weather(city: str = "上海") -> str:
    """查询指定城市的天气"""
    return f"{city}当前晴，气温 26℃，湿度 60%"

# 查询时间
@mcp.tool()
def get_time() -> str:
    """获取当前时间"""
    now = datetime.now()
    return f"当前时间：{now.strftime('%Y-%m-%d %H:%M:%S')}"

if __name__ == "__main__":
    mcp.run(transport="sse", port=8001)
