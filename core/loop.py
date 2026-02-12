from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from core.speech import get_recognizer
from core.workflow import build_workflow


# ========== 安全内容解析 ==========
def safe_content_str(content):
    if isinstance(content, str):
        return content.strip()
    elif isinstance(content, list):
        texts = [str(item.get("text", item) or "") for item in content if
                 isinstance(item, dict) and item.get("type") == "text"]
        return " ".join(texts).strip()
    return str(content).strip()


# ========== 主循环 ===========
async def loop(react_output: bool = False):
    smart_home_workflow = build_workflow(verbose=react_output)
    print("MCP 智能家居系统已启动！示例: '打开客厅灯'，'查询天气'")
    recognizer = get_recognizer()
    
    THREAD_ID = "persistent_user_session"

    while True:
        user_input = await recognizer.get_voice_input()
        print(f"你: {user_input}")
        if user_input.lower() in {"exit", "quit", "退出"}:
            print("AI: 再见！")
            break

        if not user_input:
            continue

        if react_output:
            print("-" * 40)

        all_messages_this_turn = []

        try:
            async for chunk in smart_home_workflow.astream(
                    {"messages": [HumanMessage(content=user_input)]},
                    {"configurable": {"thread_id": THREAD_ID}},
                    stream_mode="updates"
            ):
                for node_name, node_data in chunk.items():
                    if "messages" in node_data:
                        all_messages_this_turn.extend(node_data["messages"])

            recent_msgs = all_messages_this_turn[-10:]
            ai_tool_msgs = []

            for msg in reversed(recent_msgs):
                if isinstance(msg, HumanMessage):
                    break
                ai_tool_msgs.append(msg)

            if react_output:
                for msg in reversed(ai_tool_msgs):
                    if isinstance(msg, AIMessage):
                        if msg.tool_calls:
                            tool_calls_str = ", ".join([f"{tc['name']}({tc['args']})" for tc in msg.tool_calls])
                            print(f"[AI Tool Calls]: {tool_calls_str}")

                        elif msg.content:
                            print(f"AI: {safe_content_str(msg.content)}")

                    elif isinstance(msg, ToolMessage):
                        print(f"[Tool] {msg.name}: {safe_content_str(msg.content)}")

                print("-" * 60)

            else:
                ai_msg = ""
                for msg in reversed(ai_tool_msgs):
                    if isinstance(msg, AIMessage) and msg.content:
                        ai_msg = safe_content_str(msg.content)
                        break

                if ai_msg:
                    print(f"AI: {ai_msg}")
                else:
                    print("[Error] 发生错误，无 AI 回复")

        except Exception as e:
            print(f"❌ 错误: {e}")