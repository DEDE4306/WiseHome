from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from core.speech import get_recognizer
from core.workflow import build_workflow
from core.tts import tts_speech


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
async def loop(
        react_output: bool = False,
        using_speech: bool = False
):
    smart_home_workflow = build_workflow(verbose=react_output)
    print("MCP 智能家居系统已启动！")
    recognizer = get_recognizer()

    THREAD_ID = "persistent_user_session"

    while True:
        if using_speech:
            user_input = await recognizer.get_voice_input()
            print(f"你: {user_input}")
        else:
            user_input = input("你: ").strip()

        if user_input.lower() in {"exit", "quit", "退出"}:
            print("AI: 再见！")
            break

        if not user_input:
            continue

        if react_output:
            print("-" * 40)

        config = {"configurable": {"thread_id": THREAD_ID}}
        checkpoint_tuple = await smart_home_workflow.checkpointer.aget_tuple(config)
        if checkpoint_tuple is None:
            history_len = 0
        else:
            history_len = len(checkpoint_tuple.checkpoint["channel_values"]["messages"])

        all_new_messages = []

        try:
            async for chunk in smart_home_workflow.astream(
                    {"messages": [HumanMessage(content=user_input)]},
                    {"configurable": {"thread_id": THREAD_ID}},
                    stream_mode="values"
            ):
                current_msgs = chunk["messages"]
                new_msgs = current_msgs[history_len:]
                all_new_messages.extend(new_msgs)
                history_len = len(current_msgs)

            tool_call_msgs = []
            tool_response_msgs = []
            ai_msgs = []  # 这个是 List[str]

            for msg in all_new_messages:
                if isinstance(msg, AIMessage):
                    if msg.tool_calls:
                        calls = ", ".join([f"{tc['name']}({tc['args']})" for tc in msg.tool_calls])
                        tool_call_msgs.append(f"[AI Tool Calls]: {calls}")
                    if msg.content:
                        content = safe_content_str(msg.content)
                        if content:
                            ai_msgs.append(content)
                elif isinstance(msg, ToolMessage):
                    content = safe_content_str(msg.content)
                    if content:
                        tool_response_msgs.append(f"[Tool] {msg.name}: {content}")

            # === 修正：打印中间日志（只打一次）===
            if react_output:
                for log in tool_call_msgs:
                    print(log)
                for log in tool_response_msgs:
                    print(log)
                print("-" * 60)

            ai_msg = ""
            if ai_msgs:
                ai_msg = "；".join(ai_msgs) if ai_msgs else ""

            if ai_msg:
                if using_speech:
                    tts_speech(ai_msg)
                else:
                    print(f"AI: {ai_msg}")
            else:
                print("[Error] 发生错误，无 AI 回复")

            print("\n")

        except Exception as e:
            import traceback
            print(f"❌ 错误: {e}")
            if react_output:
                traceback.print_exc()