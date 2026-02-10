
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from core.workflow import smart_home_workflow
from core.speech_recognition import get_voice_input


# ========== 主循环 ==========
async def loop():
    print("MCP 智能家居系统已启动！示例: '打开客厅灯'，'查询天气'")

    while True:
        user_input = input("\n你: ").strip()
        # user_input = await get_voice_input()
        if user_input.lower() in {"exit", "quit", "退出"}:
            print("AI: 再见！")
            break

        response = await smart_home_workflow.ainvoke(
            {"messages": [HumanMessage(user_input)]},
            {"configurable": {"thread_id": "1"}},
        )

        all_msgs = response.get("messages", [])

        # 找到本轮输入位置（从后往前找最后一个匹配的 HumanMessage）
        current_round_msgs = []
        for msg in reversed(all_msgs):
            if isinstance(msg, HumanMessage) and msg.content == user_input:
                break
            if isinstance(msg, AIMessage):
                current_round_msgs.append( msg.content)
            if isinstance(msg, ToolMessage):
                current_round_msgs.append(msg.content)

        # TODO: 输出部分需要重新写了
        output = reversed(current_round_msgs)
        print("所有输出: ")
        for msg in output:
            if msg != "":
                print(msg)
