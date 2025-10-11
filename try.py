import asyncio


async def task_func(name, delay):
    print(f"任务 name={name}, delay={delay} === 111")
    await asyncio.sleep(delay)
    print(f"任务 name={name}, delay={delay} === 222")
    return f"任务 {name} 完成"


async def main():
    # 创建任务（立即加入事件循环，开始调度）
    task1 = asyncio.create_task(task_func("A", 1))
    task2 = asyncio.create_task(task_func("B", 2))

    print("任务状态:", task1.done())  # False（未完成）

    # 等待任务完成并获取结果
    result1 = await task1
    result2 = await task2

    print("结果:", result1, result2)  # 任务 A 完成 任务 B 完成
    print("任务状态:", task1.done())  # True（已完成）


asyncio.run(main())