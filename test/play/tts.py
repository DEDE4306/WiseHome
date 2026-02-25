
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建9x9围棋棋盘图
fig, ax = plt.subplots(1, 1, figsize=(12, 12))

# 棋盘背景 - 木质色
ax.set_facecolor('#E8C895')

# 绘制网格线
board_size = 9
for i in range(board_size):
    # 横线
    ax.plot([0, board_size-1], [i, i], 'k-', linewidth=1.5, alpha=0.8)
    # 竖线
    ax.plot([i, i], [0, board_size-1], 'k-', linewidth=1.5, alpha=0.8)

# 星位（9x9棋盘只有天元）
ax.plot(4, 4, 'ko', markersize=6)

# 根据用户提供的坐标
# 黑棋位置
black_stones = [
    # 第8行 y=8: 6,8
    (6, 8), (8, 8),
    # 第7行 y=7: 6,7,8
    (6, 7), (7, 7), (8, 7),
    # 第6行 y=6: 2,5,6,7
    (2, 6), (5, 6), (6, 6), (7, 6),
    # 第5行 y=5: 2,3,4,5,7
    (2, 5), (3, 5), (4, 5), (5, 5), (7, 5),
    # 第4行 y=4: 2,4,6,7,8
    (2, 4), (4, 4), (6, 4), (7, 4), (8, 4),
    # 第3行 y=3: 1,2,3,4,5,7
    (1, 3), (2, 3), (3, 3), (4, 3), (5, 3), (7, 3),
    # 第2行 y=2: 2,4,5
    (2, 2), (4, 2), (5, 2),
    # 第1行 y=1: 2,3,5
    (2, 1), (3, 1), (5, 1),
    # 第0行 y=0: 4
    (4, 0),
]

# 白棋位置
white_stones = [
    # 第8行 y=8: 5
    (5, 8),
    # 第7行 y=7: 0,2,3,4,5
    (0, 7), (2, 7), (3, 7), (4, 7), (5, 7),
    # 第6行 y=6: 0,1,3,4,8
    (0, 6), (1, 6), (3, 6), (4, 6), (8, 6),
    # 第5行 y=5: 1
    (1, 5),
    # 第4行 y=4: 1
    (1, 4),
    # 第3行 y=3: 6,8
    (6, 3), (8, 3),
    # 第2行 y=2: 7
    (7, 2),
    # 第1行 y=1: 无
    # 第0行 y=0: 1,2,3,5
    (1, 0), (2, 0), (3, 0), (5, 0),
]

# 绘制黑棋
for x, y in black_stones:
    circle = plt.Circle((x, y), 0.4, color='#1a1a1a', zorder=3)
    ax.add_patch(circle)
    # 添加高光效果
    highlight = plt.Circle((x-0.1, y+0.1), 0.12, color='#4a4a4a', zorder=4)
    ax.add_patch(highlight)

# 绘制白棋
for x, y in white_stones:
    circle = plt.Circle((x, y), 0.4, color='#f0f0f0', zorder=3)
    ax.add_patch(circle)
    circle_border = plt.Circle((x, y), 0.4, fill=False, color='#333', linewidth=1, zorder=4)
    ax.add_patch(circle_border)
    # 添加高光效果
    highlight = plt.Circle((x-0.1, y+0.1), 0.12, color='white', zorder=5)
    ax.add_patch(highlight)

# 设置坐标轴
ax.set_xlim(-0.5, board_size-0.5)
ax.set_ylim(-0.5, board_size-0.5)
ax.set_aspect('equal')
ax.axis('off')

# 添加坐标标签
for i in range(board_size):
    ax.text(i, -0.8, str(i), ha='center', va='top', fontsize=10, fontweight='bold')
    ax.text(-0.8, i, str(i), ha='right', va='center', fontsize=10, fontweight='bold')

# 添加标题
ax.set_title('9×9围棋终局局面（按用户提供坐标）\n黑子胜 18.5子 | 黑提14子 | 白提2子',
             fontsize=14, fontweight='bold', pad=20)

# 添加图例
legend_y = 9.2
ax.text(1, legend_y, f'● 黑棋 ({len(black_stones)}颗)', fontsize=12, fontweight='bold', color='#1a1a1a')
ax.text(5, legend_y, f'○ 白棋 ({len(white_stones)}颗)', fontsize=12, fontweight='bold', color='#333')

plt.tight_layout()

plt.show()

print("棋盘已生成！")
print(f"黑棋数量: {len(black_stones)}")
print(f"白棋数量: {len(white_stones)}")
print(f"总棋子数: {len(black_stones) + len(white_stones)}")