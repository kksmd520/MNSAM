import matplotlib.pyplot as plt

# # 数据
# optimizers = ['Adan', 'Adam', 'ASAM', 'SAM', 'SGDM', 'MNSAM']
# loss_ranking = [2, 3, 5, 4, 6, 1]
# flat_ranking = [2, 3, 4, 5, 6, 1]
# sizes = [1200, 1300, 1400, 1100, 1000, 1600]  # 气泡大小
# colors = ['#e74c3c', '#3498db', '#9acd32', '#dda0dd', '#f4c542', '#87cefa']
#
# # 创建图形
# plt.figure(figsize=(8, 6))
#
# # 画气泡图
# for i in range(len(optimizers)):
#     plt.scatter(loss_ranking[i], flat_ranking[i], s=sizes[i], color=colors[i], alpha=0.8)
#
# # 自定义图例（大小统一）
# from matplotlib.lines import Line2D
# legend_elements = [
#     Line2D([0], [0], marker='o', color='w', label=name,
#            markerfacecolor=color, markersize=10)
#     for name, color in zip(optimizers, colors)
# ]
#
# plt.legend(handles=legend_elements, loc='upper left')
#
# # 添加标签
# plt.xlabel('Loss Ranking')
# plt.ylabel('Flat Ranking')
#
# # 设置坐标轴范围
# plt.xlim(0, 7)
# plt.ylim(0, 7)
#
# plt.grid(False)
# plt.tight_layout()
# plt.show()
# ##
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# 构建数据
data = {
    'SAM':    [0.7712, 0.8672, 0.8660],
    'ASAM':   [0.8002, 0.8868, 0.8871],
    'ESAM':   [0.8509, 0.9182, 0.9181],
    'F-SAM':  [0.8397, 0.9178, 0.9174],
    'SSAM':   [0.8390, 0.9188, 0.9157],
    'MNSAM':  [0.8538, 0.9202, 0.9203],
}
metrics = ['miou', 'Macro F1 Score', 'Overall accuracy']

df = pd.DataFrame(data, index=metrics)

# 设置绘图风格
plt.figure(figsize=(10, 6))
sns.set(font_scale=1.2)

# 绘制热力图
ax = sns.heatmap(df, annot=True, fmt=".4g", cmap='viridis', cbar=True)

# 设置轴标签
plt.xlabel("Algorithm")
plt.ylabel("evaluation criteria")

# 自动调整布局
plt.tight_layout()
plt.show()
plt.savefig('heatmap.eps', dpi=300)  # 保存图片
## 气泡图
# import matplotlib.pyplot as plt
#
# # 数据
# optimizers = ['MNSAM', 'Adam', 'Adan', 'Asam', 'SAM', 'SGD']
# loss_ranking = [1, 3, 2, 5, 4, 6]
# smooth_ranking = [1, 3, 2, 4, 5, 6]
# params = [4, 3, 3, 3, 2, 1]  # 用于决定圆圈大小
#
# # 将 params 映射为合适的圆圈面积（放大因子）
# sizes = [p * 300 for p in params]  # 可调倍率，例如 300
#
# # 颜色对应每种算法
# colors = ['#87cefa', '#3498db', '#e74c3c', '#9acd32', '#dda0dd', '#f4c542']
#
# # 绘图
# plt.figure(figsize=(8, 6))
#
# for i in range(len(optimizers)):
#     plt.scatter(loss_ranking[i], smooth_ranking[i],
#                 s=sizes[i], color=colors[i], label=optimizers[i], alpha=0.8)
#
# # 图例大小统一
# from matplotlib.lines import Line2D
# legend_elements = [
#     Line2D([0], [0], marker='o', color='w', label=name,
#            markerfacecolor=color, markersize=10)
#     for name, color in zip(optimizers, colors)
# ]
# plt.legend(handles=legend_elements, loc='upper left')
#
# # 坐标轴标签
# plt.xlabel('Loss Ranking')
# plt.ylabel('Flat Ranking')  # 原 smooth ranking
#
# plt.xlim(0, 7)
# plt.ylim(0, 7)
# plt.tight_layout()
# plt.grid(False)
# plt.show()
# import matplotlib.pyplot as plt
# from matplotlib.lines import Line2D
# #
# # # 数据
# optimizers = ['MNSAM',  'ASAM', 'SAM', 'ESAM', 'F-SAM', 'SSAM']
# loss_ranking = [1,4,3, 2, 5,6]
# smooth_ranking = [1, 6, 5, 2, 3, 4]
# params = [4, 3, 2, 4, 4, 3]
#
# # 映射为圆圈面积
# sizes = [p * 300 for p in params]
#
# # 设置颜色
# colors = [
#     '#87cefa', '#3498db', '#e74c3c', '#9acd32', '#dda0dd',
#     '#f4c542', '#8a2be2', '#ff7f50', '#20b2aa'
# ]
#
# # 绘图
# plt.figure(figsize=(10, 7))
#
# for i in range(len(optimizers)):
#     plt.scatter(loss_ranking[i], smooth_ranking[i],
#                 s=sizes[i], color=colors[i], label=optimizers[i], alpha=0.8)
#
# # 图例统一大小，放在图框左上角
# legend_elements = [
#     Line2D([0], [0], marker='o', color='w', label=name,
#            markerfacecolor=color, markersize=10)
#     for name, color in zip(optimizers, colors)
# ]
# plt.legend(handles=legend_elements, loc='upper left')
#
# # 坐标轴设置
# plt.xlabel('Loss Ranking')
# plt.ylabel('Flat Ranking')
# plt.xlim(0, 7)
# plt.ylim(0, 7)
# plt.grid(False)
# plt.tight_layout()
# plt.show()
# plt.savefig('bubble_plot.eps', dpi=300)  # 保存图片