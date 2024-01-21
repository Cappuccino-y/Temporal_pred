from PyEMD import CEEMDAN,Visualisation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统使用SimHei
matplotlib.rcParams['font.family']='sans-serif'
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题
matplotlib.use('TkAgg')
# 从文件加载数据
with open("imfs.pkl", "rb") as file:
    imfs = pickle.load(file)
with open("res.pkl", "rb") as file:
    res= pickle.load(file)

# 设置显示范围，例如显示前1000个点
display_range = 1000  # 可以修改为您想要的范围

# 创建子图，共8行，每行2个子图，除了最后一行
fig = plt.figure(figsize=(12, 24))

# 绘制 IMF 分量
for i in range(7):
    ax1 = fig.add_subplot(8, 2, 2 * i + 1)
    ax2 = fig.add_subplot(8, 2, 2 * i + 2)

    ax1.plot(imfs[2 * i, :display_range])
    ax1.text(0.4, 0.95, f'IMF{2 * i + 1}', transform=ax1.transAxes, verticalalignment='top',color='red')
    ax1.grid(True)

    ax2.plot(imfs[2 * i + 1, :display_range])
    ax2.text(0.4, 0.95, f'IMF{2 * i + 2}', transform=ax2.transAxes, verticalalignment='top',color='red')
    ax2.grid(True)
    # if i == 6:
    #     ax1.set_xticklabels([])
    #     ax2.set_xticklabels([])


# 绘制残差在最后一行
ax_res = fig.add_subplot(8, 1, 8)
ax_res.plot(res[:display_range])
ax_res.set_title('残差',color='red')
ax_res.grid(True)
ax_res.set_xticklabels([])

# 调整布局以避免子图标题重叠
plt.tight_layout()

