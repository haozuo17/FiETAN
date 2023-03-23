import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_excel("log.xlsx")

# 全部的线段风格
styles = ['c:s', 'y:8', 'r:^', 'r:v', 'g:D', 'm:X', 'b:p', ':>']  # 其他可用风格 ':<',':H','k:o','k:*','k:*','k:*'
# 获取全部的图例
columns = [i[:-2] for i in data.columns]
n, m = data.shape

plt.figure(figsize=(10, 7))
# 设置字体
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 22})
plt.rc('legend', fontsize=15)

# 正式的进行画图
for i in range(0, m, 2):
    i_data = data.iloc[:, i:i + 2]
    x, y = i_data.values[:, 0], i_data.values[:, 1]
    plt.plot(x, y, styles[i // 2], markersize=8, label=columns[i])

# 设置图片的x,y轴的限制，和对应的标签
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel("gamma")
plt.ylabel("AUC")

# 设置图片的方格线和图例
plt.xticks(np.arange(0, 1, 0.1))  # x轴刻度，以10为一格
plt.yticks(np.arange(0, 1, 0.1))
plt.grid()
plt.legend(loc='lower right', framealpha=0.7)
plt.tight_layout()
# plt.show()

# 如果想保存图片，请把plt.show注释掉，然后把下面这行代码打开注释
plt.savefig("img.png", dpi=800)
