import pandas as pd
import pyAgrum as gum
import bnlearn as bn
from pgmpy import readwrite

# bn = gum.loadBN("./insurance/insurance.bif")
# gum.generateCSV(bn, "./insurance/data11.csv", 50, True)
learner = gum.BNLearner("./insurance/data.csv")
learner.useScoreBDeu()
bn = learner.learnBN()

print(bn.nodes())
print(bn.connectedComponents())
print(bn.connectedComponents())

# # 导入模块
# import numpy as np
# import pandas as pd
# # path处填入npy文件具体路径
# npfile = np.load("D:/GAN结合Bayes/Bayes/bn_generation/insurance/DAG.npy")
# # 将npy文件的数据格式转化为csv格式
# np_to_csv = pd.DataFrame(data=npfile)
# # 存入具体目录下的np_to_csv.csv 文件
# np_to_csv.to_csv('D:/GAN结合Bayes/Bayes/bn_generation/insurance/DAG_data.csv', index=0)
