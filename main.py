import pyAgrum as gum
from ctgan.synthesizers.ctgan import CTGAN
import pandas as pd
import csv
import random
from pgmpy.estimators import BDeuScore
from pgmpy.estimators import PC, HillClimbSearch
import igraph as ig
import numpy as np
from pgmpy import readwrite
from pgmpy.models import BayesianNetwork
from matplotlib import pyplot as plt
import os


# 贝叶斯结构学习计算函数
def is_dag(W):
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    return G.is_dag()


def count_accuracy(B_true, B_est):
    if (B_est == -1).any():  # cpdag
        if not ((B_est == 0) | (B_est == 1) | (B_est == -1)).all():
            raise ValueError('B_est should take value in {0,1,-1}')
        if ((B_est == -1) & (B_est.T == -1)).any():
            raise ValueError('undirected edge should only appear once')
    else:  # dag
        if not ((B_est == 0) | (B_est == 1)).all():
            raise ValueError('B_est should take value in {0,1}')
        if not is_dag(B_est):
            raise ValueError('B_est should be a DAG')
    d = B_true.shape[0]
    # linear index of nonzeros
    pred_und = np.flatnonzero(B_est == -1)
    pred = np.flatnonzero(B_est == 1)
    cond = np.flatnonzero(B_true)
    cond_reversed = np.flatnonzero(B_true.T)
    cond_skeleton = np.concatenate([cond, cond_reversed])
    # true pos
    true_pos = np.intersect1d(pred, cond, assume_unique=True)
    # treat undirected edge favorably
    true_pos_und = np.intersect1d(pred_und, cond_skeleton, assume_unique=True)
    true_pos = np.concatenate([true_pos, true_pos_und])
    # false pos
    false_pos = np.setdiff1d(pred, cond_skeleton, assume_unique=True)
    false_pos_und = np.setdiff1d(pred_und, cond_skeleton, assume_unique=True)
    false_pos = np.concatenate([false_pos, false_pos_und])
    # reverse
    extra = np.setdiff1d(pred, cond, assume_unique=True)
    reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)
    # compute ratio
    pred_size = len(pred) + len(pred_und)
    cond_neg_size = 0.5 * d * (d - 1) - len(cond)
    fdr = float(len(reverse) + len(false_pos)) / max(pred_size, 1)
    tpr = float(len(true_pos)) / max(len(cond), 1)
    fpr = float(len(reverse) + len(false_pos)) / max(cond_neg_size, 1)
    # structural hamming distance
    pred_lower = np.flatnonzero(np.tril(B_est + B_est.T))
    cond_lower = np.flatnonzero(np.tril(B_true + B_true.T))
    extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
    missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
    shd = len(extra_lower) + len(missing_lower) + len(reverse)
    acc_list = [round(fdr, 3), round(tpr, 3), round(fpr, 3), shd, pred_size]
    # return {'fdr': format(fdr, '.3f'), 'tpr': format(tpr, '.3f'), 'fpr': format(fpr, '.3f'),
    #         'shd': format(shd, '>4d'), 'nnz': format(pred_size, '>4d')}
    return acc_list


def learn_struc(data, nodes):
    scoring_method = BDeuScore(data)
    est = HillClimbSearch(data)
    bn = est.estimate(scoring_method=scoring_method)
    score = scoring_method.score(bn)
    # nodes = ['DrivingSkill', 'PropCost', 'HomeBase', 'RiskAversion', 'ILiCost',
    #          'DrivQuality', 'SeniorTrain', 'CarValue', 'DrivHist', 'Accident',
    #          'ThisCarCost', 'ThisCarDam', 'RuggedAuto', 'OtherCarCost',
    #          'VehicleYear', 'Airbag', 'OtherCar', 'MakeModel', 'GoodStudent',
    #          'Mileage', 'AntiTheft', 'Cushioning', 'MedCost', 'Theft', 'Age',
    #          'SocioEcon', 'Antilock']
    w_g = np.zeros((len(data.columns), len(data.columns)))
    for (i, j) in bn.edges():
        w_g[list(nodes).index(i), list(nodes).index(j)] = 1

    G_True = np.load('./bn_generation/hailfinder/DAG.npy')
    acc = count_accuracy(G_True, w_g)
    return acc, score


bn = gum.loadBN("./bn_generation/hailfinder/hailfinder.bif")
gum.generateCSV(bn, "./bn_generation/hailfinder/data.csv", 200, True)
# 生成数据
data = pd.read_csv("./bn_generation/hailfinder/data.csv")
ctgan = CTGAN(batch_size=10, epochs=200, verbose=False)
ctgan.fit(data)
samples = ctgan.sample(200)

samples.to_csv("./bn_generation/hailfinder/data_generate.csv", index=0)

# 获取TRUE_DAG
nodes = data.columns
bifmodel = readwrite.BIF.BIFReader(path="./bn_generation/hailfinder/hailfinder.bif")
model = BayesianNetwork(bifmodel.variable_edges)
model.name = bifmodel.network_name
model.add_nodes_from(bifmodel.variable_names)
a = len(nodes)
w_g = np.zeros((a, a))
for (i, j) in model.edges():
    w_g[list(nodes).index(i), list(nodes).index(j)] = 1
np.save('./bn_generation/hailfinder/DAG.npy', w_g)

# 数据混合
df1 = pd.read_csv('./bn_generation/hailfinder/data.csv')
df1.to_csv('./bn_generation/hailfinder/data_mix.csv', index=0)
df = pd.read_csv(r'./bn_generation/hailfinder/data_generate.csv')  # 文件读取
c = random.sample(range(1, 200), 100)
df.iloc[c]
with open('./bn_generation/hailfinder/data_mix.csv', 'a+', newline='') as f:
    csv_write = csv.writer(f)
    for i in c:
        csv_write.writerow(df.iloc[i])

# 贝叶斯结构学习
data = pd.read_csv('./bn_generation/hailfinder/data.csv')
data_mix = pd.read_csv('./bn_generation/hailfinder/data_mix.csv')
res, re_score = learn_struc(data, nodes)
print("SHD:", res[3])
print("AUC:", (1 + res[1] - res[2]) / 2)
res_mix, re_mix_score = learn_struc(data_mix, nodes)
print("SHD+:", res_mix[3])
print("AUC+:", (1 + res_mix[1] - res_mix[2]) / 2)
