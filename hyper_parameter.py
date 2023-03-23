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
    w_g = np.zeros((len(data.columns), len(data.columns)))
    for (i, j) in bn.edges():
        w_g[list(nodes).index(i), list(nodes).index(j)] = 1

    G_True = np.load('./bn_generation/hailfinder/DAG.npy')
    acc = count_accuracy(G_True, w_g)
    return acc, score


# 数据混合
df1 = pd.read_csv('./bn_generation/hailfinder/data.csv')
AUC = []
for i in range(10):
    df1.to_csv('./bn_generation/hailfinder/data_mix_{}.csv'.format(i+1), index=0)
    df = pd.read_csv(r'./bn_generation/hailfinder/data_generate.csv')  # 文件读取
    c = random.sample(range(200), 20*(i+1))
    df.iloc[c]
    with open('./bn_generation/hailfinder/data_mix_{}.csv'.format(i+1), 'a+', newline='') as f:
        csv_write = csv.writer(f)
        for a in c:
            csv_write.writerow(df.iloc[a])

    nodes = df1.columns
    file_path = r"./bn_generation/hailfinder/data_mix_{}.csv".format(i+1)
    data = pd.read_csv(file_path)
    res_mix, re_mix_score = learn_struc(data, nodes)
    AUC.append((1 + res_mix[1] - res_mix[2]) / 2)
print("hailfinder_AUC:", AUC)
