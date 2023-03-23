from pgmpy.estimators import BDeuScore
from pgmpy.estimators import PC, HillClimbSearch, ExhaustiveSearch
from pgmpy.models import BayesianModel
import igraph as ig
import argparse
import logging
import torch
# import warnings
import random
import numpy as np
import pandas as pd
import os


def is_dag(W):
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    return G.is_dag()


def count_accuracy(B_true, B_est):
    """Compute various accuracy metrics for B_est.

    true positive = predicted association exists in condition in correct direction
    reverse = predicted association exists in condition in opposite direction
    false positive = predicted association does not exist in condition

    Args:
        B_true (np.ndarray): [d, d] ground truth graph, {0, 1}
        B_est (np.ndarray): [d, d] estimate, {0, 1, -1}, -1 is undirected edge in CPDAG

    Returns:
        fdr: (reverse + false positive) / prediction positive
        tpr: (true positive) / condition positive
        fpr: (reverse + false positive) / condition negative
        shd: undirected extra + undirected missing + reverse
        nnz: prediction positive
    """
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
    return {'fdr': format(fdr, '.3f'), 'tpr': format(tpr, '.3f'), 'fpr': format(fpr, '.3f'),
            'shd': format(shd, '>4d'), 'nnz': format(pred_size, '>4d')}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='./insurance')
    parser.add_argument('--nodes_num', type=int, default='8')
    parser.add_argument('--size', type=int, default='1024')
    parser.add_argument('--score', type=str, default='BDeu')
    parser.add_argument('--loglevel', type=str, default='DEBUG')

    opt = parser.parse_args()

    dataset = opt.data
    seq_len = opt.nodes_num
    size = opt.size
    score = opt.score
    loglevel = opt.loglevel

    formatter = logging.Formatter('%(asctime)s -- %(levelname)s: %(message)s')
    logger = logging.getLogger('SR-DAG_LOGGER')
    logger.setLevel(logging.DEBUG)

    filelog = logging.FileHandler(dataset + '.log', 'a')
    if loglevel == 'DEBUG':
        filelevel = logging.DEBUG
    elif loglevel == 'INFO':
        filelevel = logging.INFO
    else:
        filelevel = logging.ERROR
    filelog.setLevel(filelevel)
    filelog.setFormatter(formatter)
    logger.addHandler(filelog)

    data = pd.read_csv(dataset + '/data11.csv')
    # data = pd.read_csv(dataset + '/data_gen.csv')
    # data = pd.read_csv(dataset + '/data_mix.csv')
    scoring_method = BDeuScore(data)
    est = HillClimbSearch(data)
    bn = est.estimate(scoring_method=scoring_method)

    # nodes = data.columns
    nodes = ['DrivingSkill', 'PropCost', 'HomeBase', 'RiskAversion', 'ILiCost',
             'DrivQuality', 'SeniorTrain', 'CarValue', 'DrivHist', 'Accident',
             'ThisCarCost', 'ThisCarDam', 'RuggedAuto', 'OtherCarCost',
             'VehicleYear', 'Airbag', 'OtherCar', 'MakeModel', 'GoodStudent',
             'Mileage', 'AntiTheft', 'Cushioning', 'MedCost', 'Theft', 'Age',
             'SocioEcon', 'Antilock']
    # nodes = ['either', 'xray', 'bronc', 'asia', 'dysp', 'smoke', 'tub', 'lung']

    w_g = np.zeros((len(data.columns), len(data.columns)))
    for (i, j) in bn.edges():
        w_g[list(nodes).index(i), list(nodes).index(j)] = 1

    G_True = np.load(dataset + '/DAG.npy')
    acc = count_accuracy(G_True, w_g)
    print(acc)
    logger.info(acc)

    logger.info('done')
    # Note = open('asia_log.txt', mode='a')
    # Note.write("{}\n".format(acc))
