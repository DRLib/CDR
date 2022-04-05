#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import torch
import numpy as np
import torch.nn.functional as F
from pynndescent import NNDescent

from utils.umap_utils import find_ab_params, convert_distance_to_probability
import networkx as nx
import bisect
from scipy.spatial.distance import pdist, squareform

MACHINE_EPSILON = np.finfo(np.double).eps


def _student_t_similarity(rep1, rep2, *args):
    pairwise_matrix = torch.norm(rep1 - rep2, dim=-1)
    similarity_matrix = 1 / (1 + pairwise_matrix ** 2)
    return similarity_matrix, pairwise_matrix


def _exp_similarity(rep1, rep2, *args):
    pairwise_matrix = torch.norm(rep1 - rep2, dim=-1)
    similarity_matrix = torch.exp(-pairwise_matrix ** 2)
    return similarity_matrix, pairwise_matrix


def _cosine_similarity(rep1, rep2, *args):
    x = rep2[0]
    x = F.normalize(x, dim=1)
    similarity_matrix = torch.matmul(x, x.T).clamp(min=1e-7)
    pairwise_matrix = torch.norm(rep1 - rep2, dim=-1)
    return similarity_matrix, pairwise_matrix


a = None
b = None
pre_min_dist = -1


def _umap_similarity(rep1, rep2, min_dist=0.1):
    pairwise_matrix = torch.norm(rep1 - rep2, dim=-1)
    global a, b, pre_min_dist
    if a is None or pre_min_dist != min_dist:
        pre_min_dist = min_dist
        a, b = find_ab_params(1.0, min_dist)

    similarity_matrix = convert_distance_to_probability(pairwise_matrix, a, b)
    return similarity_matrix, pairwise_matrix


def get_similarity_function(similarity_method):
    if similarity_method == "umap":
        return _umap_similarity
    elif similarity_method == "tsne":
        return _student_t_similarity
    elif similarity_method == "exp":
        return _exp_similarity
    elif similarity_method == "cosine":
        return _cosine_similarity


def get_correlated_mask(batch_size):
    diag = np.eye(batch_size)
    l1 = np.eye(batch_size, batch_size, k=int(-batch_size / 2))
    l2 = np.eye(batch_size, batch_size, k=int(batch_size / 2))
    mask = torch.from_numpy((diag + l1 + l2))
    mask = (1 - mask).type(torch.bool)
    return mask
