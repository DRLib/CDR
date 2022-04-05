import os

import numpy as np
from pynndescent import NNDescent
from sklearn.metrics import pairwise_distances

from utils.logger import InfoLogger


def cal_snn_similarity(knn, cache_path=None):
    if cache_path is not None and os.path.exists(cache_path):
        _, snn_sim = np.load(cache_path)
        InfoLogger.info("directly load accurate neighbor_graph from {}".format(cache_path))
        return knn, snn_sim

    snn_sim = np.zeros_like(knn)
    n_samples, n_neighbors = knn.shape
    for i in range(n_samples):
        sample_nn = knn[i]
        for j, neighbor_idx in enumerate(sample_nn):
            neighbor_nn = knn[int(neighbor_idx)]
            snn_num = len(np.intersect1d(sample_nn, neighbor_nn))
            snn_sim[i][j] = snn_num / n_neighbors
    if cache_path is not None and not os.path.exists(cache_path):
        np.save(cache_path, [knn, snn_sim])
        InfoLogger.info("successfully compute snn similarity and save to {}".format(cache_path))
    return knn, snn_sim


def compute_accurate_knn(flattened_data, k, neighbors_cache_path=None, pairwise_cache_path=None, metric="euclidean"):
    cur_path = None
    if neighbors_cache_path is not None:
        cur_path = neighbors_cache_path.replace(".npy", "_ac.npy")

    if cur_path is not None and os.path.exists(cur_path):
        knn_indices, knn_distances = np.load(cur_path)
        InfoLogger.info("directly load accurate neighbor_graph from {}".format(cur_path))
    else:
        preload = flattened_data.shape[0] <= 30000

        pairwise_distance = get_pairwise_distance(flattened_data, metric, pairwise_cache_path, preload=preload)
        sorted_indices = np.argsort(pairwise_distance, axis=1)
        knn_indices = sorted_indices[:, 1:k+1]
        knn_distances = []
        for i in range(knn_indices.shape[0]):
            knn_distances.append(pairwise_distance[i, knn_indices[i]])
        knn_distances = np.array(knn_distances)
        if cur_path is not None:
            np.save(cur_path, [knn_indices, knn_distances])
            InfoLogger.info("successfully compute accurate neighbor_graph and save to {}".format(cur_path))
    return knn_indices, knn_distances


def compute_knn_graph(all_data, neighbors_cache_path, k, pairwise_cache_path,
                      metric="euclidean", max_candidates=60, accelerate=False):
    flattened_data = all_data.reshape((len(all_data), np.product(all_data.shape[1:])))

    if not accelerate:
        knn_indices, knn_distances = compute_accurate_knn(flattened_data, k, neighbors_cache_path, pairwise_cache_path)
        return knn_indices, knn_distances

    if neighbors_cache_path is not None and os.path.exists(neighbors_cache_path):
        neighbor_graph = np.load(neighbors_cache_path)
        knn_indices, knn_distances = neighbor_graph
        InfoLogger.info("directly load approximate neighbor_graph from {}".format(neighbors_cache_path))
    else:
        # number of trees in random projection forest
        n_trees = 5 + int(round((all_data.shape[0]) ** 0.5 / 20.0))
        # max number of nearest neighbor iters to perform
        n_iters = max(5, int(round(np.log2(all_data.shape[0]))))
        nnd = NNDescent(
            flattened_data,
            n_neighbors=k+1,
            metric=metric,
            n_trees=n_trees,
            n_iters=n_iters,
            max_candidates=max_candidates,
            verbose=False
        )

        knn_indices, knn_distances = nnd.neighbor_graph
        knn_indices = knn_indices[:, 1:]
        knn_distances = knn_distances[:, 1:]

        if neighbors_cache_path is not None:
            np.save(neighbors_cache_path, [knn_indices, knn_distances])
        InfoLogger.info("successfully compute approximate neighbor_graph and save to {}".format(neighbors_cache_path))
    return knn_indices, knn_distances


def get_pairwise_distance(flattened_data, metric, pairwise_distance_cache_path=None, preload=False):
    if pairwise_distance_cache_path is not None and preload and os.path.exists(pairwise_distance_cache_path):
        pairwise_distance = np.load(pairwise_distance_cache_path)
        InfoLogger.info("directly load pairwise distance from {}".format(pairwise_distance_cache_path))
    else:
        pairwise_distance = pairwise_distances(flattened_data, metric=metric, squared=False)
        pairwise_distance[pairwise_distance < 1e-12] = 0.0
        if preload and pairwise_distance_cache_path is not None:
            np.save(pairwise_distance_cache_path, pairwise_distance)
            InfoLogger.info("successfully compute pairwise distance and save to {}".format(pairwise_distance_cache_path))
    return pairwise_distance
