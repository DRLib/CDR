#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import os
import random

import h5py
import numpy as np
import torch
from PIL import Image
from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA
from torch.utils.data import Dataset
from torchvision import transforms as transforms

from utils.constant_pool import ConfigInfo
from utils.logger import InfoLogger
from utils.nn_utils import compute_knn_graph, compute_accurate_knn, cal_snn_similarity
from utils.umap_utils import fuzzy_simplicial_set, construct_edge_dataset, compute_local_membership

MACHINE_EPSILON = np.finfo(np.double).eps


class MyTextDataset(Dataset):
    def __init__(self, dataset_name, root_dir):
        self.dataset_name = dataset_name
        self.root_dir = root_dir
        self.data_file_path = os.path.join(root_dir, dataset_name + ".h5")
        self.data = None
        self.target = None
        self.data_num = 0
        self.min_neighbor_num = 0
        self.symmetry_knn_indices = None
        self.symmetry_knn_weights = None
        self.symmetry_knn_dists = None
        self.transform = None
        self.__load_data()

    def __len__(self):
        return self.data.shape[0]

    def __load_data(self):
        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        train_data, train_labels = \
            load_local_h5_by_path(self.data_file_path, ['x', 'y'])
        self.data = train_data
        self.targets = train_labels
        self.data_num = self.data.shape[0]

    def __getitem__(self, index):
        text, target = self.data[index], int(self.targets[index])
        text = torch.tensor(text, dtype=torch.float)

        return text, target

    def _check_exists(self):
        return os.path.exists(self.data_file_path)

    def update_transform(self, new_transform):
        self.transform = new_transform

    def get_data(self, index):
        res = self.data[index]
        return torch.tensor(res, dtype=torch.float)

    def get_label(self, index):
        return int(self.targets[index])

    def get_dims(self):
        return int(self.data.shape[1])

    def get_all_data(self, data_num=-1):
        if data_num == -1:
            return self.data
        else:
            return self.data[torch.randperm(self.data_num)[:data_num], :]

    def get_data_shape(self):
        return self.data[0].shape


class MyImageDataset(MyTextDataset):
    def __init__(self, dataset_name, root_dir, transform=None):
        MyTextDataset.__init__(self, dataset_name, root_dir)
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        img = np.squeeze(img)
        mode = 'RGB' if len(img.shape) == 3 else 'L'
        if mode == 'RGB':
            img = Image.fromarray(img, mode=mode)
        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def get_data(self, index):
        res = self.data[index]
        res = res.astype(np.uint8)
        return res

    def get_all_data(self, data_num=-1):
        if data_num == -1:
            return np.transpose(self.data, (0, 3, 1, 2))
        else:
            return np.transpose(self.data[torch.randperm(self.data_num)[:data_num], :, :, :], (0, 3, 1, 2))


class UMAPTextDataset(MyTextDataset):
    def __init__(self, dataset_name, root_dir, repeat=1):
        MyTextDataset.__init__(self, dataset_name, root_dir)
        self.repeat = repeat
        self.edge_data = None
        self.edge_num = None
        self.edge_weight = None
        self.raw_knn_weights = None

    def build_fuzzy_simplicial_set(self, knn_cache_path, pairwise_cache_path, n_neighbors):

        knn_indices, knn_distances = compute_knn_graph(self.data, knn_cache_path, n_neighbors, pairwise_cache_path)
        umap_graph, sigmas, rhos, self.raw_knn_weights = fuzzy_simplicial_set(
            X=self.data,
            n_neighbors=n_neighbors,
            knn_indices=knn_indices,
            knn_dists=knn_distances)
        return umap_graph, sigmas, rhos

    def umap_process(self, knn_cache_path, pairwise_cache_path, n_neighbors, embedding_epoch):
        umap_graph, sigmas, rhos = self.build_fuzzy_simplicial_set(knn_cache_path, pairwise_cache_path, n_neighbors)
        self.edge_data, self.edge_num, self.edge_weight = construct_edge_dataset(
            self.data, umap_graph, embedding_epoch)

        return self.edge_data, self.edge_num

    def __getitem__(self, index):
        to_data, from_data = self.edge_data[0][index], self.edge_data[1][index]
        return torch.tensor(to_data, dtype=torch.float), torch.tensor(from_data, dtype=torch.float)

    def __len__(self):
        return self.edge_num


class UMAPImageDataset(MyImageDataset, UMAPTextDataset):
    def __init__(self, dataset_name, root_dir, transform=None, repeat=1):
        MyImageDataset.__init__(self, dataset_name, root_dir, transform)
        UMAPTextDataset.__init__(self, dataset_name, root_dir, repeat)
        self.transform = transform

    def __getitem__(self, index):
        to_data, from_data = self.edge_data[0][index], self.edge_data[1][index]
        if self.transform is not None:
            to_data = self.transform(to_data)
            from_data = self.transform(from_data)

        return to_data, from_data


class CDRTextDataset(MyTextDataset):
    def __init__(self, dataset_name, root_dir):
        MyTextDataset.__init__(self, dataset_name, root_dir)

    def __getitem__(self, index):
        text, target = self.data[index], int(self.targets[index])
        x, x_sim, idx, sim_idx = self.transform(text, index)
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float)
            x_sim = torch.tensor(x_sim, dtype=torch.float)
        return [x, x_sim, idx, sim_idx], target

    def sample_data(self, indices):
        x = self.data[indices]
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float)
        return x


class CDRImageDataset(MyImageDataset):
    def __init__(self, dataset_name, root_dir, transform=None):
        MyImageDataset.__init__(self, dataset_name, root_dir, transform)
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        img = np.squeeze(img)
        mode = 'RGB' if len(img.shape) == 3 else 'L'
        img = Image.fromarray(img, mode=mode)
        if self.transform is not None:
            img = self.transform(img, index)

        return img, target

    def sample_data(self, indices):
        num = len(indices)
        first_data = self.data[indices[0]]
        ret_data = torch.empty((num, first_data.shape[2], first_data.shape[0], first_data.shape[1]))
        count = 0
        transform = transforms.ToTensor()
        for index in indices:
            img = np.squeeze(self.data[index])
            mode = 'RGB' if len(img.shape) == 3 else 'L'
            img = Image.fromarray(img, mode=mode)
            img = transform(img)
            ret_data[count, :, :, :] = img.unsqueeze(0)
            count += 1
        return ret_data


class UMAPCDRTextDataset(CDRTextDataset):
    def __init__(self, dataset_name, root_dir):
        CDRTextDataset.__init__(self, dataset_name, root_dir)
        self.umap_graph = None
        self.raw_knn_weights = None
        self.sym_no_norm_weights = None
        self.min_neighbor_num = None
        self.knn_dist = None
        self.knn_indices = None

    def build_fuzzy_simplicial_set(self, knn_indices, knn_distances, n_neighbors, symmetric):
        self.umap_graph, sigmas, rhos, self.raw_knn_weights, knn_dist = fuzzy_simplicial_set(
            X=self.data,
            n_neighbors=n_neighbors, knn_indices=knn_indices,
            knn_dists=knn_distances, return_dists=True, symmetric=symmetric)
        self.symmetry_knn_dists = knn_dist.tocoo()

    def umap_process(self, knn_indices, knn_distances, n_neighbors, symmetric):
        self.build_fuzzy_simplicial_set(knn_indices, knn_distances, n_neighbors, symmetric)

        self.data_num = knn_indices.shape[0]
        n_samples = self.data_num

        nn_indices, nn_weights, self.min_neighbor_num, raw_weights, nn_dists \
            = get_kw_from_coo(self.umap_graph, n_neighbors, n_samples, self.symmetry_knn_dists)

        self.symmetry_knn_indices = np.array(nn_indices, dtype=object)
        self.symmetry_knn_weights = np.array(nn_weights, dtype=object)
        self.symmetry_knn_dists = np.array(nn_dists, dtype=object)
        self.sym_no_norm_weights = np.array(raw_weights, dtype=object)


class UMAPCDRImageDataset(CDRImageDataset, UMAPCDRTextDataset):
    def __init__(self, dataset_name, root_dir, transform=None):
        CDRImageDataset.__init__(self, dataset_name, root_dir, transform)
        UMAPCDRTextDataset.__init__(self, dataset_name, root_dir)


def get_kw_from_coo(csr_graph, n_neighbors, n_samples, dist_csr=None):
    nn_indices = []
    nn_weights = []
    raw_weights = []
    nn_dists = []

    tmp_min_neighbor_num = n_neighbors
    for i in range(1, n_samples + 1):
        pre = csr_graph.indptr[i-1]
        idx = csr_graph.indptr[i]
        cur_indices = csr_graph.indices[pre:idx]
        if dist_csr is not None:
            nn_dists.append(dist_csr.data[pre:idx])
        tmp_min_neighbor_num = min(tmp_min_neighbor_num, idx - pre)
        cur_weights = csr_graph.data[pre:idx]

        nn_indices.append(cur_indices)
        cur_sum = np.sum(cur_weights)
        nn_weights.append(cur_weights / cur_sum)
        raw_weights.append(cur_weights)
    return nn_indices, nn_weights, tmp_min_neighbor_num, raw_weights, nn_dists


def load_local_h5_by_path(dataset_path, keys):
    f = h5py.File(dataset_path, "r")
    res = []
    for key in keys:
        res.append(f[key][:])
    f.close()
    return res
