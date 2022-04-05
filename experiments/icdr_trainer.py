#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import random
import torch
from experiments.trainer import CDRTrainer
from utils.constant_pool import *
from utils.link_utils import CANNOT_LINK, MUST_LINK, LinkInfo, UN_SPREAD


class ICLPTrainer(CDRTrainer):
    def __init__(self, clr_model, dataset_name, configs, result_save_dir, config_path, device='cuda',
                 log_path="log.txt"):
        CDRTrainer.__init__(self, clr_model, dataset_name, configs, result_save_dir, config_path, device, log_path)
        self.link_info = None
        self.dataset = None
        self.finetune_epoch = 0
        self.finetune_data_ratio = 0.5
        self.minimum_finetune_data_num = 400
        self.finetune_data_num = 0
        self.steady_epoch = 2
        self.is_finetune = False
        self.has_link = False

    def get_label(self):
        return self.train_loader.dataset.targets.tolist()

    def update_link_stat(self, link_info, is_finetune, finetune_epoch):
        self.message_queue = Queue()
        self.link_info = link_info
        self.has_link = link_info is not None
        self.finetune_epoch = finetune_epoch
        self.is_finetune = is_finetune

        self._update_finetune_dataset()

        self._update_symm_knn_by_cl()

    def _step_prepare(self, *args):
        if self.dataset is None:
            self.dataset = self.train_loader.dataset
        return super()._step_prepare(*args)

    def _train_step(self, *args):
        x, x_sim, epoch, indices, sim_indices = args
        self.optimizer.zero_grad()

        resp_data = None
        link_embeddings = None
        final_link_types = []
        final_x_ranks = []
        final_link_weights = []
        if self.has_link:
            all_indices = np.concatenate([indices, sim_indices])
            all_link_indices = self.link_info.crs_indices[:, [1, 2]]
            all_link_types = self.link_info.crs_indices[:, 3]
            all_link_weights = self.link_info.crs_indices[:, 4]
            flattened_all_link_indices = np.ravel(all_link_indices)

            link_related_indices, x_ranks, y_ranks = np.intersect1d(all_indices, flattened_all_link_indices,
                                                                    return_indices=True)
            if len(link_related_indices) > 0:
                per_link_indices = (y_ranks // 2).astype(np.int)
                resp_indices = all_link_indices[per_link_indices, ((y_ranks + 1) % 2).astype(np.int)]
                resp_link_types = all_link_types[per_link_indices]
                resp_link_weights = all_link_weights[per_link_indices]
                final_indices = []

                for i in range(len(resp_indices)):
                    if resp_indices[i] not in link_related_indices:
                        final_indices.append(resp_indices[i])
                        final_link_types.append(resp_link_types[i])
                        final_x_ranks.append(x_ranks[i])
                        final_link_weights.append(resp_link_weights[i])

                final_indices = np.array(final_indices, dtype=np.int)
                final_link_types = np.array(final_link_types, dtype=np.int)
                final_x_ranks = np.array(final_x_ranks, dtype=np.int)
                final_link_weights = np.array(final_link_weights, dtype=np.float)

                if len(final_indices) > 0:
                    resp_data = self.dataset.data[final_indices]
                    resp_data = torch.tensor(resp_data, dtype=torch.float).to(self.device, non_blocking=True)
                    if self.is_image:
                        resp_data /= 255.

        if resp_data is not None:
            x_and_resp = torch.cat([x, resp_data], dim=0)
            resp_num = resp_data.shape[0]
            x_and_resp_embeddings = self.encode(x_and_resp)[1]

            link_embeddings = x_and_resp_embeddings[-resp_num:]
            x_embeddings = x_and_resp_embeddings[:-resp_num]
            x_sim_embeddings = self.encode(x_sim)[1]
        else:
            x_embeddings = self.encode(x)[1]
            x_sim_embeddings = self.encode(x_sim)[1]

        train_loss = self.model.compute_loss(x_embeddings, x_sim_embeddings, epoch, link_embeddings, final_link_types,
                                             final_x_ranks, final_link_weights)

        train_loss.backward()
        self.optimizer.step()
        return train_loss

    def _update_symm_knn_by_cl(self):

        def delete_neighbor(self_idx, other_indices):
            inter_knn_indices = np.setdiff1d(symm_knn_indices[self_idx], other_indices)
            if len(inter_knn_indices) != len(symm_knn_indices[self_idx]):
                indices = []
                for item in inter_knn_indices:
                    indices.append(np.argwhere(symm_knn_indices[self_idx] == item)[0][0])
                symm_knn_indices[self_idx] = inter_knn_indices
                symm_no_norm_weights[self_idx] = symm_no_norm_weights[self_idx][indices]

        link_weight = 1
        symm_knn_indices = self.dataset.symmetry_knn_indices
        symm_knn_weights = self.dataset.symmetry_knn_weights
        symm_no_norm_weights = self.dataset.sym_no_norm_weights

        link_num = self.link_info.new_link_num if self.link_info is not None else 0
        if link_num == 0:
            self.dataset.transform.build_neighbor_repo(self.finetune_epoch, self.n_neighbors, symm_knn_indices,
                                                       symm_knn_weights)
            return

        link_crs = self.link_info.new_crs_indices
        link_sims = self.link_info.new_crs_sims
        repeat_num = self.link_info.new_link_spreads + ~(self.link_info.new_link_spreads.astype(np.bool))

        link_spreads = np.repeat(self.link_info.new_link_spreads, repeat_num)

        total_link_num = link_crs.shape[0]

        for i in range(total_link_num):
            uuid, h_idx, t_idx, link_type, _ = link_crs[i]
            h_sim, t_sim = link_sims[i][[1, 2]]
            h_idx = int(h_idx)
            t_idx = int(t_idx)

            if link_type == MUST_LINK:
                if link_spreads[i] == UN_SPREAD:
                    symm_knn_indices[h_idx] = np.array([t_idx], dtype=np.int)
                    symm_knn_indices[t_idx] = np.array([h_idx], dtype=np.int)
                    symm_no_norm_weights[h_idx] = np.array([1], dtype=np.float)
                    symm_no_norm_weights[t_idx] = np.array([1], dtype=np.float)
                else:
                    symm_knn_indices[h_idx] = np.append(symm_knn_indices[h_idx], t_idx)
                    symm_knn_indices[t_idx] = np.append(symm_knn_indices[t_idx], h_idx)

                    h_weight, t_weight = h_sim, t_sim
                    w = h_weight * t_weight * link_weight

                    symm_no_norm_weights[h_idx] = np.append(symm_no_norm_weights[h_idx], w)
                    symm_no_norm_weights[t_idx] = np.append(symm_no_norm_weights[t_idx], w)
            else:
                delete_neighbor(h_idx, t_idx)
                delete_neighbor(t_idx, h_idx)

            symm_knn_weights[h_idx] = symm_no_norm_weights[h_idx] / np.sum(symm_no_norm_weights[h_idx])
            symm_knn_weights[t_idx] = symm_no_norm_weights[t_idx] / np.sum(symm_no_norm_weights[t_idx])

        self.dataset.transform.build_neighbor_repo(self.finetune_epoch, self.n_neighbors, symm_knn_indices,
                                                   symm_knn_weights)

    def _after_epoch(self, ckp_save_inter, epoch, training_loss, training_loss_history, val_inter):
        ret_val = super()._after_epoch(ckp_save_inter, epoch, training_loss, training_loss_history, val_inter)
        if epoch % 10 == 0 and self.has_link:
            self._update_finetune_dataset()
            pass
        return ret_val

    def _update_finetune_dataset(self):
        sampled_num = max(int(self.n_samples * self.finetune_data_ratio), self.minimum_finetune_data_num)
        sampled_indices = random.sample(list(np.arange(0, self.n_samples, 1)), sampled_num)
        if self.link_info is not None:
            sampled_indices = np.union1d(sampled_indices,
                                         np.ravel(self.link_info.crs_indices[:, [1, 2]]).astype(np.int))
        self.train_loader.sampler.update_indices(sampled_indices, True)
        self.finetune_data_num = len(sampled_indices)
