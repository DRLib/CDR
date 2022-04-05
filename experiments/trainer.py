#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import math

import torch

from dataset.warppers import DataSetWrapper
from utils.common_utils import check_path_exists, time_stamp_to_date_time_adjoin
from utils.math_utils import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from torch.optim.lr_scheduler import MultiStepLR
import shutil
from utils.constant_pool import ConfigInfo
from multiprocessing import Queue
from utils.logger import InfoLogger, LogWriter
import seaborn as sns
import time


def draw_loss(training_loss, idx, save_path=None):
    plt.figure()
    plt.plot(idx, training_loss, color="blue", label="training loss")
    plt.legend()
    plt.xlabel("epochs")
    plt.ylabel("loss")
    if save_path is not None:
        plt.savefig(save_path)
    # plt.show()


def draw_projections(embeddings, labels, vis_save_path):
    x = embeddings[:, 0]
    y = embeddings[:, 1]

    plt.figure(figsize=(8, 8))
    if labels is None:
        sns.scatterplot(x=x, y=y, s=8, legend=False, alpha=1.0)
    else:
        classes = np.unique(labels)
        num_classes = classes.shape[0]
        palette = "tab10" if num_classes <= 10 else "tab20"
        sns.scatterplot(x=x, y=y, hue=labels, s=8, palette=palette, legend=False, alpha=0.8)
    plt.xticks([])
    plt.yticks([])

    if vis_save_path is not None:
        plt.savefig(vis_save_path, dpi=600, bbox_inches='tight', pad_inches=0.1)
    # plt.show()


class CDRTrainer:
    def __init__(self, model, dataset_name, configs, result_save_dir, config_path, device='cuda',
                 log_path="log.txt"):
        self.model = model
        self.config_path = config_path
        self.configs = configs
        self.device = device
        self.result_save_dir = result_save_dir
        self.dataset_name = dataset_name
        self.batch_size = configs.exp_params.batch_size
        self.epoch_num = configs.training_params.epoch_nums
        self.n_neighbors = configs.exp_params.n_neighbors
        self.print_iter = int(self.configs.training_params.epoch_print_inter_ratio * self.epoch_num)
        self.is_image = not isinstance(self.configs.exp_params.input_dims, int)
        self.lr = configs.exp_params.LR
        self.ckp_save_dir = self.result_save_dir

        self.batch_num = 0
        self.val_inter = 0
        self.start_epoch = 0
        self.train_loader = None
        self.launch_date_time = None
        self.optimizer = None
        self.scheduler = None

        self.tmp_log_path = log_path
        self.log_process = None
        self.log_path = None
        self.message_queue = Queue()
        self.pre_embeddings = None
        self.fixed_k = 15

        self.clr_dataset = None
        self.resume_epochs = 0
        self.model.to(self.device)
        self.steps = 0
        self.resume_start_epoch = self.resume_epochs if self.resume_epochs > 0 else self.epoch_num
        self.gradient_redefine = configs.exp_params.gradient_redefine
        self.warmup_epochs = 0
        self.separation_epochs = 0
        if self.gradient_redefine:
            self.warmup_epochs = int(self.epoch_num * configs.exp_params.separation_begin_ratio)
            self.separation_epochs = int(self.epoch_num * configs.exp_params.steady_begin_ratio)

    def update_configs(self, configs):
        self.configs = configs
        self.dataset_name = configs.exp_params.dataset
        self.epoch_num = configs.training_params.epoch_nums

    def encode(self, x):
        return self.model.encode(x)

    def _train_begin(self, launch_time_stamp=None):
        self.sta_time = time.time() if launch_time_stamp is None else launch_time_stamp

        InfoLogger.info("Start Training for {} Epochs".format(self.epoch_num - self.start_epoch))

        param_template = "Experiment Configurations: \nDataset: %s Epochs: %d Batch Size: %d \n" \
                         "Learning rate: %4f Optimizer: %s\n"

        param_str = param_template % (self.dataset_name, self.epoch_num, self.batch_size,
            self.lr, self.configs.exp_params.optimizer)

        InfoLogger.info(param_str)
        self.message_queue.put(param_str)

        InfoLogger.info("Start Training for {} Epochs".format(self.epoch_num))
        if self.launch_date_time is None:
            if launch_time_stamp is None:
                launch_time_stamp = int(time.time())
            self.launch_date_time = time_stamp_to_date_time_adjoin(launch_time_stamp)

        self.result_save_dir = os.path.join(self.result_save_dir,
                                            "{}_{}".format(self.dataset_name, self.launch_date_time))

        self.log_path = os.path.join(self.result_save_dir, "log.txt")
        self.ckp_save_dir = self.result_save_dir

        if self.optimizer is None:
            self.init_optimizer()
            self.init_scheduler(cur_epochs=self.epoch_num)

        val_inter = math.ceil(self.epoch_num * self.configs.training_params.val_inter_ratio)
        ckp_save_inter = math.ceil(self.epoch_num * self.configs.training_params.ckp_inter_ratio)

        return val_inter, ckp_save_inter

    def init_optimizer(self):
        if self.configs.exp_params.optimizer == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=0.0001)
        elif self.configs.exp_params.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9,
                                             weight_decay=0.0001)
        else:
            raise RuntimeError("Unsupported optimizer! Please check the configuration and ensure the param "
                               "name is one of 'adam/sgd'")

    def init_scheduler(self, cur_epochs, base=0, gamma=0.1, milestones=None):
        if milestones is None:
            milestones = [0.8]
        if self.configs.exp_params.scheduler == "multi_step":
            self.scheduler = MultiStepLR(self.optimizer, milestones=[int(base + p * cur_epochs) for p in milestones],
                                         gamma=gamma, last_epoch=-1)
        elif self.configs.exp_params.scheduler == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=len(self.train_loader),
                                                                        eta_min=0.00001, last_epoch=-1)
        else:
            raise RuntimeError("Unsupported learning scheduler! Please check the configuration and ensure the param "
                               "name is one of 'multi_step/cosine'")

    def _before_epoch(self, epoch):
        self.model = self.model.to(self.device)
        if self.gradient_redefine:
            if epoch == self.warmup_epochs:
                self.train_loader.dataset.transform.build_neighbor_repo(self.separation_epochs - self.warmup_epochs,
                                                                        self.n_neighbors)
            elif epoch == self.separation_epochs:
                self.train_loader.dataset.transform.build_neighbor_repo(self.epoch_num - self.separation_epochs,
                                                                        self.n_neighbors)

        train_iterator = iter(self.train_loader)
        return train_iterator, 0

    def _step_prepare(self, *args):
        data, epoch = args
        x, x_sim, indices, sim_indices = data[0]

        x = x.to(self.device, non_blocking=True)
        x_sim = x_sim.to(self.device, non_blocking=True)
        return x, x_sim, epoch, indices, sim_indices

    def _train_step(self, *args):
        x, x_sim, epoch, indices, sim_indices = args

        self.optimizer.zero_grad()
        _, x_embeddings, _, x_sim_embeddings = self.forward(x, x_sim)

        train_loss = self.model.compute_loss(x_embeddings, x_sim_embeddings, epoch)

        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0, norm_type=2)
        self.optimizer.step()
        return train_loss

    def model_prepare(self):
        self.model.preprocess()

    def _after_epoch(self, ckp_save_inter, epoch, training_loss, training_loss_history, val_inter):

        if self.configs.exp_params.scheduler == "cosine" and epoch >= 10:
            self.scheduler.step()
        elif self.configs.exp_params.scheduler == "multi_step":
            self.scheduler.step()

        train_loss = training_loss / self.batch_num
        if epoch % self.print_iter == 0:
            epoch_template = 'Epoch %d/%d, Train Loss: %.5f, '
            epoch_output = epoch_template % (epoch, self.epoch_num, train_loss)
            InfoLogger.info(epoch_output)
            self.message_queue.put(epoch_output)

        training_loss_history.append(train_loss)
        embeddings = self.post_epoch(ckp_save_inter, epoch, val_inter)

        return embeddings

    def _train_end(self, training_loss_history, embeddings):
        np.save(os.path.join(self.result_save_dir, "embeddings_{}.npy".format(self.epoch_num)), embeddings)
        self.message_queue.put("end")
        self.save_weights(self.epoch_num)

        x_idx = np.linspace(self.start_epoch, self.epoch_num, self.epoch_num - self.start_epoch)
        save_path = os.path.join(self.result_save_dir,
                                 "loss_{}.jpg".format(self.epoch_num))
        draw_loss(training_loss_history, x_idx, save_path)
        self.log_process.join(timeout=5)
        shutil.copyfile(self.tmp_log_path, self.log_path)
        InfoLogger.info("Training process logging to {}".format(self.log_path))

    def train(self, launch_time_stamp=None):
        self.val_inter, ckp_save_inter = self._train_begin(launch_time_stamp)
        embeddings = None
        net = self.model
        net.batch_num = self.batch_num
        training_loss_history = []

        for epoch in range(self.start_epoch, self.epoch_num):
            train_iterator, training_loss = self._before_epoch(epoch)
            for idx, data in enumerate(train_iterator):
                self.steps += 1
                train_data = self._step_prepare(data, epoch)
                loss = self._train_step(*train_data)
                training_loss += loss

            embeddings = self._after_epoch(ckp_save_inter, epoch + 1, training_loss, training_loss_history,
                                           self.val_inter)

        self._train_end(training_loss_history, embeddings)
        return embeddings

    def resume_train(self, resume_epoch):
        self.resume_start_epoch = self.epoch_num
        self.start_epoch = self.epoch_num
        self.epoch_num = self.resume_start_epoch + resume_epoch
        self.optimizer.param_groups[0]['lr'] = self.lr
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=len(self.train_loader),
                                                                    eta_min=0.00001, last_epoch=-1)
        return self.train()

    def save_weights(self, epoch, prefix_name=None):
        if prefix_name is None:
            prefix_name = epoch
        if not os.path.exists(self.ckp_save_dir):
            os.mkdir(self.ckp_save_dir)
        weight_save_path = os.path.join(self.ckp_save_dir, "{}.pth.tar".
                                        format(prefix_name))
        torch.save({'epoch': epoch, 'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'lr': self.lr, 'launch_time': self.launch_date_time}, weight_save_path)
        InfoLogger.info("model weights successfully saved to {}".format(weight_save_path))

    def forward(self, x, x_sim):
        return self.model.forward(x, x_sim)

    def load_weights(self, checkpoint_path, train=True):
        self.preprocess(train)
        model_ckpt = torch.load(checkpoint_path, map_location=torch.device(self.device))
        self.model.load_state_dict(model_ckpt['state_dict'])
        self.init_optimizer()
        self.optimizer.load_state_dict(model_ckpt['optimizer'])
        self.optimizer.param_groups[0]['lr'] = self.lr
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(self.device)
        return model_ckpt

    def load_weights_train(self, checkpoint_path):
        model_ckpt = self.load_weights(checkpoint_path)
        self.start_epoch = model_ckpt["epoch"]
        self.launch_date_time = model_ckpt["launch_time"]
        self.train()

    def load_weights_visualization(self, checkpoint_path, vis_save_path, device='cuda'):
        self.load_weights(checkpoint_path, train=False)
        embeddings = self.visualize(vis_save_path, device=device)
        return embeddings

    def train_for_visualize(self):
        InfoLogger.info("Start train for Visualize")
        launch_time_stamp = int(time.time())
        self.preprocess()
        self.pre_embeddings = self.train(launch_time_stamp)
        return self.pre_embeddings

    def cal_lower_embeddings(self, data):
        if self.is_image:
            data = data / 255.
        embeddings = self.acquire_latent_code_allin(data)
        return embeddings

    def visualize(self, vis_save_path=None, device="cuda"):
        self.model.to(device)
        data = torch.tensor(self.train_loader.dataset.get_all_data()).to(device).float()
        embeddings = self.cal_lower_embeddings(data)

        draw_projections(embeddings, self.train_loader.dataset.targets, vis_save_path)

        return embeddings

    def acquire_latent_code(self, inputs):
        return self.model.acquire_latent_code(inputs)

    def acquire_latent_code_allin(self, data):
        with torch.no_grad():
            self.model.eval()
            embeddings = self.model.acquire_latent_code(data).cpu().numpy()
            self.model.train()
        return embeddings

    def preprocess(self, train=True):
        self.build_dataset()
        if train:
            self.log_process = LogWriter(self.tmp_log_path, self.log_path, self.message_queue)
            self.log_process.start()
        self.model_prepare()

    def build_dataset(self):
        knn_cache_path = os.path.join(ConfigInfo.NEIGHBORS_CACHE_DIR,
                                      "{}_k{}.npy".format(self.dataset_name, self.n_neighbors))
        pairwise_cache_path = os.path.join(ConfigInfo.PAIRWISE_DISTANCE_DIR, "{}.npy".format(self.dataset_name))
        check_path_exists(ConfigInfo.NEIGHBORS_CACHE_DIR)
        check_path_exists(ConfigInfo.PAIRWISE_DISTANCE_DIR)

        cdr_dataset = DataSetWrapper(self.batch_size)
        resume_start_epoch = self.resume_start_epoch
        if self.gradient_redefine:
            resume_start_epoch = self.warmup_epochs

        self.train_loader, self.n_samples = cdr_dataset.get_data_loaders(
            resume_start_epoch, self.dataset_name, ConfigInfo.DATASET_CACHE_DIR, self.n_neighbors, knn_cache_path,
            pairwise_cache_path, self.is_image)

        self.batch_num = cdr_dataset.batch_num
        self.model.batch_num = self.batch_num

    def post_epoch(self, ckp_save_inter, epoch, val_inter):
        embeddings = None
        vis_save_path = os.path.join(self.result_save_dir, '{}_vis_{}.jpg'.format(self.dataset_name, epoch))

        if epoch % val_inter == 0:
            if not os.path.exists(self.result_save_dir):
                os.makedirs(self.result_save_dir)
                if self.config_path is not None:
                    shutil.copyfile(self.config_path, os.path.join(self.result_save_dir, "config.yaml"))

            embeddings = self.visualize(vis_save_path, device=self.device)

        # save model
        if epoch % ckp_save_inter == 0:
            if not os.path.exists(self.ckp_save_dir):
                os.makedirs(self.ckp_save_dir)
            self.save_weights(epoch)

        return embeddings

