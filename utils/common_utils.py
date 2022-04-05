#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import os
import os.path
import time
from multiprocessing import Queue

import numpy as np
import yaml
from easydict import EasyDict as edict
from sklearn.decomposition import PCA
from yaml import FullLoader

from dataset.datasets import MyImageDataset, MyTextDataset

DATE_TIME_ADJOIN_FORMAT = "%Y%m%d_%Hh%Mm%Ss"


def check_path_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_dataset(dataset_name, root_dir, train, transform=None, is_image=True):
    if is_image:
        dataset = MyImageDataset(dataset_name, root_dir, train, transform)
    else:
        dataset = MyTextDataset(dataset_name, root_dir, train)
    return dataset


def get_principle_components(data, attr_names=None, target_components=16):
    n_samples = data.shape[0]
    flattened_data = np.reshape(data, (n_samples, -1))

    if flattened_data.shape[1] <= target_components:
        low_data = flattened_data
    else:
        pca = PCA()
        z = pca.fit_transform(flattened_data)
        low_data = z[:, :target_components]

    if attr_names is None:
        attr_names = ["A{}".format(i) for i in range(low_data.shape[1])]
    return low_data, attr_names


def time_stamp_to_date_time_adjoin(time_stamp):
    time_array = time.localtime(time_stamp)
    return time.strftime(DATE_TIME_ADJOIN_FORMAT, time_array)


class QueueSet:
    def __init__(self):
        self.eval_data_queue = Queue()
        self.eval_result_queue = Queue()

        self.test_eval_data_queue = Queue()
        self.test_eval_result_queue = Queue()


class YamlParser(edict):
    """
    This is yaml parser based on EasyDict.
    """

    def __init__(self, cfg_dict=None, config_file=None):
        if cfg_dict is None:
            cfg_dict = {}

        if config_file is not None:
            assert (os.path.isfile(config_file))
            with open(config_file, 'r') as fo:
                cfg_dict.update(yaml.load(fo.read(), Loader=FullLoader))

        super(YamlParser, self).__init__(cfg_dict)

    def merge_from_file(self, config_file):
        with open(config_file, 'r') as fo:
            self.update(yaml.load(fo.read(), Loader=FullLoader))

    def merge_from_dict(self, config_dict):
        self.update(config_dict)


def get_config(config_file=None):
    return YamlParser(config_file=config_file)