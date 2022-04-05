#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import numpy as np
from multiprocessing import Queue

CDR_METHOD = "CDR"
NX_CDR_METHOD = "NX_CDR"


class ProjectSettings:
    LABEL_COLORS = {0: 'blue', 1: 'orange', 2: 'green', 3: 'red', 4: 'blueviolet', 5: 'maroon', 6: 'deeppink',
                    7: 'greenyellow', 8: 'olive', 9: 'cyan', 10: 'yellow', 11: 'purple'}


class ConfigInfo:
    # method_name, dataset_name, method_name+dataset_name.ckpt
    MODEL_CONFIG_PATH = "./configs/"
    RESULT_SAVE_DIR = r"results\{}\n{}"
    NEIGHBORS_CACHE_DIR = r"data\neighbors"
    PAIRWISE_DISTANCE_DIR = r"data\pair_distance"
    DATASET_CACHE_DIR = r"data\H5 Data"
    IMAGE_DIR = r"static/images"

    DATASETS_META = ["name", "num", "dim", "type"]
    AVAILABLE_DATASETS = []
