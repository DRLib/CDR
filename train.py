#!/usr/bin/env python
# -*- coding:utf-8 -*-
from model.cdr import CDRModel
import torch

from utils.common_utils import get_config
from utils.constant_pool import *
import argparse
from experiments.trainer import CDRTrainer
import os

log_path = "log.txt"


def cdr_pipeline(config_path):

    cfg.merge_from_file(config_path)
    method_name = CDR_METHOD if cfg.exp_params.gradient_redefine else NX_CDR_METHOD
    result_save_dir = ConfigInfo.RESULT_SAVE_DIR.format(method_name, cfg.exp_params.n_neighbors)
    if not os.path.exists(result_save_dir):
        os.makedirs(result_save_dir)

    clr_model = CDRModel(cfg, device=device)
    trainer = CDRTrainer(clr_model, cfg.exp_params.dataset, cfg, result_save_dir, config_path,
                         device=device, log_path=log_path)
    trainer.train_for_visualize()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", type=str, default="configs/CDR.yaml", help="configuration file path")
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    cfg = get_config()
    device = args.device
    cdr_pipeline(args.configs)
