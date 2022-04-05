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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", type=str, default="configs/CDR.yaml", help="configuration file path")
    parser.add_argument("--ckpt", type=str, default="model_weights/usps.pth.tar")
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    cfg = get_config()
    cfg.merge_from_file(args.configs)
    device = args.device
    clr_model = CDRModel(cfg, device=device)
    trainer = CDRTrainer(clr_model, cfg.exp_params.dataset, cfg, None, args.configs,
                         device=device, log_path=log_path)
    trainer.load_weights_visualization(args.ckpt, vis_save_path="vis.jpg", device=device)
