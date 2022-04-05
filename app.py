#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import argparse
import os
from datetime import timedelta

import h5py
from flask import Flask, render_template, request
from experiments.icdr_trainer import ICLPTrainer
from model.cdr import CDRModel
from model.icdr import ICDRModel
from utils.constant_pool import *
from utils.common_utils import get_principle_components, get_config
from utils.link_utils import LinkInfo
import numpy as np


app = Flask(__name__)
experimenter: ICLPTrainer
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = timedelta(seconds=1)


def wrap_results(embeddings, principle_comps=None, attr_names=None):
    ret_dict = {}
    ret_dict["embeddings"] = embeddings.tolist()
    ret_dict["label"] = experimenter.get_label()
    if principle_comps is not None:
        ret_dict["low_data"] = principle_comps.tolist()
        ret_dict["attrs"] = attr_names
    return ret_dict


def build_link_info(embeddings, min_dist):
    links = request.form.get("links")
    link_spreads = request.form.get("link_spreads")
    finetune_epochs = request.form.get("finetune_epochs", type=int)

    links = np.array(eval(links))
    print(links)
    link_spreads = np.array(eval(link_spreads))

    if links.shape[0] == 0:
        experimenter.link_info = None
        return experimenter.link_info

    if experimenter.link_info is None:
        experimenter.link_info = LinkInfo(links, link_spreads, finetune_epochs, embeddings, min_dist)
    else:
        experimenter.link_info.process_cur_links(links, link_spreads, embeddings)

    return experimenter.link_info


def update_config():
    global configs
    ds_name = request.form.get("dataset", type=str)
    configs.exp_params.dataset = ds_name
    configs.exp_params.n_neighbors = request.form.get("n_neighbors", type=int)
    configs.training_params.epoch_nums = request.form.get("epoch_nums", type=int)
    configs.exp_params.input_dims = request.form.get("input_dims", type=int)
    configs.exp_params.split_upper = request.form.get("split_upper", type=float)
    configs.exp_params.batch_size = int(request.form.get("n_samples", type=int) / 10)


def load_experiment(cfg):
    method_name = CDR_METHOD if cfg.exp_params.gradient_redefine else NX_CDR_METHOD
    result_save_dir = ConfigInfo.RESULT_SAVE_DIR.format(method_name, cfg.exp_params.n_neighbors)
    # 创建CLP模型
    clr_model = ICDRModel(cfg, device=device)
    global experimenter
    experimenter = ICLPTrainer(clr_model, cfg.exp_params.dataset, cfg, result_save_dir, None, device=device)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/load_dataset_list")
def load_dataset_list():
    data = []
    for item in ConfigInfo.AVAILABLE_DATASETS:
        data_obj = {}
        for i, k in enumerate(ConfigInfo.DATASETS_META):
            data_obj[k] = item[i]
        data.append(data_obj)

    return {"data": data}


@app.route("/train_for_vis", methods=["POST"])
def train_for_vis():
    update_config()
    load_experiment(configs)

    embeddings = experimenter.train_for_visualize()
    principle_comps, attr_names = get_principle_components(experimenter.dataset.data, attr_names=None)
    ret_dict = wrap_results(embeddings, principle_comps, attr_names)
    return ret_dict


@app.route("/constraint_resume", methods=["POST"])
def constraint_resume():
    update_config()
    link_info = build_link_info(experimenter.pre_embeddings, experimenter.configs.exp_params.min_dist)
    ft_epoch = request.form.get("finetune_epochs", type=int)

    ml_strength = request.form.get("ml_strength", type=float)
    cl_strength = request.form.get("cl_strength", type=float)
    experimenter.update_link_stat(link_info, is_finetune=True, finetune_epoch=ft_epoch)

    if link_info is not None:
        experimenter.model.link_stat_update(ft_epoch, experimenter.steady_epoch, ml_strength, cl_strength)

    embeddings = experimenter.resume_train(ft_epoch)
    return wrap_results(embeddings)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", type=str, default="configs/ICDR.yaml", help="configuration file path")
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def load_available_data():
    for item in os.listdir(ConfigInfo.DATASET_CACHE_DIR):
        ds = item.split(".")[0]
        n_samples, dims = np.array(h5py.File(os.path.join(ConfigInfo.DATASET_CACHE_DIR, item), "r")['x']).shape
        ds_type = "image" if os.path.exists(os.path.join(ConfigInfo.IMAGE_DIR, ds)) else "tabular"
        ConfigInfo.AVAILABLE_DATASETS.append([ds, n_samples, dims, ds_type])


if __name__ == '__main__':
    app.jinja_env.variable_start_string = '[['
    app.jinja_env.variable_end_string = ']]'

    args = parse_args()
    device = args.device
    config_path = args.configs
    configs = get_config()
    configs.merge_from_file(config_path)
    load_available_data()
    load_experiment(configs)
    app.run(debug=False)
