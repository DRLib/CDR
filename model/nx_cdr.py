import numpy as np
from torch.nn import Module
from model.nce_loss import torch_app_skewnorm_func
from utils.math_utils import get_similarity_function, get_correlated_mask
from utils.umap_utils import find_ab_params
from model.baseline_encoder import *


def exp_ce(data_matrix, t, data_labels, accumulation="MEAN"):
    exp_data = torch.exp(data_matrix / t)
    return ce(exp_data, data_labels, accumulation)


def skewnorm_ce(data_matrix, ratio, data_labels, accumulation="MEAN"):
    sn_data = torch_app_skewnorm_func(data_matrix, ratio)
    return ce(sn_data, data_labels, accumulation)


def ce(data_matrix, data_labels, accumulation="MEAN"):
    softmax_data = data_matrix / torch.sum(data_matrix, dim=1).unsqueeze(1)
    loss = -torch.log(softmax_data[torch.arange(0, data_matrix.shape[0]), data_labels])
    if accumulation == "MEAN":
        return torch.mean(loss)
    elif accumulation == "SUM":
        return torch.sum(loss)
    else:
        return loss


class NX_CDRModel(Module):
    def __init__(self, cfg, device='cuda'):
        Module.__init__(self)
        self.device = device
        self.config = cfg
        self.input_dims = cfg.exp_params.input_dims
        self.encoder_name = "Conv" if isinstance(self.input_dims, int) else "FC"
        self.in_channels = 1 if isinstance(self.input_dims, int) else self.input_dims[-1]

        self.input_size = int(np.sqrt(self.input_dims / self.in_channels))
        self.latent_dim = 2
        self.batch_size = cfg.exp_params.batch_size
        self.similarity_method = "umap"
        self.temperature = cfg.exp_params.temperature

        self.batch_num = 0
        self.max_neighbors = 0
        self.encoder = None
        self.pro_head = None

        self.criterion = None
        self.correlated_mask = get_correlated_mask(2 * self.batch_size)
        self.min_dist = 0.1

        self._a, self._b = find_ab_params(1, self.min_dist)
        self.similarity_func = get_similarity_function(self.similarity_method)

        self.reduction = "mean"
        self.epoch_num = self.config.training_params.epoch_nums
        self.batch_count = 0

    def build_model(self):
        encoder, encoder_out_dims = get_encoder(self.encoder_name, self.input_size, self.input_dims, self.in_channels)
        self.encoder = encoder
        pro_dim = 512
        self.pro_head = nn.Sequential(
            nn.Linear(encoder_out_dims, pro_dim),
            nn.ReLU(),
            nn.Linear(pro_dim, self.latent_dim)
        )

    def preprocess(self):
        self.build_model()
        self.criterion = nn.CrossEntropyLoss(reduction=self.reduction)

    def encode(self, x):
        if x is None:
            return None, None
        reps = self.encoder(x)
        reps = reps.squeeze()

        embeddings = self.pro_head(reps)
        return reps, embeddings

    def forward(self, x, x_sim):
        # get the representations and the projections
        x_reps, x_embeddings = self.encode(x)  # [N,C]

        # get the representations and the projections
        x_sim_reps, x_sim_embeddings = self.encode(x_sim)  # [N,C]

        return x_reps, x_embeddings, x_sim_reps, x_sim_embeddings

    def acquire_latent_code(self, inputs):
        reps, embeddings = self.encode(inputs)
        return embeddings

    def compute_loss(self, x_embeddings, x_sim_embeddings, *args):
        epoch = args[0]
        logits = self.batch_logits(x_embeddings, x_sim_embeddings, *args)
        loss = self._post_loss(logits, x_embeddings, epoch, None, *args)
        return loss

    def _post_loss(self, logits, x_embeddings, epoch, item_weights, *args):
        labels = torch.zeros(logits.shape[0]).to(self.device).long()
        loss = self.criterion(logits / self.temperature, labels)
        return loss

    def batch_logits(self, x_embeddings, x_sim_embeddings, *args):
        self.batch_count += 1
        all_embeddings = torch.cat([x_embeddings, x_sim_embeddings], dim=0)
        representations = all_embeddings.unsqueeze(0).repeat(all_embeddings.shape[0], 1, 1)
        similarity_matrix, pairwise_dist = self.similarity_func(representations.transpose(0, 1), representations,
                                                                self.min_dist)

        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)

        positives = torch.cat([l_pos, r_pos]).view(all_embeddings.shape[0], 1)
        negatives = similarity_matrix[self.correlated_mask].view(all_embeddings.shape[0], -1)

        logits = torch.cat((positives, negatives), dim=1)
        return logits


