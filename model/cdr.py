from model.nce_loss import NT_Xent, Mixture_NT_Xent
from model.nx_cdr import NX_CDRModel
import torch


class CDRModel(NX_CDRModel):
    def __init__(self, cfg, device='cuda'):
        NX_CDRModel.__init__(self, cfg, device)

        self.a = torch.tensor(-40)
        self.miu = torch.tensor(cfg.exp_params.separate_upper)
        self.lower_thresh = torch.tensor(0.015)
        self.scale = torch.tensor(0.13)
        self.alpha = torch.tensor(5)

        self.separate_epoch = int(self.epoch_num * cfg.exp_params.separation_begin_ratio)
        self.steady_epoch = int(self.epoch_num * cfg.exp_params.steady_begin_ratio)

    def preprocess(self):
        self.build_model()
        self.criterion = NT_Xent.apply

    def _post_loss(self, logits, x_embeddings, epoch, item_weights, *args):
        if self.separate_epoch <= epoch <= self.steady_epoch:
            epoch_ratio = torch.tensor((epoch - self.separate_epoch) / (self.steady_epoch - self.separate_epoch))
            cur_lower_thresh = 0.001 + (self.lower_thresh - 0.001) * epoch_ratio
            loss = Mixture_NT_Xent.apply(logits, torch.tensor(self.temperature), self.alpha, self.a, self.miu,
                                         cur_lower_thresh, self.scale, item_weights)
        else:
            loss = self.criterion(logits, torch.tensor(self.temperature), item_weights)
        return loss
