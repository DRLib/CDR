from model.cdr import CDRModel
from utils.link_utils import MUST_LINK, CANNOT_LINK
import torch


class ICDRModel(CDRModel):
    def __init__(self, cfg, device='cuda'):
        CDRModel.__init__(self, cfg, device)
        self.cur_ml_num = 0
        self.cur_cl_num = 0
        self.cur_ml_indices = None
        self.cur_cl_indices = None
        self.max_prob_thresh = 0.95
        self.ml_strength = 0
        self.cl_strength = 0
        self.gather_weight = 0.2

    def link_stat_update(self, finetune_epochs, steady_epoch, ml_strength, cl_strength):
        self.separate_epoch = self.epoch_num
        self.steady_epoch = self.separate_epoch + finetune_epochs - steady_epoch
        self.epoch_num = self.separate_epoch + finetune_epochs
        self.ml_strength = ml_strength
        self.cl_strength = cl_strength

    def batch_logits(self, x_embeddings, x_sim_embeddings, *args):
        logits = super().batch_logits(x_embeddings, x_sim_embeddings, *args)

        link_embeddings, cur_link_types, related_indices = args[-4:-1]
        cur_link_num = len(related_indices)
        if cur_link_num == 0:
            self.cur_ml_num = 0
            self.cur_cl_num = 0
            return logits

        all_embeddings = torch.cat([x_embeddings, x_sim_embeddings], dim=0)
        cl_indices = torch.where(torch.tensor(cur_link_types) == CANNOT_LINK)[0]
        ml_indices = torch.where(torch.tensor(cur_link_types) == MUST_LINK)[0]
        self.cur_cl_indices = cl_indices
        self.cur_ml_indices = ml_indices
        self.cur_ml_num = len(ml_indices)
        self.cur_cl_num = len(cl_indices)

        if self.cur_ml_num > 0:
            h_embeddings = all_embeddings[related_indices[ml_indices]]
            t_embeddings = link_embeddings[ml_indices]

            positives = self.similarity_func(h_embeddings, t_embeddings, self.min_dist)[0].view(-1, 1)
            negatives = logits[related_indices[ml_indices], 1:].view(self.cur_ml_num, -1)
            ml_logits = torch.cat([positives, negatives.detach()], dim=1)
            logits = torch.cat((logits, ml_logits), dim=0)

        if self.cur_cl_num > 0:
            h_embeddings = all_embeddings[related_indices[cl_indices]]
            t_embeddings = link_embeddings[cl_indices]

            negative = self.similarity_func(h_embeddings, t_embeddings, self.min_dist)[0].view(-1, 1)
            other_negatives = logits[related_indices[cl_indices], 1:-1].view(self.cur_cl_num, -1)
            positive = logits[related_indices[cl_indices], 0].view(self.cur_cl_num, 1).clone()

            indices = torch.where(positive > self.max_prob_thresh)[0]
            positive[indices] = positive[indices].detach()

            cl_logits = torch.cat([positive, other_negatives.detach(), negative], dim=1)
            logits = torch.cat((logits, cl_logits), dim=0)

        return logits

    def compute_loss(self, x_embeddings, x_sim_embeddings, *args):
        epoch = args[0]
        total_link_weights = args[-1]
        logits = self.batch_logits(x_embeddings, x_sim_embeddings, *args)
        loss = self._post_loss(logits, x_embeddings, epoch, total_link_weights, *args)
        return loss

    def _post_loss(self, logits, x_embeddings, epoch, total_link_weights, *args):
        total_link_num = self.cur_ml_num + self.cur_cl_num
        normal_num = self.batch_size * 2
        normal_loss = super()._post_loss(logits[:normal_num], x_embeddings[:normal_num], epoch, None, *args)
        if total_link_num == 0:
            return normal_loss
        else:
            total_link_weights = torch.tensor(total_link_weights, dtype=torch.float).to(self.device)
            t = self.temperature
            ml_loss, cl_loss = 0, 0
            link_logits = logits[-total_link_num:]
            if self.cur_ml_num > 0:
                ml_logits = link_logits[:self.cur_ml_num]
                ml_link_weight = total_link_weights[self.cur_ml_indices]

                indices = torch.where(ml_logits[:, 0] < self.max_prob_thresh)[0]
                if len(indices) > 0:
                    ml_loss = super()._post_loss(ml_logits[indices], None, epoch, ml_link_weight[indices], *args)

            if self.cur_cl_num > 0:
                cl_logits = link_logits[-self.cur_cl_num:]
                cl_link_weight = total_link_weights[self.cur_cl_indices]
                pos_cl_logits = torch.cat([cl_logits[:, 0].unsqueeze(1), cl_logits[:, 1:].detach()], dim=1)
                neg_cl_logits = torch.cat([1 - cl_logits[:, -1].unsqueeze(1), cl_logits[:, 1:].detach()], dim=1)

                indices = torch.where(cl_logits[:, -1] > self.lower_thresh)[0]

                cl_pos_loss = self.criterion(pos_cl_logits[indices], torch.tensor(t), cl_link_weight[indices])
                cl_neg_loss = self.criterion(neg_cl_logits[indices], torch.tensor(t), cl_link_weight[indices])

                cl_loss = self.gather_weight * cl_pos_loss + self.cl_strength * cl_neg_loss

            link_loss = (self.ml_strength * ml_loss + cl_loss)
            loss = (normal_loss + link_loss)
            return loss
