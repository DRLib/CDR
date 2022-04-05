from torch.autograd import Function
import torch

import math

normal_obj = torch.distributions.normal.Normal(0, 1)


def torch_norm_pdf(data):
    return torch.exp(-torch.square(data) / 2) / math.sqrt(2 * math.pi)


def torch_norm_cdf(data):
    global normal_obj
    return normal_obj.cdf(data)


def torch_skewnorm_pdf(data, a, loc, scale):
    y = (data - loc) / scale
    output = 2 * torch_norm_pdf(y) * torch_norm_cdf(a * y) / scale
    return output


def torch_app_skewnorm_func(data, r, a=-40, loc=0.11, scale=0.13):
    y = torch_skewnorm_pdf(data, a, loc, scale)
    y = y * r
    return y


class NT_Xent(Function):

    @staticmethod
    def forward(ctx, probabilities, t, item_weights):
        exp_prob = torch.exp(probabilities / t)

        similarities = exp_prob / torch.sum(exp_prob, dim=1).unsqueeze(1)

        ctx.save_for_backward(similarities, t, item_weights)

        pos_loss = -torch.log(similarities[:, 0]).mean()

        return pos_loss

    @staticmethod
    def backward(ctx, grad_output):
        similarities, t, item_weights = ctx.saved_tensors
        pos_grad_coeff = -((torch.sum(similarities, dim=1) - similarities[:, 0]) / t).unsqueeze(1)
        neg_grad_coeff = similarities[:, 1:] / t
        grad = torch.cat([pos_grad_coeff, neg_grad_coeff], dim=1) * grad_output / similarities.shape[0]
        if item_weights is not None:
            grad *= item_weights.view(-1, 1)
        return grad, None, None


class Mixture_NT_Xent(Function):

    @staticmethod
    def forward(ctx, probabilities, t, alpha, a, loc, lower_thresh, scale, item_weight):

        def nt_xent_grad(data, tau):
            exp_prob = torch.exp(data / tau)
            norm_exp_prob = exp_prob / torch.sum(exp_prob, dim=1).unsqueeze(1)
            gradients = norm_exp_prob[:, 1:] / tau
            return norm_exp_prob, gradients

        similarities, exp_neg_grad_coeff = nt_xent_grad(probabilities, t)

        skewnorm_prob = torch_skewnorm_pdf(probabilities[:, 1:], a, loc, scale)
        skewnorm_similarities = skewnorm_prob / torch.sum(skewnorm_prob, dim=1).unsqueeze(1)
        sn_max_val_indices = torch.argmax(skewnorm_similarities, dim=1)
        rows = torch.arange(0, skewnorm_similarities.shape[0], 1)
        skewnorm_max_value = skewnorm_similarities[rows, sn_max_val_indices].unsqueeze(1)
        ref_exp_value = exp_neg_grad_coeff[rows, sn_max_val_indices].unsqueeze(1)

        raw_alpha = ref_exp_value / skewnorm_max_value

        ctx.save_for_backward(probabilities, similarities, t, skewnorm_similarities, loc, lower_thresh, alpha,
                              raw_alpha, item_weight)

        pos_loss = -torch.log(similarities[:, 0]).mean()

        return pos_loss

    @staticmethod
    def backward(ctx, grad_output):
        prob, exp_sims, t, sn_sims, loc, lower_thresh, alpha, raw_alpha, item_weights = ctx.saved_tensors

        pos_grad_coeff = -((torch.sum(exp_sims, dim=1) - exp_sims[:, 0]) / t).unsqueeze(1)
        high_thresh = loc
        sn_sims[prob[:, 1:] < lower_thresh] = 0
        sn_sims[prob[:, 1:] >= high_thresh] = 0
        exp_sims[:, 1:][prob[:, 1:] < lower_thresh] = 0

        neg_grad_coeff = exp_sims[:, 1:] / t + alpha * sn_sims * raw_alpha
        grad = torch.cat([pos_grad_coeff, neg_grad_coeff], dim=1) * grad_output / exp_sims.shape[0]
        if item_weights is not None:
            grad *= item_weights.view(-1, 1)
        return grad, None, None, None, None, None, None, None

