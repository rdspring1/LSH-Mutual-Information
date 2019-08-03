import math
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from mi_data import MI

class MI_Estimator(nn.Module):
    def __init__(self, device, D, ED, HD):
        super(MI_Estimator, self).__init__()
        self.device = device
        self.h1 = nn.Linear(D, HD)
        self.h2 = nn.Linear(HD, ED)

        self.g1 = nn.Linear(D, HD)
        self.g2 = nn.Linear(HD, ED)

        self.fn1 = nn.Linear(D, HD)
        self.fn2 = nn.Linear(HD, 1)

    def cosine_similarity(self, x, y):
        # x - (batch_size, D)
        # y - (batch_size, D)
        h = self.h2(F.relu(self.h1(y)))
        g = self.g2(F.relu(self.g1(x)))
        scores = torch.matmul(h, g.t())

        x_norm = torch.norm(h, dim=-1, keepdim=True)
        y_norm = torch.norm(g, dim=-1, keepdim=True)
        norms = torch.matmul(x_norm, y_norm.t())

        asim = 1. - torch.acos(scores / norms) / math.pi
        return asim

    def forward(self, x, y, mi_type, args=None):
        h = self.h2(F.relu(self.h1(y)))
        g = self.g2(F.relu(self.g1(x)))
        scores = torch.matmul(h, g.t())

        if mi_type == MI.NCE:
            return self.infoNCE(scores)
        elif mi_type == MI.NWJ:
            return self.tuba(scores, log_baseline=1.)
        elif mi_type == MI.TUBA:
            # unnormalized baseline
            log_baseline = self.fn2(F.relu(self.fn1(y)))
            return self.tuba(scores, log_baseline)
        elif mi_type == MI.INTERPOLATE:
            log_baseline = self.fn2(F.relu(self.fn1(y)))
            at = torch.FloatTensor([args['alpha']])
            return self.interpolate(scores, log_baseline, at)
        elif mi_type == MI.IS:
            return self.importance_sampling(scores, args['desired_batch_size'])

    def my_log_softmax(self, scores, p, dim):
        # log_sum_exp
        batch_size = scores.size(0)

        max_scores, max_indices = torch.max(scores, dim=1, keepdim=True)
        scores -= max_scores
        eye = torch.eye(batch_size).to(self.device)
        ps = torch.diag(eye * scores)
        ns = scores - (eye * 1e38)
        eps = torch.exp(ps)
        ens = torch.exp(ns) / p

        escores = torch.diag(eps) + ens
        partition = torch.sum(escores, dim, keepdim=True)
        return scores - torch.log(partition)

    def importance_sampling(self, scores, desired_size):
        desired_size = torch.LongTensor([desired_size]).to(self.device)
        batch_size = torch.LongTensor([scores.size(0)]).to(self.device)

        # use importance sampling to artifically increase batch size
        p = batch_size.float() / desired_size.float()
        smax = self.my_log_softmax(scores, p, dim=-1)
        nll = torch.mean(torch.diag(smax))

        #target = torch.arange(batch_size.item()).to(self.device)
        #nll = -F.nll_loss(smax, target, reduction='mean')

        # replace local batch size with desired batch size
        mi = torch.log(desired_size.float()) + nll
        return mi

    def infoNCE(self, scores):
        batch_size = torch.LongTensor([scores.size(0)]).to(self.device)
        #nll = torch.mean(torch.diag(scores) - torch.logsumexp(scores, dim=1))
        target = torch.arange(batch_size.item()).to(self.device)
        nll = -F.cross_entropy(scores, target, reduction='mean')
        mi = torch.log(batch_size.float()) + nll
        return mi

    def logmeanexp_nodiag(self, scores):
        batch_size = torch.LongTensor([scores.size(0)]).to(self.device)
        num_elem = batch_size * (batch_size-1.)
        eye = torch.eye(batch_size.item()).to(self.device)
        no_diag_scores = scores - (eye * 1e38)
        logsumexp = torch.logsumexp(no_diag_scores.view(-1), dim=-1)
        return logsumexp - torch.log(num_elem.float())

    def tuba(self, scores, log_baseline=None):
        if log_baseline is not None:
            scores -= log_baseline
        joint = torch.mean(torch.diag(scores))
        marginal = self.logmeanexp_nodiag(scores)
        return 1. + joint - marginal

    def interpolate(self, scores, baseline, alpha):
        softplus = lambda x: torch.log(torch.exp(x) + 1)
        softplus_inverse = lambda x: torch.log(torch.exp(x) - 1)

        batch_size = torch.LongTensor([scores.size(0)])
        def _compute_log_loo_mean(scores):
            max_scores, max_indices = torch.max(scores, dim=1, keepdim=True)
            lse_minus_max = torch.logsumexp(scores - max_scores, dim=1, keepdim=True)
            d = lse_minus_max + (max_scores - scores)
            d_ok = 1. - torch.eq(d, 0.)
            safe_d = torch.where(d_ok, d, torch.ones_like(d))
            loo_lse = scores + softplus_inverse(safe_d)
            loo_lme = loo_lse - torch.log(batch_size.float() - 1.)
            return loo_lme

        def _log_interpolate(log_a, log_b, alpha):
            log_alpha = -softplus(-alpha)
            log_1_minus_alpha = -softplus(alpha)
            log_part = torch.stack((log_alpha + log_a, log_1_minus_alpha + log_b))
            return torch.logsumexp(log_part, dim=0)

        # InfoNCE baseline
        nce_baseline = _compute_log_loo_mean(scores)
        # interpolated baseline
        interpolated_baseline = _log_interpolate(nce_baseline, baseline.repeat(1, batch_size), alpha)
        # joint term
        num_elem = batch_size * (batch_size-1.)
        critic_joint = torch.diag(scores)[:, None] - interpolated_baseline
        joint = (torch.sum(critic_joint) - torch.sum(torch.diag(critic_joint))) / num_elem.float()

        # marginal term
        critic_marginal = scores - torch.diag(interpolated_baseline)[:, None]
        marginal = torch.exp(self.logmeanexp_nodiag(critic_marginal))
        return 1. + joint - marginal

def train(model, optimizer, iterations, D, MI, batch_size, mi_type, args):
    model.train()
    for batch_idx in range(iterations):
        optimizer.zero_grad()
        x, y = generate(D, MI, batch_size)
        mi = model(x, y, mi_type, args)

        loss = -mi
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print('Train {}\tLoss: {:.6f}\tMI: {:.6f}'
                    .format(batch_idx, loss.item(), mi.item()))

def main():
    N = 5000
    BS = 64
    d = 20
    ed = 32
    hd = 512
    mi = 4
    lr = 5e-4
    alpha = 0.01

    model = MI_Estimator(d, ed, hd)
    optimizer = optim.Adam(model.parameters(), lr)
    train(model, optimizer, N, d, mi, BS, MI.NCE)
    #train(model, optimizer, N, d, mi, BS, MI.NWJ)
    #train(model, optimizer, N, d, mi, BS, MI.TUBA)
    #train(model, optimizer, N, d, mi, BS, MI.INTERPOLATE, alpha)

if __name__ == '__main__':
    main()
