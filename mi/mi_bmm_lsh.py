import math
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

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

    def cosine_similarity(self, scores, x, y):
        # y - (batch_size, D)
        # x - (batch_size, 1+num_samples, D)
        x_norm = torch.norm(x, dim=-1, keepdim=True)
        y_norm = torch.norm(y, dim=-1, keepdim=True)
        norms = torch.squeeze(torch.bmm(x_norm, torch.unsqueeze(y_norm, dim=-1)), dim=-1)
        asim = 1. - torch.acos(scores / norms) / math.pi
        return asim

    def embed_y(self, y):
        return self.h2(F.relu(self.h1(y)))

    def embed_x(self, x):
        return self.g2(F.relu(self.g1(x)))

    def forward(self, x, y, mask, args):
        # y = (batch_size, D)
        # x = (batch_size, 1+num_samples, D)
        # positive sample is always first
        h = self.h2(F.relu(self.h1(y)))
        g = self.g2(F.relu(self.g1(x)))
        scores = torch.squeeze(torch.bmm(g, torch.unsqueeze(h, dim=-1)), dim=-1)

        csim = self.cosine_similarity(scores.detach(), g.detach(), h.detach())
        p = 1. - torch.pow((1. - torch.pow(csim, args['k'])), args['l'])
        return self.importance_sampling(scores, mask, p, args['desired_batch_size'])

    def my_log_softmax(self, scores, mask, p):
        max_scores, max_indices = torch.max(scores, dim=-1, keepdim=True)
        scores -= max_scores

        ps = scores[:,0]
        eps = torch.unsqueeze(torch.exp(ps), dim=-1)

        eye = torch.zeros_like(scores)
        eye[:,0] = 1e38
        ns = scores - eye
        ens = torch.exp(ns) / torch.clamp(p,1e-38, 1.)
        ens = torch.mul(ens, mask)

        # scores not present in lsh samples do not contribute to partition function
        partition = torch.sum(ens, dim=-1, keepdim=True) + eps
        return scores - torch.log(partition)

    def importance_sampling(self, scores, mask, p, desired_size):
        desired_size = torch.LongTensor([desired_size]).to(self.device)
        target = torch.LongTensor([0] * scores.size(0)).to(self.device)
        # use importance sampling to artifically increase batch size
        smax = self.my_log_softmax(scores, mask, p)
        # replace local batch size with desired batch size
        nll = -F.nll_loss(smax, target, reduction='mean')
        mi = torch.log(desired_size.float()) + nll
        return mi
