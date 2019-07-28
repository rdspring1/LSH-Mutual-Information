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
        # x - (batch_size, D)
        # y - (batch_size, D)
        x_norm = torch.norm(x, dim=1, keepdim=True)
        y_norm = torch.norm(y, dim=1, keepdim=True)
        norms = torch.mul(x_norm, y_norm).t()
        asim = 1. - torch.acos(scores / norms) / math.pi
        return asim

    def embed_y(self, y):
        h = self.h2(F.relu(self.h1(y)))
        return h

    def embed_x(self, x):
        g = self.g2(F.relu(self.g1(x)))
        return g

    def forward(self, x, y, args):
        h = self.h2(F.relu(self.h1(y)))
        g = self.g2(F.relu(self.g1(x)))
        scores = torch.matmul(h, g.t())

        csim = self.cosine_similarity(scores.detach(), g.detach(), h.detach())
        p = 1. - torch.pow((1. - torch.pow(csim, args['k'])), args['l'])
        return self.importance_sampling(scores, p, args['desired_batch_size'])

    def my_log_softmax(self, scores, p):
        max_scores, max_indices = torch.max(scores, dim=1)
        scores -= max_scores

        eye = torch.zeros_like(scores)
        eye[:,0] = 1e38
        ps = scores[:,0]

        ns = scores - eye 
        eps = torch.exp(ps)
        ens = torch.exp(ns) / (p + 1e-6)

        partition = torch.sum(ens) + eps
        return scores - torch.log(partition)

    def importance_sampling(self, scores, p, desired_size):
        desired_size = torch.LongTensor([desired_size]).to(self.device)
        target = torch.LongTensor([0] * scores.size(0)).to(self.device)
        # use importance sampling to artifically increase batch size
        smax = self.my_log_softmax(scores, p)
        # replace local batch size with desired batch size
        nll = -F.nll_loss(smax, target, reduction='mean')
        mi = torch.log(desired_size.float()) + nll
        return mi
