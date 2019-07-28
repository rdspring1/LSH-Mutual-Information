import math
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import enum
class MI(enum.IntEnum):
    NCE = 0
    NWJ = 1
    TUBA = 2
    INTERPOLATE = 3
    IS = 4
    LSH = 5

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

    def forward(self, x, y, target, args=None):
        h = self.h2(F.relu(self.h1(y)))
        g = self.g2(F.relu(self.g1(x)))
        scores = torch.matmul(h, g.t())
        #return self.infoNCE(scores, target)
        return self.importance_sampling(scores, target, args['desired_batch_size'])

    def my_log_softmax(self, scores, target, p):
        # log_sum_exp
        max_scores, max_indices = torch.max(scores, dim=1)
        scores -= max_scores

        eye = torch.zeros_like(scores)
        eye[0,target] = 1e38
        ps = scores[0, target]
        ns = scores - eye 
        eps = torch.exp(ps)
        ens = torch.exp(ns) / (p + 1e-6)

        partition = torch.sum(ens) + eps
        return scores - torch.log(partition)

    def importance_sampling(self, scores, target, desired_size):
        desired_size = torch.LongTensor([desired_size]).to(self.device)
        batch_size = torch.LongTensor([scores.size(0)]).to(self.device)
        N = torch.LongTensor([scores.size(1)]).to(self.device)

        # use importance sampling to artifically increase batch size
        p = N.float() / desired_size.float()
        smax = self.my_log_softmax(scores, target, p)

        # replace local batch size with desired batch size
        nll = -F.nll_loss(smax, target, reduction='mean')
        mi = torch.log(desired_size.float()) + nll
        return mi

    def infoNCE(self, scores, target):
        batch_size = torch.LongTensor([scores.size(0)]).to(self.device)
        N = torch.LongTensor([scores.size(1)]).to(self.device)
        nll = -F.cross_entropy(scores, target, reduction='mean')
        mi = torch.log(N.float()) + nll
        return mi
