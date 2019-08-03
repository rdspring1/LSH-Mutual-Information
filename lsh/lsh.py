import collections
import os
import sys
import math
import random
import numpy as np
import numpy.random
import scipy as sp
import scipy.stats

from clsh import pyLSH
import torch

class LSH:
    def __init__(self, func_, K_, L_, threads_=8):
        self.func = func_
        self.K = K_
        self.L = L_
        self.lsh_ = pyLSH(self.K, self.L, threads_)

        self.sample_size = 0
        self.count = 0

    def stats(self):
        avg_size = self.sample_size // max(self.count, 1)
        self.sample_size = 0
        self.count = 0
        print("Avg. Sample Size:", avg_size)

    def insert(self, item_id, item):
        fp = self.func.hash(item).int().cpu().numpy()
        self.lsh_.insert(np.squeeze(fp), item_id)

    def insert_multi(self, items, N):
        fp = self.func.hash(items).int().cpu().numpy()
        self.lsh_.insert_multi(fp, N)

    def query_remove_matrix(self, items, labels, total_size):
        # for each data sample, query lsh data structure, remove accidental hit
        # find maximum number of samples
        # create matrix and pad appropriately
        batch_size, D = items.size()
        fp = self.func.hash(items).int().cpu().numpy()
        result, total_count = self.lsh_.query_matrix(fp, labels.cpu().numpy(), batch_size, total_size)
        batch_size, ssize = result.shape
        self.sample_size += total_count
        self.count += batch_size
        return result

    def query_remove(self, item, label):
        fp = self.func.hash(item).int().cpu().numpy()
        result = self.lsh_.query(np.squeeze(fp))
        if label in result:
            result.remove(label)
        self.sample_size += len(result)
        self.count += 1
        return list(result)

    def query(self, item):
        fp = self.func.hash(item).int().cpu().numpy()
        result = list(self.lsh_.query(np.squeeze(fp)))
        self.sample_size += len(result)
        self.count += 1
        return result

    def query_multi(self, items, N):
        fp = self.func.hash(items).int().cpu().numpy()
        return list(self.lsh_.query_multi(fp, N))

    def clear(self):
        self.lsh_.clear()
