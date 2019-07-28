import sys
import time

import pandas as pd # used for exponential moving average
from scipy.special import logit
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

sys.path.append("../lsh")
sys.path.append("../mi")
from lsh import LSH
from matrix_simhash import SimHash

from mi_data import generate_dataset, mi_schedule, MI
#from mi_lsh import MI_Estimator
from mi_bmm_lsh import MI_Estimator

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

num_iterations = 20000
batch_size = 1
wsize = 50
total_size = num_iterations * batch_size
d = 20
ed = 32
K = 10
L = 32
device='cuda'


estimators = {
        'LSH': dict(mi_type=MI.LSH, args=dict(desired_batch_size=total_size, k=K, l=L)),
        }

def build(lsh, model, xs, bs=100):
    lsh.clear()
    n_iter = xs.size(0) // bs
    start_time = time.time()
    for batch_idx in range(n_iter):
        start = batch_idx * bs
        end = start + bs
        x = xs[start:end]
        xe = model.embed_x(x)
        lsh.insert_multi(xe, bs)
    end_time = time.time()
    #print("LSH Built {:2f}".format(end_time - start_time))

def train(device, data, schedule, mi_type, args):
    model = MI_Estimator(device, D=d, ED=ed, HD=256)
    model.to(device)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=5e-4)

    xs, ys = data
    xs = xs.to(device)
    ys = ys.to(device)

    lsh = LSH(SimHash(ed, K, L), K, L)

    estimates = []
    avg_estimate = []

    id_set = set()
    n_iters = num_iterations * batch_size
    for batch_idx in range(n_iters):
        iteration = batch_idx // batch_size
        MI = schedule[iteration]

        t = 10 if batch_idx <= 1000 else 100
        if batch_idx % t == 0:
            build(lsh, model, xs)

        optimizer.zero_grad()

        y = ys[batch_idx:batch_idx+1]
        ey = model.embed_y(y)

        id_list = lsh.query(ey)
        id_set = id_set.union(set(id_list))
        indices = torch.LongTensor(id_list).to(device)

        nx = F.embedding(indices, xs)
        px = xs[batch_idx:batch_idx+1]
        x = torch.cat([px, nx], dim=0)
        x = torch.unsqueeze(x, dim=0)

        mi = model(x, y, args)
        loss = -mi
        loss.backward()
        optimizer.step()

        avg_estimate.append(mi.item())
        if (batch_idx+1) % 100 == 0:
            '''
            asim = model.cosine_similarity(x, y)
            true = torch.mean(torch.diag(asim))
            neye = 1. - torch.eye(batch_size).to(device)
            noise = torch.sum(torch.mul(asim, neye)).item() / (batch_size * (batch_size-1))
            print("MI:{} true: {:.4f}, noise: {:.4f}".format(MI, true, noise))
            '''
            avg_mi = sum(avg_estimate) / float(len(avg_estimate))
            print('{} {} MI:{}, E_MI: {:.6f}'.format(mi_type.name, batch_idx+1, MI, avg_mi))
            sys.stdout.flush()

        if (batch_idx+1) % wsize == 0:
            print(len(id_set), len(id_set)//wsize)
            id_set.clear()
            avg_mi = sum(avg_estimate) / float(len(avg_estimate))
            estimates.append(avg_mi)
            avg_estimate.clear()
    lsh.stats()
    return estimates

# Ground truth MI
mi_true = mi_schedule(num_iterations)
start_time = time.time()
data = generate_dataset(num_iterations, mi_true, d, batch_size)
end_time = time.time()
print("Data Built {:2f}".format(end_time - start_time))

estimates = {}
for estimator, mi_params in estimators.items():
    print("Training %s..." % estimator)
    estimates[estimator] = train(device, data, mi_true, mi_params['mi_type'], mi_params['args'])


niter = num_iterations // wsize
mi_true = mi_schedule(niter)

# Smooting span for Exponential Moving Average
EMA_SPAN = 50

# Names specifies the key and ordering for plotting estimators
names = np.sort(list(estimators.keys()))
lnames = list(map(lambda s: s.replace('alpha', '$\\alpha$'), names))

nrows = min(2, len(estimates))
ncols = int(np.ceil(len(estimates) / float(nrows)))
fig, axs = plt.subplots(nrows, ncols, figsize=(2.7 * ncols, 3 * nrows)) 
if len(estimates) == 1:
  axs = [axs]
axs = np.ravel(axs)
  
for i, name in enumerate(names):
  plt.sca(axs[i])
  plt.title(lnames[i])

  # Plot estimated MI and smoothed MI
  mis = estimates[name]  
  mis_smooth = pd.Series(mis).ewm(span=EMA_SPAN).mean()
  p1 = plt.plot(mis, alpha=0.3)[0]
  plt.plot(mis_smooth, c=p1.get_color())

  # Plot true MI and line for log(batch size)
  plt.plot(mi_true, color='k', label='True MI')

  estimator = estimators[name]['mi_type']
  # Add theoretical upper bound lines
  if estimator == MI.INTERPOLATE:
      log_alpha = -np.log( 1+ tf.exp(-estimators[name]['alpha_logit']))
      plt.axhline(1 + np.log(batch_size) - log_alpha, c='k', linestyle='--', label=r'1 + log(K/$\alpha$)' )
  elif estimator == MI.NCE:
      log_alpha = 1.
      plt.axhline(1 + np.log(batch_size) - log_alpha, c='k', linestyle='--', label=r'1 + log(K/$\alpha$)' )
  elif estimator == MI.IS or estimator == MI.LSH:
      log_alpha = 1.
      dbs = estimators[name]['args']['desired_batch_size']
      plt.axhline(1 + np.log(dbs) - log_alpha, c='k', linestyle='--', label=r'1 + log(K/$\alpha$)' )

  #plt.ylim(-1, mi_true.max()+1)
  plt.ylim(-1, 11)
  #plt.xlim(0, num_iterations)
  plt.xlim(0, niter)
  if i == len(estimates) - ncols:
    plt.xlabel('steps')
    plt.ylabel('Mutual information (nats)')
plt.legend(loc='best', fontsize=8, framealpha=0.0)
fig = plt.gcf()
fig.savefig(sys.argv[1])
plt.close()
