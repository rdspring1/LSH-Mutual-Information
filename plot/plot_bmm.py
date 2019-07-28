import sys
import time
import functools

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
mi_range = num_iterations // 1
batch_size = 1
d = 20
ed = 32
K = 12
L = 16 
device='cuda'
zerot = torch.zeros(1, d).to(device)
onet = torch.ones(batch_size, 1).to(device)

estimators = {
        'LSH': dict(mi_type=MI.LSH, args=dict(desired_batch_size=num_iterations, k=K, l=L)),
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
    zxs = torch.cat([xs, zerot], dim=0)

    lsh = LSH(SimHash(ed, K, L), K, L)

    estimates = []
    for batch_idx, MI in enumerate(schedule):
        #t = 10 if batch_idx <= 1000 else 100
        t=100
        if batch_idx % t == 0:
            build(lsh, model, xs)

        optimizer.zero_grad()

        sdx_offset = (batch_idx // mi_range) * mi_range
        sdx = torch.from_numpy(np.random.choice(mi_range, batch_size, replace=False) + sdx_offset).to(device)
        #sdx = torch.from_numpy(np.random.choice(num_iterations, batch_size, replace=False)).to(device)
        y = F.embedding(sdx, ys).detach()
        px = torch.unsqueeze(F.embedding(sdx, xs), dim=1).detach()

        #start = batch_idx * batch_size
        #end = start + batch_size
        #y = ys[start:end]
        #px = torch.unsqueeze(xs[start:end], dim=1)

        ey = model.embed_y(y)
        id_lists = list() 
        for idx in range(batch_size):
            local_ey = torch.unsqueeze(ey[idx,:], dim=0)
            id_list = lsh.query(local_ey)
            id_lists.append(id_list)

        max_size = functools.reduce(lambda x,y: max(x, len(y)), id_lists, 0)

        id_tensors = list()
        for idx, id_list in enumerate(id_lists):
            remainder = max_size - len(id_list)
            local_indices = torch.LongTensor(id_list).to(device)
            new_indices = F.pad(local_indices, pad=(0, remainder), mode='constant', value=xs.size(0))
            id_tensors.append(torch.unsqueeze(new_indices, dim=0))
        indices = torch.cat(id_tensors, dim=0)

        mask = 1.0 - torch.eq(indices, xs.size(0)).float()
        mask = torch.cat([onet, mask], dim=1)

        nx = F.embedding(indices, zxs, padding_idx=xs.size(0))
        x = torch.cat([px, nx], dim=1).detach()

        mi = model(x, y, mask, args)
        loss = -mi
        loss.backward()
        optimizer.step()

        estimates.append(mi.item())
        if (batch_idx+1) % 100 == 0:
            print('{} {} {} MI:{}, E_MI: {:.6f}'.format(mi_type.name, sdx_offset, batch_idx+1, MI, mi.item()))
            sys.stdout.flush()
    lsh.stats()
    return estimates

# Ground truth MI
mi_true = mi_schedule(num_iterations)
start_time = time.time()
#data = generate_dataset(num_iterations, mi_true, d, batch_size)
data = generate_dataset(num_iterations, mi_true, d, 1)
end_time = time.time()
print("Data Built {:2f}".format(end_time - start_time))

estimates = {}
for estimator, mi_params in estimators.items():
    print("Training %s..." % estimator)
    estimates[estimator] = train(device, data, mi_true, mi_params['mi_type'], mi_params['args'])

# Smooting span for Exponential Moving Average
EMA_SPAN = 200

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
  plt.xlim(0, num_iterations)
  if i == len(estimates) - ncols:
    plt.xlabel('steps')
    plt.ylabel('Mutual information (nats)')
plt.legend(loc='best', fontsize=8, framealpha=0.0)
fig = plt.gcf()
fig.savefig(sys.argv[1])
plt.close()
