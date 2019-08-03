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

batch_size = 100
num_iterations = 20000
total_size = num_iterations * batch_size
mi_range = num_iterations // 5
desired_size = 50000
print("lsh size {:d} out of {:d}".format(desired_size, total_size))

d = 20
ed = 32
K = 10
L = 10
device='cuda'
zerot = torch.zeros(1, d).to(device)
bs_zerot = torch.zeros(batch_size, 1).to(device)
bs_onet = torch.ones(batch_size, 1).to(device)

estimators = {
        'LSH IS': dict(mi_type=MI.LSH, args=dict(desired_batch_size=desired_size, k=K, l=L)),
        }

def build(lsh, model, xs, bs=1000):
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
        optimizer.zero_grad()

        # randomly select data from data distribution
        sdx_iter = (batch_idx // mi_range) * mi_range
        sdx_offset = sdx_iter * batch_size
        sdx = torch.from_numpy(np.random.choice(mi_range*batch_size, batch_size, replace=False) + sdx_offset).to(device)

        t = 10 if batch_idx <= 1000 else 100
        if batch_idx % t == 0:
            # Load first section of desired size into lsh hash tables
            lxs = xs[:desired_size, :]
            assert(lxs.size(0) == desired_size)
            build(lsh, model, lxs)

            #lsh.stats()
            # Full - Load All Data
            #build(lsh, model, xs)

        # embed data
        y = F.embedding(sdx, ys).detach()
        ey = model.embed_y(y)

        # for each data sample, query lsh data structure
        id_lists = list() 
        for idx in range(batch_size):
            local_y = sdx[idx].item()
            local_ey = torch.unsqueeze(ey[idx,:], dim=0)
            id_list = lsh.query_remove(local_ey, local_y)
            id_lists.append(id_list)

        # find maximum number of samples
        max_size = functools.reduce(lambda x,y: max(x, len(y)), id_lists, 0)

        # create matrix and pad appropriately
        id_tensors = list()
        for idx, id_list in enumerate(id_lists):
            remainder = max_size - len(id_list)
            local_indices = torch.LongTensor(id_list).to(device)
            new_indices = F.pad(local_indices, pad=(0, remainder), mode='constant', value=xs.size(0))
            id_tensors.append(torch.unsqueeze(new_indices, dim=0))
        indices = torch.cat(id_tensors, dim=0)

        # create mask distinguishing between samples and padding
        mask = 1.0 - torch.eq(indices, xs.size(0)).float()
        mask = torch.cat([bs_onet, mask], dim=1).detach()

        px = torch.unsqueeze(F.embedding(sdx, xs), dim=1)
        nx = F.embedding(indices, zxs, padding_idx=xs.size(0))
        x = torch.cat([px, nx], dim=1).detach()

        mi = model(x, y, mask, args)
        loss = -mi
        loss.backward()
        optimizer.step()

        estimates.append(mi.item())
        if (batch_idx+1) % 100 == 0:
            print('{} {} MI:{}, E_MI: {:.6f}'.format(mi_type.name, batch_idx+1, MI, mi.item()))
            sys.stdout.flush()
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
  #plt.title(lnames[i])
  title = "{:s} - {:d}".format(lnames[i], batch_size)
  plt.title(title)

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
    #plt.xlabel('steps')
    plt.ylabel('Mutual information (nats)')
#plt.legend(loc='best', fontsize=8, framealpha=0.0)
fig = plt.gcf()
fig.savefig(sys.argv[1])
plt.close()
