import math
import numpy as np
import torch
import enum

# x, y - 20-d gaussian variable with mean 0 and correlation p
# mutual information I(x,y) = -d/2 * log(1-p^2)

class MI(enum.IntEnum):
    NCE = 0
    NWJ = 1
    TUBA = 2
    INTERPOLATE = 3
    IS = 4
    LSH = 5

def mutual_information(d, p):
    return -d/2. * math.log(1.-math.pow(p, 2.))

def correlation(d, mi):
    return math.sqrt(1. - math.exp(-2. * mi / d))

def sample(d, p):
    mean = np.zeros(2)
    cov = np.eye(2)
    cov[0, 1] = p
    cov[1, 0] = p
    item = np.random.multivariate_normal(mean, cov, d)
    x, y = zip(*item)
    return np.asarray(x), np.asarray(y)

def sample_single(d, p):
    x = list()
    y = list()
    for idx in range(d):
        a = np.random.normal(0, 1)
        b = np.random.normal(0, 1)
        u = a
        v = p*a - math.sqrt(1.-math.pow(p, 2.)) * b
        x.append(u)
        y.append(v)
    return np.asarray(x), np.asarray(y)

def generate(d, mi, N, transform=None):
    p = correlation(d, mi)
    xt = list()
    yt = list()
    for idx in range(N):
        x, y = sample_single(d, p)
        xt.append(x)
        yt.append(y)
    x_tensor = np.asarray(xt, dtype=np.float32)
    y_tensor = np.asarray(yt, dtype=np.float32)
    if transform is not None:
       y_tensor = np.power(np.matmul(y_tensor, transform), 3)
    return torch.from_numpy(x_tensor), torch.from_numpy(y_tensor)

def mi_schedule(n_iter, value=None):
    """Generate schedule for increasing correlation over time."""
    if value is None:
        #mis = np.round(np.linspace(5.5-1e-9, 0.5, n_iter))*2.0
        mis = np.round(np.linspace(0.5, 5.5-1e-9, n_iter))*2.0
        #mis = np.round(np.linspace(0.5, 2.5-1e-9, n_iter))*2.0
        #mis = np.asarray([2] * (n_iter//2) + [8] * (n_iter//2))
    else:
        mis = np.asarray([value] * n_iter)
    return mis.astype(np.float32)

def generate_dataset(n_iter, schedule, d, batch_size):
    xs = list()
    ys = list()
    #W = np.random.normal(size=(d, d)).astype(np.float32)
    W = None
    for batch_idx, MI in enumerate(schedule):
        x, y = generate(d, MI, batch_size, W)
        xs.append(x)
        ys.append(y)
    return torch.cat(xs, dim=0), torch.cat(ys, dim=0)
