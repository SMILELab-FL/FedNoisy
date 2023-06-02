import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


def mixup_data(x, y, alpha=1.0, device=None):
    if alpha > 0:
        lmbd = np.random.beta(alpha, alpha)
    else:
        lmbd = 1.0

    batch_size = x.shape[0]
    if device is not None:
        index = torch.randperm(batch_size).to(device)
    else:
        index = torch.randperm(batch_size)

    mixed_x = lmbd * x + (1 - lmbd) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lmbd
