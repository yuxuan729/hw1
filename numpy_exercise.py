import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from torch import utils

# plt.ion()   # interactive mode

def test_net():
    batch, in_n, hid_n, out_n = 64, 1000, 100, 10
    x = np.random.randn(batch, in_n)
    y = np.random.randn(batch, out_n)

    w1 = np.random.randn(in_n, hid_n)
    w2 = np.random.randn(hid_n, out_n)

    learning_rate = 0.00001
    for i in range(5000):
        h = np.dot(x, w1)
        h_relu = np.maximum(h, 0)
        y_pre = np.dot(h_relu, w2)

        loss = np.square(y_pre-y).sum()
        print(loss)

        grad_y_pre = 2.0 * (y_pre-y)
        grad_w2 = np.dot(h_relu.T, grad_y_pre)
        grad_h_relu = np.dot(grad_y_pre, w2.T)
        grad_h = grad_h_relu.copy()
        grad_h[h < 0] = 0
        grad_w1 = np.dot(x.T, grad_h)

        w1 -= grad_w1 * learning_rate
        w2 -= grad_w2 * learning_rate


