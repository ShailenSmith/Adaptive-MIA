# from https://github.com/lephuoccat/Fisher-Information-NAS/blob/main/fisher_distance_cifar.py
# work in progress

import os
import argparse

import numpy as np
from copy import deepcopy

import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.nn import functional as F
from torch.autograd import Variable
import torch.utils.data


def diag_fisher(model, loader, device): 
    squared_grads = {} # why call these precision matrices?
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    for n, p in deepcopy(params).items():
        p.data.zero_()
        squared_grads[n] = variable(p.data)

    model.eval()
    error = nn.CrossEntropyLoss()
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        model.zero_grad()
        output = model(inputs)
        # print(output.shape)

        loss = error(output, labels)
        loss.backward()

        for n, p in model.named_parameters():
            squared_grads[n].data += (p.grad.data ** 2).mean(0) # average over the minibatch
        
    for n, p in squared_grads.items(): # average over all minibatches
        squared_grads[n].data /= len(loader)

    conv_squared_grads = {}
    for n, p in squared_grads.items():
        if 'conv' in n:
            conv_squared_grads[n] = p
    
    return conv_squared_grads


def variable(t: torch.Tensor, use_cuda=True, **kwargs):
    if torch.cuda.is_available() and use_cuda:
        t = t.cuda()
    return Variable(t, **kwargs)