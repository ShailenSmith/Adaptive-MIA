import torch
from torch import nn
from torch.nn.parallel import DataParallel, DistributedDataParallel


def unwrapped_parallel_module(module):

    if isinstance(module, (DataParallel, DistributedDataParallel)):
        return module.module
    return module
