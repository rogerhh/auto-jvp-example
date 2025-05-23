from . import _C

import torch.nn as nn
import torch

def matmul_cuda(a, b):
    return _C.matmul_cuda(a, b)

def matmul_cuda_floatgrad(a, b):
    return _C.matmul_cuda_floatgrad(a, b)

