from . import _C

import torch.nn as nn
import torch

def test_floatgrad():
    return _C.test_floatgrad()

# def matmul_cuda(a, b):
#     return _C.matmul_cuda(a, b)
# 
# def matmul_cuda_jvp(a, b):
#     return _C.matmul_cuda_jvp(a, b)
# 
# def float2_dot(a, b):
#     return _C.float2_dot_cuda(a, b)
# 
# def float2_dot_jvp(a, b):
#     return _C.float2_dot_cuda_jvp(a, b)
