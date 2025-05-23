import torch
from auto_jvp_example import matmul_cuda, matmul_cuda_floatgrad

A = torch.randn(4, 5, device='cuda', dtype=torch.float32)
B = torch.randn(5, 3, device='cuda', dtype=torch.float32)

C = matmul_cuda(A, B)
print("CUDA matmul:\n", C)

# Compare with PyTorch
C_ref = A @ B
print("PyTorch matmul:\n", C_ref)
print("Max error:", (C - C_ref).abs().max().item())

A_floatgrad = torch.empty(4, 5, 2, device='cuda', dtype=torch.float32)
B_floatgrad = torch.empty(5, 3, 2, device='cuda', dtype=torch.float32)

A_floatgrad[:, :, 0] = A
B_floatgrad[:, :, 0] = B
A_floatgrad[:, :, 1] = 0
B_floatgrad[:, :, 1] = 0

C_floatgrad = matmul_cuda_floatgrad(A_floatgrad, B_floatgrad)

print("CUDA matmul with floatgrad:\n", C_floatgrad)
print("Max error with floatgrad:", (C_floatgrad[:, :, 0] - C_ref).abs().max().item())
