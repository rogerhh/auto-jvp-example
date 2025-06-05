import torch
from auto_jvp_example import matmul_cuda, matmul_cuda_jvp

A = torch.randn(4, 5, device='cuda', dtype=torch.float32)
B = torch.randn(5, 3, device='cuda', dtype=torch.float32)

C = matmul_cuda(A, B)
print("CUDA matmul:\n", C)

# Compare with PyTorch
C_ref = A @ B
print("PyTorch matmul:\n", C_ref)
print("Max error:", (C - C_ref).abs().max().item())

A_jvp = torch.empty(4, 5, 2, device='cuda', dtype=torch.float32)
B_jvp = torch.empty(5, 3, 2, device='cuda', dtype=torch.float32)

A_jvp[:, :, 0] = A
B_jvp[:, :, 0] = B
A_jvp[:, :, 1] = 0
B_jvp[:, :, 1] = 0

C_jvp = matmul_cuda_jvp(A_jvp, B_jvp)

print("CUDA matmul with jvp:\n", C_jvp)
print("Max error with jvp:", (C_jvp[:, :, 0] - C_ref).abs().max().item())
