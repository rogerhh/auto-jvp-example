import torch
from auto_jvp_example import float2_dot, float2_dot_jvp

A = torch.randn(100, 2, device='cuda', dtype=torch.float32)
B = torch.randn(100, 2, device='cuda', dtype=torch.float32)

C = float2_dot(A, B)
print("CUDA float2 dot:\n", C)

# Compare with PyTorch
C_ref = torch.sum(A * B, dim=1)
import code; code.interact(local=locals()) 
print("PyTorch float2 dot:\n", C_ref)
print("Max error:", (C - C_ref).abs().max().item())

A_jvp = torch.empty(100, 2, 2, device='cuda', dtype=torch.float32)
B_jvp = torch.empty(100, 2, 2, device='cuda', dtype=torch.float32)

A_jvp[:, :, 0] = A
B_jvp[:, :, 0] = B
A_jvp[:, :, 1] = 0
B_jvp[:, :, 1] = 0

C_jvp = float2_dot_jvp(A_jvp, B_jvp)

print("CUDA float2 dot with jvp:\n", C_jvp)
print("Max error with jvp:", (C_jvp[:, 0] - C_ref).abs().max().item())
