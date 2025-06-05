import torch
import torch.autograd.forward_ad as fwAD
from auto_jvp_example import float2_dot_jvp

def float2_dot_fn(a, b):
    return torch.sum(a * b, dim=1)

A = torch.randn(100, 2, device='cuda', dtype=torch.float32)
A_tangent = torch.randn(100, 2, device='cuda', dtype=torch.float32)
B = torch.randn(100, 2, device='cuda', dtype=torch.float32)
B_tangent = torch.randn(100, 2, device='cuda', dtype=torch.float32)

with fwAD.dual_level():
    A_dual = fwAD.make_dual(A, A_tangent)
    B_dual = fwAD.make_dual(B, B_tangent)

    C_dual = float2_dot_fn(A_dual, B_dual)

    C = fwAD.unpack_dual(C_dual).primal
    C_tangent = fwAD.unpack_dual(C_dual).tangent

C_ref = torch.sum(A * B, dim=1)
C_tangent_ref = torch.sum(A_tangent * B, dim=1) + torch.sum(A * B_tangent, dim=1)

print(f"data error = {(C - C_ref).abs().max().item()}")
print(f"tangent error = {(C_tangent - C_tangent_ref).abs().max().item()}")

A_jvp = torch.empty(100, 2, 2, device='cuda', dtype=torch.float32)
B_jvp = torch.empty(100, 2, 2, device='cuda', dtype=torch.float32)
A_jvp[:, :, 0] = A
B_jvp[:, :, 0] = B
A_jvp[:, :, 1] = A_tangent
B_jvp[:, :, 1] = B_tangent

C_jvp = float2_dot_jvp(A_jvp, B_jvp)

print(f"Auto JVP data error = {(C_jvp[:, 0] - C).abs().max().item()}")
print(f"Auto JVP tangent error = {(C_jvp[:, 1] - C_tangent).abs().max().item()}")
