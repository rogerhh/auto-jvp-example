import torch
import torch.autograd.forward_ad as fwAD
from auto_jvp_example import matmul_cuda_floatgrad

def matmul_fn(a, b):
    return a @ b

A = torch.randn(4, 5, device='cuda', dtype=torch.float32)
A_tangent = torch.randn(4, 5, device='cuda', dtype=torch.float32)
B = torch.randn(5, 3, device='cuda', dtype=torch.float32)
B_tangent = torch.randn(5, 3, device='cuda', dtype=torch.float32)

with fwAD.dual_level():
    A_dual = fwAD.make_dual(A, A_tangent)
    B_dual = fwAD.make_dual(B, B_tangent)

    C_dual = matmul_fn(A_dual, B_dual)


    C = fwAD.unpack_dual(C_dual).primal
    C_tangent = fwAD.unpack_dual(C_dual).tangent

C_tangent_ref = A @ B_tangent + A_tangent @ B

print(f"data error = {(C - A @ B).abs().max().item()}")
print(f"tangent error = {(C_tangent - C_tangent_ref).abs().max().item()}")

A_floatgrad = torch.empty(4, 5, 2, device='cuda', dtype=torch.float32)
B_floatgrad = torch.empty(5, 3, 2, device='cuda', dtype=torch.float32)
A_floatgrad[:, :, 0] = A
B_floatgrad[:, :, 0] = B
A_floatgrad[:, :, 1] = A_tangent
B_floatgrad[:, :, 1] = B_tangent

C_floatgrad = matmul_cuda_floatgrad(A_floatgrad, B_floatgrad)

print(f"Auto JVP data error = {(C_floatgrad[:, :, 0] - C).abs().max().item()}")
print(f"Auto JVP tangent error = {(C_floatgrad[:, :, 1] - C_tangent).abs().max().item()}")
