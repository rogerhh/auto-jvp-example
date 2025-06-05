#include <torch/extension.h>
#include <cuda_runtime.h>
#include <tuple>
#include <float_grad.h>
#include <helper_math.h>
#include <iostream>

// CUDA kernel
template <typename FloatType=float>
__global__ void float2_dot_kernel(
        int n,
        const Float2<FloatType>* a,
        const Float2<FloatType>* b,
        FloatType* c) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) return;

    c[idx] = dot(a[idx], b[idx]);
}

// Launcher function (visible to PyTorch)
template <typename FloatType=float, int len=2>
torch::Tensor float_dot_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.dtype() == torch::kFloat32, "A must be float32");
    TORCH_CHECK(B.dtype() == torch::kFloat32, "B must be float32");
    TORCH_CHECK(A.device().is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.device().is_cuda(), "B must be a CUDA tensor");

    // Check A and B have the same dimensions
    TORCH_CHECK(A.dim() == B.dim(), "A and B must have the same number of dimensions");
    for (int i = 0; i < A.dim(); ++i) {
        TORCH_CHECK(A.size(i) == B.size(i), "A and B must have the same dimensions");
    }

    constexpr bool is_jvp = std::is_same<FloatType, FloatGrad>::value;
    int red_dim = A.dim() - (is_jvp ? 2 : 1);
    int jvp_dim = A.dim() - 1;

    // Check A, B's last dimension which is the reduction dimension must be len
    TORCH_CHECK(A.size(red_dim) == len, "A must be a tensor with reduction dimension dim(" + std::to_string(red_dim) + ") of size " + std::to_string(len));
    TORCH_CHECK(B.size(red_dim) == len, "B must be a tensor with reduction dimension dim(" + std::to_string(red_dim) + ") of size " + std::to_string(len));

    if constexpr (is_jvp) {
        // If JVP, the JVP dim must be 2
        TORCH_CHECK(A.size(jvp_dim) == 2, "A must be a tensor with JVP dimension dim(" + std::to_string(jvp_dim) + ") of size 2");
        TORCH_CHECK(B.size(jvp_dim) == 2, "B must be a tensor with JVP dimension dim(" + std::to_string(jvp_dim) + ") of size 2");
    }

    std::vector<int64_t> C_sizes(A.sizes().begin(), A.sizes().begin() + red_dim);

    if constexpr (is_jvp) {
        // If JVP, the last dimension is the JVP dimension
        C_sizes.push_back(2);
    } 

    auto C = torch::empty(
        C_sizes, 
        A.options());

    int num_threads = 1;
    for (int i = 0; i < red_dim; ++i) {
        num_threads *= A.size(i);
    }

    if constexpr (len == 2) {
        // Launch the kernel for float2 dot product
        float2_dot_kernel<FloatType><<<
            (num_threads + 255) / 256, 256>>>(
            num_threads,
            reinterpret_cast<const Float2<FloatType>*>(A.data_ptr<float>()),
            reinterpret_cast<const Float2<FloatType>*>(B.data_ptr<float>()),
            reinterpret_cast<FloatType*>(C.data_ptr<float>()));

    }

    return C;
}


/*
// Launcher function (visible to PyTorch)
template <int len=2>
torch::Tensor float_dot_cuda_jvp(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.dtype() == torch::kFloat32, "A must be float32");
    TORCH_CHECK(B.dtype() == torch::kFloat32, "B must be float32");
    TORCH_CHECK(A.device().is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.device().is_cuda(), "B must be a CUDA tensor");

    // Check A and B have the same dimensions
    TORCH_CHECK(A.dim() == B.dim(), "A and B must have the same number of dimensions");
    for (int i = 0; i < A.dim(); ++i) {
        TORCH_CHECK(A.size(i) == B.size(i), "A and B must have the same dimensions");
    }

    // Check A, B's last dimension which is the reduction dimension must be len
    TORCH_CHECK(A.size(-2) == len, "A must be a tensor with second to last dimension " + std::to_string(len));
    TORCH_CHECK(B.size(-2) == len, "B must be a tensor with second to last dimension " + std::to_string(len));

    // Check A, B's last dimension which is the reduction dimension must be len
    TORCH_CHECK(A.size(-1) == len, "A must be a tensor with last dimension " + std::to_string(len));
    TORCH_CHECK(B.size(-1) == len, "B must be a tensor with last dimension " + std::to_string(len));

    auto C = torch::zeros(
        A.sizes().slice(0, A.dim() - 1), 
        A.options());

    int num_threads = 1;
    for (int i = 0; i < A.dim() - 1; ++i) {
        num_threads *= A.size(i);
    }

    if (len == 2) {
        // Launch the kernel for float2 dot product
        float2_dot_kernel<float><<<
            (num_threads + 255) / 256, 256>>>(
            num_threads,
            reinterpret_cast<const Float2<float>*>(A.data_ptr<float>()),
            reinterpret_cast<const Float2<float>*>(B.data_ptr<float>()),
            A.data_ptr<float>());

    }

    return C;
}
*/
