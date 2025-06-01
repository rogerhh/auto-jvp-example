#include <torch/extension.h>
#include <cuda_runtime.h>
#include <tuple>
#include <float_grad.h>
#include <helper_math.h>

// CUDA kernel
template <typename FloatType=float>
__global__ void matmul_kernel(
        const float* A, 
        const float* B, 
        float* C, 
        int M, int N, int K) {

    const FloatType* A_ptr = reinterpret_cast<const FloatType*>(A);
    const FloatType* B_ptr = reinterpret_cast<const FloatType*>(B);
    FloatType* C_ptr = reinterpret_cast<FloatType*>(C);
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        FloatType sum(0.0f);
        for (int k = 0; k < K; ++k) {
            sum += A_ptr[row * K + k] * B_ptr[k * N + col];
        }
        C_ptr[row * N + col] = sum;
    }
}

// Launcher function (visible to PyTorch)
torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.dtype() == torch::kFloat32, "A must be float32");
    TORCH_CHECK(B.dtype() == torch::kFloat32, "B must be float32");
    TORCH_CHECK(A.device().is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.device().is_cuda(), "B must be a CUDA tensor");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    TORCH_CHECK(B.size(0) == K, "A and B dimensions mismatch");

    auto C = torch::zeros({M, N}, A.options());

    dim3 blockDim(16, 16);
    dim3 gridDim((N + 15) / 16, (M + 15) / 16);

    matmul_kernel<<<gridDim, blockDim>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K
    );

    return C;
}

torch::Tensor matmul_cuda_floatgrad(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.dtype() == torch::kFloat32, "A must be float32");
    TORCH_CHECK(B.dtype() == torch::kFloat32, "B must be float32");
    TORCH_CHECK(A.device().is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.device().is_cuda(), "B must be a CUDA tensor");
    
    // Check A has 3 dimensions and the last dimension is 2
    // Check B has 3 dimensions and the last dimension is 2
    TORCH_CHECK(A.dim() == 3 && A.size(2) == 2, "A must be a 3D tensor with last dimension 2");
    TORCH_CHECK(B.dim() == 3 && B.size(2) == 2, "B must be a 3D tensor with last dimension 2");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    TORCH_CHECK(B.size(0) == K, "A and B dimensions mismatch");

    auto C = torch::zeros({M, N, 2}, A.options());

    dim3 blockDim(16, 16);
    dim3 gridDim((N + 15) / 16, (M + 15) / 16);

    matmul_kernel<FloatGrad><<<gridDim, blockDim>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K
    );

    return C;
}

// If needed: return a tuple
std::tuple<torch::Tensor> matmul_cuda_tuple(torch::Tensor A, torch::Tensor B) {
    auto C = matmul_cuda(A, B);
    return std::make_tuple(C);
}

