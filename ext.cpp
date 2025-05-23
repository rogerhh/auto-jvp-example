#include <torch/extension.h>

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B);
torch::Tensor matmul_cuda_floatgrad(torch::Tensor A, torch::Tensor B);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul_cuda", &matmul_cuda, "Matrix multiplication (CUDA)");
    m.def("matmul_cuda_floatgrad", &matmul_cuda_floatgrad, "Matrix multiplication (CUDA) with gradient propagation");
}

