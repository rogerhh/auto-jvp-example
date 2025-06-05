#include <torch/extension.h>

// #include "float_grad.h"
// #include "vector_kernel_impl.h"

int test_floatgrad();

// torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B);
// torch::Tensor matmul_cuda_jvp(torch::Tensor A, torch::Tensor B);
// template <typename FloatTpye, int len>
// torch::Tensor float_dot_cuda(torch::Tensor A, torch::Tensor B);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("test_floatgrad", &test_floatgrad, "Test FloatGrad functionality");
    // m.def("matmul_cuda", &matmul_cuda, "Matrix multiplication (CUDA)");
    // m.def("matmul_cuda_jvp", &matmul_cuda_jvp, "Matrix multiplication (CUDA) with gradient propagation");
    // m.def("float2_dot_cuda", &float_dot_cuda<float, 2>, "Float2 dot product (CUDA)");
    // m.def("float2_dot_cuda_jvp", &float_dot_cuda<FloatGrad, 2>, "Float2 dot product with JVP (CUDA)");
}

