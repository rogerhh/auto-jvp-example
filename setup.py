from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os

setup(
    name="auto_jvp_example",
    packages=["auto_jvp_example"],
    ext_modules=[
        CUDAExtension(
            name="auto_jvp_example._C",
            sources=[
                "cuda/test_floatgrad.cu",
                # "matmul_kernel.cu",
                "ext.cu"
            ],
            extra_compile_args={
                "nvcc": ["-O3", "-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)))],
                "cxx": ["-O3"]
            }
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
)

