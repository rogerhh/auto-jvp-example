cmake_minimum_required(VERSION 3.18)  # CUDA support is more robust >= 3.18
project(CudaTests LANGUAGES CXX CUDA)

include(FetchContent)

# Fetch GoogleTest
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.zip
)
FetchContent_MakeAvailable(googletest)

# Enable testing
enable_testing()

# CUDA settings (optional)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)  # or Static


# Add test sources and CUDA source
add_executable(auto_jvp_tests
    main.cpp
    basic_tests.cu
    test_floatgrad.cu
    test_floatgrad_float2.cu
    test_floatgrad_float3.cu
    test_floatgrad_float4.cu
    test_helper_math.cu
    test_floatgrad_array.cu
    advanced_tests.cu
)

message(STATUS "${PROJECT_SOURCE_DIR}/../../cuda")
target_include_directories(auto_jvp_tests PRIVATE ${PROJECT_SOURCE_DIR}/../../cuda ${PROJECT_SOURCE_DIR}/../../)
target_link_libraries(auto_jvp_tests PRIVATE gtest gtest_main)
