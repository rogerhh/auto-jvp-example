#include <gtest/gtest.h>
#include <iostream>

#include "float_grad.h"
#include "test_utils.h"
#include "helper_math.h"

template <typename T>
void print_type() {
    static_assert(always_false<T>::value, "This is a placeholder to ensure the function is not optimized out.");
}

FloatGrad<float4> transformPoint4x4(const FloatGradRef<const float3>& p, const FloatGradArray<const float> matrix)
{
    FloatGrad<float4> transformed = make_float4(
        matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
        matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13], 
        matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
        matrix[3] * p.x + matrix[7] * p.y + matrix[11] * p.z + matrix[15]
    );
    return transformed;
}


TEST(AdvancedTest, TransformPoint4x4) {
    float3 p_data = make_float3(1.0f, 2.0f, 3.0f);
    float3 p_grad = make_float3(0.1f, 0.2f, 0.3f);
    FloatGradRef<const float3> p(&p_data, &p_grad);

    float matrix_data[16];
    float matrix_grad[16];

    for (int i = 0; i < 16; ++i) {
        matrix_data[i] = static_cast<float>(i + 1);
        matrix_grad[i] = static_cast<float>((i + 1) * 0.2);
    }
    FloatGradArray<const float> matrix(matrix_data, matrix_grad);

    float4 transformed_data = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 transformed_grad = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    FloatGradRef<float4> transformed(&transformed_data, &transformed_grad);
    FloatGrad<float4> test(transformed_data, transformed_grad);

    transformed = transformPoint4x4(p, matrix);

}
