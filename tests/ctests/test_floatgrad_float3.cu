#include <gtest/gtest.h>
#include <iostream>

#include "float_grad.h"
#include "helper_math.h"
#include "test_utils.h"

TEST(FloatGradFloat3, MakeFloat3) {
    FloatGrad<float> ax(1.0f, 0.1f);
    FloatGrad<float> ay(2.0f, 0.2f);
    FloatGrad<float> az(4.0f, 0.4f);
    FloatGrad<float3> a = make_float3(ax, ay, az);

    EXPECT_TRUE(float_eq(a, FloatGrad<float3>(make_float3(1.0f, 2.0f, 4.0f), 
                                              make_float3(0.1f, 0.2f, 0.4f))));

}

TEST(FloatGradFloat3, VectorElementAccess) {
    float3 a_data = make_float3(1.0f, 2.0f, 5.0f);
    float3 a_grad = make_float3(0.1f, 0.2f, 0.5f);
    float3 b_data = make_float3(3.0f, 4.0f, 6.0f);
    float3 b_grad = make_float3(0.3f, 0.4f, 0.6f);

    FloatGrad<float3> a(a_data, a_grad);
    FloatGrad<float3> b(b_data, b_grad);

    EXPECT_TRUE(float_eq(a.x, FloatGrad<float>(1.0f, 0.1f)));
    EXPECT_TRUE(float_eq(a.y, FloatGrad<float>(2.0f, 0.2f)));
    EXPECT_TRUE(float_eq(a.z, FloatGrad<float>(5.0f, 0.5f)));

    a.z += b.z;

    EXPECT_TRUE(float_eq(a, FloatGrad<float3>(float3{1.0f, 2.0f, 11.0f}, 
                                              float3{0.1f, 0.2f, 1.10f})));
    EXPECT_TRUE(float_eq(b, FloatGrad<float3>(float3{3.0f, 4.0f, 6.0f}, 
                                              float3{0.3f, 0.4f, 0.6f})));

    a.y *= b.y;

    EXPECT_TRUE(float_eq(a, FloatGrad<float3>(float3{1.0f, 8.0f, 11.0f}, 
                                              float3{0.1f, 1.6f, 1.10f})));
    EXPECT_TRUE(float_eq(b, FloatGrad<float3>(float3{3.0f, 4.0f, 6.0f}, 
                                              float3{0.3f, 0.4f, 0.6f})));
}

TEST(FloatGradFloat3, VectorOperators) {
    float3 a_data = make_float3(1.0f, 2.0f, 3.0f);
    float3 a_grad = make_float3(0.1f, 0.2f, 0.3f);
    float3 b_data = make_float3(3.0f, 4.0f, 6.0f);
    float3 b_grad = make_float3(0.3f, 0.4f, 0.6f);

    FloatGrad<float3> a(a_data, a_grad);
    FloatGrad<float3> b(b_data, b_grad);

    auto c = b / a;

    EXPECT_TRUE(float_eq(c, FloatGrad<float3>(make_float3(3.0f, 2.0f, 2.0f), 
                                              make_float3(0.0f, 0.0f, 0.0f))));

}
