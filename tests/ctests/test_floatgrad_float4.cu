#include <gtest/gtest.h>
#include <iostream>

#include "float_grad.h"
#include "helper_math.h"
#include "test_utils.h"

TEST(FloatGradFloat4, MakeFloat4) {
    FloatGrad<float> ax(1.0f, 0.1f);
    FloatGrad<float> ay(2.0f, 0.2f);
    FloatGrad<float> az(4.0f, 0.4f);
    FloatGrad<float> aw(6.0f, 0.6f);
    FloatGrad<float4> a = make_float4(ax, ay, az, aw);

    EXPECT_TRUE(float_eq(a, FloatGrad<float4>(make_float4(1.0f, 2.0f, 4.0f, 6.0f), 
                                              make_float4(0.1f, 0.2f, 0.4f, 0.6f))));

}

TEST(FloatGradFloat4, Float4Ctor) {
    FloatGrad<float> ax(1.0f, 0.1f);
    float ay = 2.0f;
    FloatGrad<float> az(4.0f, 0.4f);
    float aw = 6.0f;
    FloatGrad<float4> a(ax, ay, az, aw);

    EXPECT_TRUE(float_eq(a, FloatGrad<float4>(make_float4(1.0f, 2.0f, 4.0f, 6.0f), 
                                              make_float4(0.1f, 0.0f, 0.4f, 0.0f))));

}

TEST(FloatGradFloat4, VectorElementAccess) {
    float4 a_data = make_float4(1.0f, 2.0f, 5.0f, -4.0f);
    float4 a_grad = make_float4(0.1f, 0.2f, 0.5f, -0.4f);
    float4 b_data = make_float4(3.0f, 4.0f, 6.0f, 8.0f);
    float4 b_grad = make_float4(0.3f, 0.4f, 0.6f, 0.8f);

    FloatGrad<float4> a(a_data, a_grad);
    FloatGrad<float4> b(b_data, b_grad);

    EXPECT_TRUE(float_eq(a.x, FloatGrad<float>(1.0f, 0.1f)));
    EXPECT_TRUE(float_eq(a.y, FloatGrad<float>(2.0f, 0.2f)));
    EXPECT_TRUE(float_eq(a.z, FloatGrad<float>(5.0f, 0.5f)));
    EXPECT_TRUE(float_eq(a.w, FloatGrad<float>(-4.0f, -0.4f)));

    a.w -= b.w;

    EXPECT_TRUE(float_eq(a, FloatGrad<float4>(float4{1.0f, 2.0f, 5.0f, -12.0f}, 
                                              float4{0.1f, 0.2f, 0.5f, -1.20f})));
    EXPECT_TRUE(float_eq(b, FloatGrad<float4>(float4{3.0f, 4.0f, 6.0f, 8.0f}, 
                                              float4{0.3f, 0.4f, 0.6f, 0.8f})));

    a.y *= b.y;

    EXPECT_TRUE(float_eq(a, FloatGrad<float4>(float4{1.0f, 8.0f, 5.0f, -12.0f}, 
                                              float4{0.1f, 1.6f, 0.5f, -1.20f})));
    EXPECT_TRUE(float_eq(b, FloatGrad<float4>(float4{3.0f, 4.0f, 6.0f, 8.0f}, 
                                              float4{0.3f, 0.4f, 0.6f, 0.8f})));

    FloatGrad<const float4> c(a_data, a_grad);
    FloatGradRef<const float4> d(&b_data, &b_grad);

    EXPECT_TRUE(float_eq(c.x, FloatGrad<float>(1.0f, 0.1f)));
    EXPECT_TRUE(float_eq(c.y, FloatGrad<float>(2.0f, 0.2f)));
    EXPECT_TRUE(float_eq(c.z, FloatGrad<float>(5.0f, 0.5f)));
    EXPECT_TRUE(float_eq(c.w, FloatGrad<float>(-4.0f, -0.4f)));
    EXPECT_TRUE(float_eq(d.x, FloatGrad<float>(3.0f, 0.3f)));
    EXPECT_TRUE(float_eq(d.y, FloatGrad<float>(4.0f, 0.4f)));
    EXPECT_TRUE(float_eq(d.z, FloatGrad<float>(6.0f, 0.6f)));
    EXPECT_TRUE(float_eq(d.w, FloatGrad<float>(8.0f, 0.8f)));
}

TEST(FloatGradFloat4, VectorOperators) {
    float4 a_data = make_float4(1.0f, 2.0f, 3.0f, 2.0f);
    float4 a_grad = make_float4(0.1f, 0.2f, 0.3f, 0.2f);
    float4 b_data = make_float4(3.0f, 4.0f, 6.0f, 8.0f);
    float4 b_grad = make_float4(0.3f, 0.4f, 0.6f, 0.8f);

    FloatGrad<float4> a(a_data, a_grad);
    FloatGrad<float4> b(b_data, b_grad);

    auto c = b - a;

    EXPECT_TRUE(float_eq(c, FloatGrad<float4>(make_float4(2.0f, 2.0f, 3.0f, 6.0f), 
                                              make_float4(0.2f, 0.2f, 0.3f, 0.6f))));

}
