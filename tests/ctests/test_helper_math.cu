#include <gtest/gtest.h>
#include <iostream>

#include "float_grad.h"
#include "test_utils.h"
#include "helper_math.h"

TEST(FloatGradHelperMathTest, FloatOperations) {
    float test = fminf(0.3, 0.4);
    EXPECT_FLOAT_EQ(test, 0.3f) << "Expected: 0.3, Got: " << test;

    FloatGrad<float> a(3.0f, 1.0f);
    FloatGrad<float> b(4.0f, 1.0f);

    FloatGrad<float> c = fminf(a, b);
    EXPECT_TRUE(float_eq(c, a));

    c = fminf(FloatGrad<float>(1.0f, 10.0f), a);
    EXPECT_TRUE(float_eq(c, FloatGrad<float>(1.0f, 10.0f)));

    c = fmaxf(-b, a);
    EXPECT_TRUE(float_eq(c, a));

    c = sqrtf(b);
    EXPECT_TRUE(float_eq(c, FloatGrad<float>(2.0f, 0.25f)));

    c = rsqrtf(b);
    EXPECT_TRUE(float_eq(c, FloatGrad<float>(0.5f, -1.0f / 16.0f)));
}

TEST(FloatGradHelperMathTest, MakeFloat2) {
    float3 a_data = make_float3(1.0f, 2.0f, 3.0f);
    float3 a_grad = make_float3(0.1f, 0.2f, 0.3f);

    FloatGrad<float3> a(a_data, a_grad);

    FloatGrad<float2> b = make_float2(a.z);

    EXPECT_TRUE(float_eq(b, FloatGrad<float2>(make_float2(3.0f, 3.0f), 
                                              make_float2(0.3f, 0.3f))));

    FloatGrad<float2> c = make_float2(a);
    EXPECT_TRUE(float_eq(c, FloatGrad<float2>(make_float2(1.0f, 2.0f), 
                                              make_float2(0.1f, 0.2f))));
}

TEST(FloatGradHelperMathTest, MakeFloat3) {
    FloatGrad<float> a(1.0f, 0.1f);
    FloatGrad<float3> b = 3.0f * make_float3(a);
    EXPECT_TRUE(float_eq(b, FloatGrad<float3>(make_float3(3.0f, 3.0f, 3.0f), 
                                              make_float3(0.3f, 0.3f, 0.3f))));

    float2 c = make_float2(2.0f, 4.0f);
    FloatGrad<float3> d = make_float3(c, a);
    EXPECT_TRUE(float_eq(d, FloatGrad<float3>(make_float3(2.0f, 4.0f, 1.0f), 
                                              make_float3(0.0f, 0.0f, 0.1f))));

}

TEST(FloatGradHelperMathTest, MakeFloat4) {
    FloatGrad<float> a(1.0f, 0.1f);
    FloatGrad<float4> b = 3.0f * make_float4(a);
    EXPECT_TRUE(float_eq(b, FloatGrad<float4>(make_float4(3.0f, 3.0f, 3.0f, 3.0f), 
                                              make_float4(0.3f, 0.3f, 0.3f, 0.3f))));

    float3 c = make_float3(2.0f, 4.0f, 0.1f);
    FloatGrad<float4> d = make_float4(c, a);
    EXPECT_TRUE(float_eq(d, FloatGrad<float4>(make_float4(2.0f, 4.0f, 0.1f, 1.0f), 
                                              make_float4(0.0f, 0.0f, 0.0f, 0.1f))));

}

TEST(FloatGradHelperMathTest, ArithmeticOperators) {
    float4 a_data = make_float4(1.0f, 2.0f, 3.0f, 4.0f);
    float4 a_grad = make_float4(0.1f, 0.2f, 0.3f, 0.4f);

    float4 b_data = make_float4(-2.0f, -4.0f, -6.0f, -8.0f);
    float4 b_grad = make_float4(-0.2f, -0.4f, -0.6f, -0.8f);

    FloatGrad<float4> a(a_data, a_grad);
    FloatGrad<float4> b(b_data, b_grad);

    FloatGrad<float4> c = a + b;

    EXPECT_TRUE(float_eq(c, FloatGrad<float4>(make_float4(-1.0f, -2.0f, -3.0f, -4.0f), 
                                              make_float4(-0.1f, -0.2f, -0.3f, -0.4f))));

    c = a - b_grad;

    EXPECT_TRUE(float_eq(c, FloatGrad<float4>(make_float4(1.2f, 2.4f, 3.6f, 4.8f), 
                                              make_float4(0.1f, 0.2f, 0.3f, 0.4f))));

    FloatGrad<float3> d = make_float3(c);
    FloatGrad<float3> e = make_float3(a.w);

    d += e;

    EXPECT_TRUE(float_eq(d, FloatGrad<float3>(make_float3(5.2f, 6.4f, 7.6f), 
                                              make_float3(0.5f, 0.6f, 0.7f))));

    float2 f_data = make_float2(1.0f, 2.0f);
    float2 f_grad = make_float2(0.1f, 0.3f);

    FloatGrad<float2> f(f_data, f_grad);

    float2 g_data = make_float2(3.0f, 4.0f);
    float2 g_grad = make_float2(0.2f, 0.4f);

    FloatGrad<float2> g(g_data, g_grad);

    EXPECT_TRUE(float_eq(f * g_grad, FloatGrad<float2>(make_float2(0.2f, 0.8f), 
                                                       make_float2(0.02f, 0.12f))));

    EXPECT_TRUE(float_eq(f * g, FloatGrad<float2>(make_float2(3.0f, 8.0f), 
                                                  make_float2(0.5f, 2.0f))));

    EXPECT_TRUE(float_eq(g / f, FloatGrad<float2>(make_float2(3.0f, 2.0f), 
                                                  make_float2(-0.1f / 1, -0.4f / 4))));

}

TEST(FloatGradHelperMathTest, MinMaxFunctions) {
    float4 a_data = make_float4(1.0f, 2.0f, 3.0f, 4.0f);
    float4 a_grad = make_float4(0.1f, 0.2f, 0.3f, 0.4f);

    float4 b_data = make_float4(-2.0f, 3.0f, 2.5f, 5.0f);
    float4 b_grad = make_float4(-0.2f, -0.4f, -0.6f, -0.8f);

    FloatGrad<float4> a(a_data, a_grad);
    FloatGrad<float4> b(b_data, b_grad);

    FloatGrad<float4> c = fminf(a, b);

    EXPECT_TRUE(float_eq(c, FloatGrad<float4>(make_float4(-2.0f, 2.0f, 2.5f, 4.0f), 
                                              make_float4(-0.2f, 0.2f, -0.6f, 0.4f))));

    float4 d = make_float4(1.0f, 1.0f, -3.0f, 5.0f);

    FloatGrad<float4> e = fminf(c, d);

    EXPECT_TRUE(float_eq(e, FloatGrad<float4>(make_float4(-2.0f, 1.0f, -3.0f, 4.0f), 
                                              make_float4(-0.2f, 0.0f, -0.0f, 0.4f))));

    FloatGrad<float4> f = fmaxf(a, b);

    EXPECT_TRUE(float_eq(f, FloatGrad<float4>(make_float4(1.0f, 3.0f, 3.0f, 5.0f), 
                                              make_float4(0.1f, -0.4f, 0.3f, -0.8f))));

    FloatGrad<float4> g = fmaxf(d, c);

    EXPECT_TRUE(float_eq(g, FloatGrad<float4>(make_float4(1.0f, 2.0f, 2.5f, 5.0f), 
                                              make_float4(0.0f, 0.2f, -0.6f, 0.0f))));

}

TEST(FloatGradHelperMathTest, LerpFunctions) {
    float4 a_data = make_float4(1.0f, 2.0f, 3.0f, 4.0f);
    float4 a_grad = make_float4(0.1f, 0.2f, 0.3f, 0.4f);

    float4 b_data = make_float4(2.0f, 4.0f, 6.0f, 8.0f);
    float4 b_grad = make_float4(0.2f, 0.4f, 0.6f, 0.8f);

    FloatGrad<float4> a(a_data, a_grad);
    FloatGrad<float4> b(b_data, b_grad);

    EXPECT_TRUE(float_eq(lerp(a.x, 3.0f, 0.5f), 
                         FloatGrad<float>(2.0f, 0.05f)));

    FloatGrad<float> t(0.1f, 1.0f);

    FloatGrad<float4> c = lerp(a, b, t);

    EXPECT_TRUE(float_eq(c, FloatGrad<float4>(make_float4(1.1f, 2.2f, 3.3f, 4.4f), 
                                              make_float4(0.1f + 1.01f, 0.2f + 2.02f, 0.3f + 3.03f, 0.4f + 4.04f))));


}
