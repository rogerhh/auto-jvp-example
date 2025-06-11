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

    FloatGrad<float> d(3.4f, 0.25f);
    EXPECT_TRUE(float_eq(floorf(d), FloatGrad<float>(3.0f, 0.0f)));
    EXPECT_TRUE(float_eq(ceilf(d), FloatGrad<float>(4.0f, 0.0f)));
    EXPECT_TRUE(float_eq(roundf(d), FloatGrad<float>(3.0f, 0.0f)));
    EXPECT_TRUE(float_eq(truncf(d), FloatGrad<float>(3.0f, 0.0f)));
    EXPECT_TRUE(float_eq(fmodf(d, 2.0f), FloatGrad<float>(1.4f, 0.25f)));
    EXPECT_TRUE(float_eq(fmodf(d, 1.1f), FloatGrad<float>(0.1f, 0.25f)));
    EXPECT_TRUE(float_eq(fabs(d), FloatGrad<float>(3.4f, 0.25f)));

    FloatGrad<float> e(-5.6f, 1.0f);
    EXPECT_TRUE(float_eq(floorf(e), FloatGrad<float>(-6.0f, 0.0f)));
    EXPECT_TRUE(float_eq(ceilf(e), FloatGrad<float>(-5.0f, 0.0f)));
    EXPECT_TRUE(float_eq(roundf(e), FloatGrad<float>(-6.0f, 0.0f)));
    EXPECT_TRUE(float_eq(truncf(e), FloatGrad<float>(-5.0f, 0.0f)));
    EXPECT_TRUE(float_eq(fmodf(e, 2.0f), FloatGrad<float>(-1.6f, 1.0f)));
    EXPECT_TRUE(float_eq(fabs(e), FloatGrad<float>(5.6f, -1.0f)));

    FloatGrad<float> f(2.0f, 1.0f);
    EXPECT_TRUE(float_eq(fmodf(4.4f, f), FloatGrad<float>(0.4f, -2.0f)));

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

    EXPECT_TRUE(float_eq(-b, FloatGrad<float4>(make_float4(2.0f, 4.0f, 6.0f, 8.0f),
                                               make_float4(0.2f, 0.4f, 0.6f, 0.8f))));

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
                                              make_float4(1.11f, 2.22f, 3.33f, 4.44f))));
}

TEST(FloatGradHelperMathTest, ClampFunctions) {
    float4 a_data = make_float4(11.0f, 2.5f, -1.0f, 4.0f);
    float4 a_grad = make_float4(-0.1f, -0.2f, 0.3f, -0.4f);

    float4 b_data = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 b_grad = make_float4(0.2f, 0.4f, 0.6f, 0.8f);

    float4 c_data = make_float4(10.0f, 10.0f, 10.0f, 10.0f);
    float4 c_grad = make_float4(0.1f, 0.2f, 0.3f, 0.4f);

    FloatGrad<float4> a(a_data, a_grad);
    FloatGrad<float4> b(b_data, b_grad);
    FloatGrad<float4> c(c_data, c_grad);
    FloatGrad<float4> d = clamp(a, b, c);

    EXPECT_TRUE(float_eq(d, FloatGrad<float4>(make_float4(10.0f, 2.5f, 0.0f, 4.0f), 
                                              make_float4(0.1f, -0.2f, 0.6f, -0.4f))));

}

TEST(FloatGradHelperMathTest, DotFunctions) {
    float4 a_data = make_float4(11.0f, 2.5f, -1.0f, 4.0f);
    float4 a_grad = make_float4(-0.1f, -0.2f, 0.3f, -0.4f);

    float4 b_data = make_float4(1.0f, 2.0f, 3.0f, 4.0f);
    float4 b_grad = make_float4(0.2f, 0.4f, 0.6f, 0.8f);

    FloatGrad<float4> a(a_data, a_grad);
    FloatGrad<float4> b(b_data, b_grad);

    auto c = dot(a, b);

    EXPECT_TRUE(float_eq(c, FloatGrad<float>(29.0f, 4.6f)));

}

TEST(FloatGradHelperMathTest, LengthFunctions) {
    float3 a_data = make_float3(1.0f, 2.0f, 3.0f);
    float3 a_grad = make_float3(0.1f, 0.2f, 0.3f);

    FloatGrad<float3> a(a_data, a_grad);

    auto c = length(a);

    EXPECT_TRUE(float_eq(c, FloatGrad<float>(sqrtf(14.0f), 0.5f * 2.8f / sqrtf(14.0f))));

    auto d = normalize(a);

    float a_norm_sq = dot(a_data, a_data);
    float rsqrt_a_norm_sq = rsqrtf(a_norm_sq);
    float3 d_grad_ref = rsqrt_a_norm_sq * a_grad 
                        - rsqrt_a_norm_sq / a_norm_sq * dot(a_data, a_grad) * a_data;

    EXPECT_TRUE(float_eq(d, FloatGrad<float3>(normalize(a_data), d_grad_ref)));

}

TEST(FloatGradHelperMathTest, FloorFunctions) {
    float3 a_data = make_float3(1.1f, 2.9f, 3.5f);
    float3 a_grad = make_float3(0.1f, 0.2f, 0.3f);

    FloatGrad<float3> a(a_data, a_grad);

    FloatGrad<float3> b = floorf(a);

    EXPECT_TRUE(float_eq(b, FloatGrad<float3>(make_float3(1.0f, 2.0f, 3.0f), 
                                              make_float3(0.0f, 0.0f, 0.0f))));

}

TEST(FloatGradHelperMathTest, FracFunctions) {
    float3 a_data = make_float3(1.1f, 2.9f, 3.5f);
    float3 a_grad = make_float3(0.1f, 0.2f, 0.3f);

    FloatGrad<float3> a(a_data, a_grad);

    FloatGrad<float3> b = fracf(a);

    EXPECT_TRUE(float_eq(b, FloatGrad<float3>(make_float3(0.1f, 0.9f, 0.5f), 
                                              make_float3(0.1f, 0.2f, 0.3f))));

}

TEST(FloatGradHelperMathTest, FmodFunctions) {
    float3 a_data = make_float3(1.1f, 2.9f, 3.5f);
    float3 a_grad = make_float3(0.1f, 0.2f, 0.3f);
    float3 b_data = make_float3(0.3f, 0.2f, 0.7f);
    float3 b_grad = make_float3(0.2f, 0.4f, 0.5f);

    FloatGrad<float3> a(a_data, a_grad);
    FloatGrad<float3> b(b_data, b_grad);

    FloatGrad<float3> c = fmodf(a, b);

    EXPECT_TRUE(float_eq(c, FloatGrad<float3>(make_float3(0.2f, 0.1f, 0.0f), 
                                              make_float3(-0.5f, -5.4f, -2.2f))));

}

TEST(FloatGradHelperMathTest, FabsFunctions) {
    float3 a_data = make_float3(1.1f, -2.9f, 3.5f);
    float3 a_grad = make_float3(0.1f, 0.2f, 0.3f);

    FloatGrad<float3> a(a_data, a_grad);
    FloatGrad<float3> b = fabs(a);

    EXPECT_TRUE(float_eq(b, FloatGrad<float3>(make_float3(1.1f, 2.9f, 3.5f), 
                                              make_float3(0.1f, -0.2f, 0.3f))));

}

TEST(FloatGradHelperMathTest, ReflectAutodiff) {
    float3 a_data = make_float3(1.1f, -2.9f, 3.5f);
    float3 a_grad = make_float3(0.1f, 0.2f, 0.3f);
    float3 b_data = normalize(make_float3(0.3f, 0.2f, 0.7f));
    float3 b_grad = make_float3(0.2f, 0.4f, 0.5f);

    FloatGrad<float3> a(a_data, a_grad);
    FloatGrad<float3> b(b_data, b_grad);

    float3 c_data_ref = reflect(a_data, b_data);
    FloatGrad<float3> c = reflect(a, b);

    EXPECT_FLOAT_EQ(c_data_ref.x, c.x.data());
    EXPECT_FLOAT_EQ(c_data_ref.y, c.y.data());
    EXPECT_FLOAT_EQ(c_data_ref.z, c.z.data());

    float* x_ptr[6] = {&a_data.x, &a_data.y, &a_data.z, 
                       &b_data.x, &b_data.y, &b_data.z};
    float* g_ptr[6] = {&a_grad.x, &a_grad.y, &a_grad.z,
                       &b_grad.x, &b_grad.y, &b_grad.z};

    float3 c_grad_ref = make_float3(0.0f, 0.0f, 0.0f);

    float eps = 1e-4;   // Need to choose large enough epsilon to avoid numerical issues
    for(int i = 0; i < 6; i++) {
        float x_data_backup = *x_ptr[i];
        *x_ptr[i] += eps;
        float3 c_plus = reflect(a_data, b_data);
        *x_ptr[i] = x_data_backup - eps;
        float3 c_minus = reflect(a_data, b_data);
        *x_ptr[i] = x_data_backup;

        c_grad_ref += (c_plus - c_minus) / (2 * eps) * (*g_ptr[i]);
    }

    EXPECT_TRUE(float_eq(c.grad(), c_grad_ref, 1e-3)) << "Expected: " << c_grad_ref.x << ", " 
                                                      << c_grad_ref.y << ", " << c_grad_ref.z 
                                                      << " Got: " << c.grad().x << ", " 
                                                      << c.grad().y << ", " << c.grad().z;

}

TEST(FloatGradHelperMathTest, CrossAutodiff) {
    float3 a_data = make_float3(1.1f, -2.9f, 3.5f);
    float3 a_grad = make_float3(0.1f, 0.2f, 0.3f);
    float3 b_data = normalize(make_float3(0.3f, 0.2f, 0.7f));
    float3 b_grad = make_float3(0.2f, 0.4f, 0.5f);

    FloatGrad<float3> a(a_data, a_grad);
    FloatGrad<float3> b(b_data, b_grad);

    float3 c_data_ref = cross(a_data, b_data);
    FloatGrad<float3> c = cross(a, b);

    EXPECT_FLOAT_EQ(c_data_ref.x, c.x.data());
    EXPECT_FLOAT_EQ(c_data_ref.y, c.y.data());
    EXPECT_FLOAT_EQ(c_data_ref.z, c.z.data());

    float* x_ptr[6] = {&a_data.x, &a_data.y, &a_data.z, 
                       &b_data.x, &b_data.y, &b_data.z};
    float* g_ptr[6] = {&a_grad.x, &a_grad.y, &a_grad.z,
                       &b_grad.x, &b_grad.y, &b_grad.z};

    float3 c_grad_ref = make_float3(0.0f, 0.0f, 0.0f);

    float eps = 1e-4;
    for(int i = 0; i < 6; i++) {
        float x_data_backup = *x_ptr[i];
        *x_ptr[i] += eps;
        float3 c_plus = cross(a_data, b_data);
        *x_ptr[i] = x_data_backup - eps;
        float3 c_minus = cross(a_data, b_data);
        *x_ptr[i] = x_data_backup;

        c_grad_ref += (c_plus - c_minus) / (2 * eps) * (*g_ptr[i]);
    }

    EXPECT_TRUE(float_eq(c.grad(), c_grad_ref, 1e-3)) << "Expected: " << c_grad_ref.x << ", " 
                                                      << c_grad_ref.y << ", " << c_grad_ref.z 
                                                      << " Got: " << c.grad().x << ", " 
                                                      << c.grad().y << ", " << c.grad().z;

}

TEST(FloatGradHelperMathTest, SmoothStepAutodiff) {
    float3 a_data = make_float3(1.1f, -2.9f, 3.5f);
    float3 a_grad = make_float3(0.1f, 0.2f, 0.3f);
    float3 b_data = normalize(make_float3(0.3f, 0.2f, 0.7f));
    float3 b_grad = make_float3(0.2f, 0.4f, 0.5f);
    float3 d_data = make_float3(1.3f, -2.0f, 3.65f);
    float3 d_grad = make_float3(0.45f, 0.4f, -0.5f);

    FloatGrad<float3> a(a_data, a_grad);
    FloatGrad<float3> b(b_data, b_grad);
    FloatGrad<float3> d(d_data, d_grad);

    float3 c_data_ref = smoothstep(a_data, b_data, d_data);
    FloatGrad<float3> c = smoothstep(a, b, d);

    EXPECT_FLOAT_EQ(c_data_ref.x, c.x.data());
    EXPECT_FLOAT_EQ(c_data_ref.y, c.y.data());
    EXPECT_FLOAT_EQ(c_data_ref.z, c.z.data());

    float* x_ptr[9] = {&a_data.x, &a_data.y, &a_data.z, 
                       &b_data.x, &b_data.y, &b_data.z,
                       &d_data.x, &d_data.y, &d_data.z};
    float* g_ptr[9] = {&a_grad.x, &a_grad.y, &a_grad.z,
                       &b_grad.x, &b_grad.y, &b_grad.z,
                       &d_grad.x, &d_grad.y, &d_grad.z};

    float3 c_grad_ref = make_float3(0.0f, 0.0f, 0.0f);

    float eps = 1e-4;
    for(int i = 0; i < 9; i++) {
        float x_data_backup = *x_ptr[i];
        *x_ptr[i] += eps;
        float3 c_plus = smoothstep(a_data, b_data, d_data);
        *x_ptr[i] = x_data_backup - eps;
        float3 c_minus = smoothstep(a_data, b_data, d_data);
        *x_ptr[i] = x_data_backup;

        c_grad_ref += (c_plus - c_minus) / (2 * eps) * (*g_ptr[i]);
    }

    EXPECT_TRUE(float_eq(c.grad(), c_grad_ref, 1e-3)) << "Expected: " << c_grad_ref.x << ", " 
                                                      << c_grad_ref.y << ", " << c_grad_ref.z 
                                                      << " Got: " << c.grad().x << ", " 
                                                      << c.grad().y << ", " << c.grad().z;

}
