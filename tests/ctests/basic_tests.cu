#include <gtest/gtest.h>
#include <iostream>

#include "float_grad.h"
#include "test_utils.h"

TEST(BasicTests, ScalarConstructor) {
    float a_data = 3.0f, a_grad = 1.0f;
    FloatGrad<float> a(a_data, a_grad);

    float b_data[2] = {4.0f, 5.0f};
    float b_grad[2] = {2.0f, 3.0f};

    FloatGrad<float> b0(b_data[0], b_grad[0]);
    FloatGrad<float> b1(b_data[1], b_grad[1]);
    FloatGrad<float2> b = make_float2(b0, b1);

    EXPECT_TRUE(float_eq(b.x, FloatGrad<float>(4.0f, 2.0f)));
    EXPECT_TRUE(float_eq(b.y, FloatGrad<float>(5.0f, 3.0f)));

}

TEST(BasicTests, ScalarOperators) {
    float a_data = 3.0f, a_grad = 1.0f;
    FloatGrad<float> a(a_data, a_grad);

    const float b_data = 4.0f;
    const float b_grad = 2.0f;
    FloatGradRef<const float> b(&b_data, &b_grad);

    float c_data = 5.0f, c_grad = 3.0f;
    FloatGradRef<float> c(&c_data, &c_grad);

    c = a + b;

    EXPECT_TRUE(float_eq(c, FloatGrad<float>(7.0f, 3.0f)));

    FloatGradRef<float> d(&a_data, &a_grad);
    d += b;

    EXPECT_TRUE(float_eq(d, c));

}

TEST(BasicTests, VectorConstructor) {
    const float2 a_data = make_float2(3.0f, 4.0f);
    const float2 a_grad = make_float2(1.0f, 2.0f);

    FloatGradRef<const float2> a(&a_data, &a_grad);

    float b_data[2] = {4.0f, 5.0f};
    float b_grad[2] = {2.0f, 3.0f};

    FloatGrad<float> b0(b_data[0], b_grad[0]);
    FloatGrad<float> b1(b_data[1], b_grad[1]);
    FloatGrad<float2> b = make_float2(b0, b1);

    EXPECT_TRUE(float_eq(b.x, FloatGrad<float>(4.0f, 2.0f)));
    EXPECT_TRUE(float_eq(b.y, FloatGrad<float>(5.0f, 3.0f)));

    float4 c_data = make_float4(6.0f, 7.0f, 8.0f, 9.0f);
    float4 c_grad = make_float4(4.0f, 5.0f, 6.0f, 7.0f);
    FloatGrad<float4> c(c_data, c_grad);
    FloatGrad<float4> d(c);

}

TEST(BasicTests, VectorRefConstructor) {
    float a_data = 3.0f, a_grad = 1.0f;
    FloatGrad<float> a(0.0f, 0.0f);

    a = a_data;

    EXPECT_TRUE(float_eq(a, FloatGrad<float>(3.0f, 0.0f)));


}

TEST(BasicTests, ArrayCompoundOperator) {
    float a_data[2] = {1.0f, 2.0f};
    float a_grad[2] = {0.1f, 0.2f};

    float b_data = 5.0f;
    float b_grad = 0.5f;
    FloatGradArray<float> a(a_data, a_grad);
    FloatGradRef<float> b(a_data, a_grad);
    FloatGradRef<float> c = a[0];

    c += b;
    a[0] += b;
}

