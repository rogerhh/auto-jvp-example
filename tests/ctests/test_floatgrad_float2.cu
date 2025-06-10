#include <gtest/gtest.h>
#include <iostream>

#include "float_grad.h"
#include "helper_math.h"
#include "test_utils.h"

TEST(FloatGradFloat2, MakeFloat2) {
    FloatGrad<float> ax(1.0f, 0.1f);
    FloatGrad<float> ay(2.0f, 0.2f);
    FloatGrad<float2> a = make_float2(ax, ay);

    EXPECT_TRUE(float_eq(a, FloatGrad<float2>(make_float2(1.0f, 2.0f), make_float2(0.1f, 0.2f))));

    FloatGrad<float> bx(3.0f, 0.3f);
    FloatGrad<float2> b = make_float2(bx, 4.0f);

    EXPECT_TRUE(float_eq(b, FloatGrad<float2>(make_float2(3.0f, 4.0f), make_float2(0.3f, 0.0f))));

}

TEST(FloatGradFloat2, VectorElementAccess) {
    float2 a_data = make_float2(1.0f, 2.0f);
    float2 a_grad = make_float2(0.1f, 0.2f);
    float2 b_data = make_float2(3.0f, 4.0f);
    float2 b_grad = make_float2(0.3f, 0.4f);

    FloatGrad<float2> a(a_data, a_grad);
    FloatGrad<float2> b(b_data, b_grad);

    EXPECT_TRUE(float_eq(a.x, FloatGrad<float>(1.0f, 0.1f)));
    EXPECT_TRUE(float_eq(a.y, FloatGrad<float>(2.0f, 0.2f)));

    a.x += b.x;

    EXPECT_TRUE(float_eq(a, FloatGrad<float2>(float2{4.0f, 2.0f}, float2{0.4f, 0.2f})));
    EXPECT_TRUE(float_eq(b, FloatGrad<float2>(float2{3.0f, 4.0f}, float2{0.3f, 0.4f})));

    a.y *= b.y;

    EXPECT_TRUE(float_eq(a, FloatGrad<float2>(float2{4.0f, 8.0f}, float2{0.4f, 1.6f})));
    EXPECT_TRUE(float_eq(b, FloatGrad<float2>(float2{3.0f, 4.0f}, float2{0.3f, 0.4f})));
}

TEST(FloatGradFloat2, VectorOperators) {
    float2 a_data = make_float2(1.0f, 2.0f);
    float2 a_grad = make_float2(0.1f, 0.2f);
    float2 b_data = make_float2(3.0f, 4.0f);
    float2 b_grad = make_float2(0.3f, 0.4f);

    FloatGrad<float2> a(a_data, a_grad);
    FloatGrad<float2> b(b_data, b_grad);

    auto c = a + b;

    EXPECT_TRUE(float_eq(c, FloatGrad<float2>(make_float2(4.0f, 6.0f), 
                                              make_float2(0.4f, 0.6f))));

}
