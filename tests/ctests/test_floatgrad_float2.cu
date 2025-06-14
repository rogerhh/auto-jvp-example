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

    FloatGrad<const float2> c(a_data, a_grad);
    FloatGradRef<const float2> d(&b_data, &b_grad);

    EXPECT_TRUE(float_eq(c.x, FloatGrad<float>(1.0f, 0.1f)));
    EXPECT_TRUE(float_eq(c.y, FloatGrad<float>(2.0f, 0.2f)));
    EXPECT_TRUE(float_eq(d.x, FloatGrad<float>(3.0f, 0.3f)));
    EXPECT_TRUE(float_eq(d.y, FloatGrad<float>(4.0f, 0.4f)));
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

TEST(FloatGradFloat2, VectorArray) {
    float2 a_data[10];
    float2 a_grad[10];
    float2 b_data[10];
    float2 b_grad[10];

    FloatGradArray<float2> a(a_data, a_grad);
    FloatGradArray<float2> b(b_data, b_grad);

    for (int i = 0; i < 10; ++i) {
        a_data[i] = make_float2(float(i + 1), float(i + 2));
        a_grad[i] = make_float2(float(0.1f * (i + 1)), float(0.2f * (i + 1)));
        b_data[i] = make_float2(float(i + 3), float(i + 4));
        b_grad[i] = make_float2(float(0.3f * (i + 1)), float(0.4f * (i + 1)));
    }

    const float2* c_data = a_data;
    const float2* c_grad = a_grad;
    const float2* d_data = b_data;
    const float2* d_grad = b_grad;

    FloatGradArray<const float2> c(c_data, c_grad);
    FloatGradArray<const float2> d(d_data, d_grad);

    int i = 5;
    FloatGrad<float2> e = c[i] + d[i];
    EXPECT_TRUE(float_eq(e, FloatGrad<float2>(make_float2(14.0f, 16.0f), 
                                              make_float2(2.4f, 3.6f))));

}
