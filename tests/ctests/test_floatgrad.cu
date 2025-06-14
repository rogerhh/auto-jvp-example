#include <gtest/gtest.h>
#include <iostream>

#include "float_grad.h"
#include "test_utils.h"

TEST(FloatGradTest, RefConstructorAndAssignment) {
    float a_data = 3.0f, a_grad = 1.0f;
    const float b_data = 4.0f;
    const float b_grad = 2.0f;

    FloatGradRef<float> a(&a_data, &a_grad);
    FloatGradRef<const float> b(&b_data, &b_grad);
    FloatGradRef<const float> c(a);
    FloatGradRef<const float> d(b);
    // FloatGradRef<float> e(b);   // This should not compile

    EXPECT_TRUE(float_eq(a, FloatGrad<float>(3.0f, 1.0f)));
    EXPECT_TRUE(float_eq(b, FloatGrad<float>(4.0f, 2.0f)));
    EXPECT_TRUE(float_eq(c, FloatGrad<float>(3.0f, 1.0f)));
    EXPECT_TRUE(float_eq(d, FloatGrad<float>(4.0f, 2.0f)));

    // Ref to Ref
    float f_data = 5.0f, f_grad = 3.0f;
    FloatGradRef<float> f(&f_data, &f_grad);
    a = f;
    EXPECT_TRUE(float_eq(a, FloatGrad<float>(5.0f, 3.0f)));

    // Const Ref to Ref
    a = d;
    EXPECT_TRUE(float_eq(a, FloatGrad<float>(4.0f, 2.0f)));

    // Val to Ref
    float g_data = 6.0f, g_grad = 4.0f;
    FloatGrad<float> g(g_data, g_grad);
    a = g;
    EXPECT_TRUE(float_eq(a, FloatGrad<float>(6.0f, 4.0f)));

    // Const Val to Ref
    const float h_data = 7.0f;
    const float h_grad = 5.0f;
    FloatGrad<const float> h(h_data, h_grad);
    a = h;
    EXPECT_TRUE(float_eq(a, FloatGrad<float>(7.0f, 5.0f)));

}

TEST(FloatGradTest, ValConstructorAndAssignment) {
    float a_data = 3.0f, a_grad = 1.0f;
    const float b_data = 4.0f;
    const float b_grad = 2.0f;

    FloatGrad<float> a(a_data, a_grad);
    FloatGrad<const float> b(b_data, b_grad);
    FloatGrad<const float> c(a);
    FloatGrad<const float> d(b);
    FloatGrad<float> e(b);   // This should not compile

    EXPECT_TRUE(float_eq(a, FloatGrad<float>(3.0f, 1.0f)));
    EXPECT_TRUE(float_eq(b, FloatGrad<float>(4.0f, 2.0f)));
    EXPECT_TRUE(float_eq(c, FloatGrad<float>(3.0f, 1.0f)));
    EXPECT_TRUE(float_eq(d, FloatGrad<float>(4.0f, 2.0f)));
    EXPECT_TRUE(float_eq(e, FloatGrad<float>(4.0f, 2.0f)));

    // Val to Val
    float f_data = 5.0f, f_grad = 3.0f;
    FloatGrad<float> f(f_data, f_grad);
    a = f;
    EXPECT_TRUE(float_eq(a, FloatGrad<float>(5.0f, 3.0f)));

    // Const Val to Val
    a = d;
    EXPECT_TRUE(float_eq(a, FloatGrad<float>(4.0f, 2.0f)));

    // Ref to Val
    float g_data = 6.0f, g_grad = 4.0f;
    FloatGradRef<float> g(&g_data, &g_grad);
    a = g;
    EXPECT_TRUE(float_eq(a, FloatGrad<float>(6.0f, 4.0f)));

    // Ref to Const Val
    FloatGradRef<const float> h(&a_data, &a_grad);
    a = h;
    EXPECT_TRUE(float_eq(a, FloatGrad<float>(3.0f, 1.0f)));
}

TEST(FloatGradTest, ScalarOperators) {
    FloatGrad<float> a(3.0f, 1.0f);
    FloatGrad<float> b(4.0f, 2.0f);
     
    FloatGrad<float> c = a + b;
    EXPECT_TRUE(float_eq(c, FloatGrad<float>(7.0f, 3.0f)));

    FloatGrad<float> d = a - b;
    EXPECT_TRUE(float_eq(d, FloatGrad<float>(-1.0f, -1.0f)));

    FloatGrad<float> e = a * b;
    // grad = 3 * 4 + 1 * 2
    EXPECT_TRUE(float_eq(e, FloatGrad<float>(12.0f, 10.0f))); 

    FloatGrad<float> f = a / b;
    // grad = (3 * 4 - 1 * 2) / (4 * 4)
    EXPECT_TRUE(float_eq(f, FloatGrad<float>(0.75f, -0.125f))) << "Expected: 0.75, -0.125, Got: " << f.data() << ", " << f.grad(); 

    FloatGrad<float> g = sqrtf(b);
    // sqrt(4) = 2, grad = 1/2 * 1/sqrt(4)
    EXPECT_TRUE(float_eq(g, FloatGrad<float>(2.0f, 0.5f))); 

    // Comparators
    EXPECT_TRUE(a < b);
    EXPECT_FALSE(a > b);
    EXPECT_TRUE(a <= b);
    EXPECT_FALSE(a >= b);
    EXPECT_FALSE(a == b);
    EXPECT_TRUE(a != b);

    FloatGrad<float> m(5.0f, 10.0f);
    FloatGrad<float> n(5.0f, 7.0f);

    EXPECT_TRUE(m == n); // Same data, different grad
    EXPECT_FALSE(m != n); // Same data, different grad
    EXPECT_FALSE(float_eq(m, n)); // Should be false since grad is different

}

TEST(FloatGradTest, ScalarCompoundOperators) {
    FloatGrad<float> a(3.0f, 1.0f);
    FloatGrad<float> b(4.0f, 2.0f);

    a += b;
    EXPECT_TRUE(float_eq(a, FloatGrad<float>(7.0f, 3.0f)));

    a *= b;
    EXPECT_TRUE(float_eq(a, FloatGrad<float>(28.0f, 26.0f))); // 7 * 4 + 3 * 2

    a -= b;
    EXPECT_TRUE(float_eq(a, FloatGrad<float>(24.0f, 24.0f)));

    a /= b;
    EXPECT_TRUE(float_eq(a, FloatGrad<float>(6.0f, 3.0f))); // (24 * 4 - 24 * 2) / (4 * 4)
}

