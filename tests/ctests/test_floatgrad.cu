#include <gtest/gtest.h>
#include <iostream>

#include "float_grad.h"
#include "test_utils.h"


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

TEST(FloatGradTest, ScalarArrayOperators) {
    float a_data[10] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                            6.0f, 7.0f, 8.0f, 9.0f, 10.0f};
    float a_grad[10] = {1.0f, 0.9f, 0.8f, 0.7f, 0.6f,
                            0.5f, 0.4f, 0.3f, 0.2f, 0.1f};
    float b_data[10] = {2.0f, 4.0f, 6.0f, 8.0f, 10.0f,
                            12.0f, 14.0f, 16.0f, 18.0f, 20.0f};
    float b_grad[10] = {0.5f, 0.4f, 0.3f, 0.2f, 0.1f,
                            0.05f, 0.04f, 0.03f, 0.02f, 0.01f};

    FloatGradArray<float> a(a_data, a_grad);
    FloatGradArray<float> b(b_data, b_grad);

    for(int i = 0; i < 10; i++) {
        FloatGrad<float> ai{a_data[i], a_grad[i]};
        FloatGrad<float> bi{b_data[i], b_grad[i]};

        EXPECT_TRUE(float_eq(ai, a[i]));
        EXPECT_TRUE(float_eq(bi, b[i]));
        EXPECT_TRUE(float_eq(ai + bi, a[i] + b[i]));
        EXPECT_TRUE(float_eq(ai - bi, a[i] - b[i]));
        EXPECT_TRUE(float_eq(ai * bi, a[i] * b[i]));
        EXPECT_TRUE(float_eq(ai / bi, a[i] / b[i]));
        EXPECT_TRUE(float_eq(sqrtf(bi), sqrtf(b[i])));

        EXPECT_TRUE(!((ai < bi) ^ (a[i] < b[i])));
        EXPECT_TRUE(!((ai > bi) ^ (a[i] > b[i])));
        EXPECT_TRUE(!((ai <= bi) ^ (a[i] <= b[i])));
        EXPECT_TRUE(!((ai >= bi) ^ (a[i] >= b[i])));
    }

}

TEST(FloatGradTest, ScalarArrayCompoundOperators) {
    float a_data[10] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                            6.0f, 7.0f, 8.0f, 9.0f, 10.0f};
    float a_grad[10] = {1.0f, 0.9f, 0.8f, 0.7f, 0.6f,
                            0.5f, 0.4f, 0.3f, 0.2f, 0.1f};
    float b_data[10] = {2.0f, 4.0f, 6.0f, 8.0f, 10.0f,
                            12.0f, 14.0f, 16.0f, 18.0f, 20.0f};
    float b_grad[10] = {0.5f, 0.4f, 0.3f, 0.2f, 0.1f,
                            0.05f, 0.04f, 0.03f, 0.02f, 0.01f};
    float c_data[10] = {-5.0f, -4.0f, -3.0f, -2.0f, -1.0f,
                            0.0f, 1.0f, 2.0f, 3.0f, 4.0f};
    float c_grad[10] = {-1.0f, -0.9f, -0.8f, -0.7f, -0.6f,
                            -0.5f, -0.4f, -0.3f, -0.2f, -0.1f};

    float c_copy_data[10];
    float c_copy_grad[10];

    for(int i = 0; i < 10; i++) {
        c_copy_data[i] = c_data[i];
        c_copy_grad[i] = c_grad[i];
    }

    FloatGradArray<float> a(a_data, a_grad);
    FloatGradArray<float> b(b_data, b_grad);
    FloatGradArray<float> c(c_data, c_grad);

    for(int i = 0; i < 10; i++) {
        c[i] += a[i] * b[i];

        EXPECT_TRUE(float_eq(c[i], 
                    FloatGrad<float>(c_copy_data[i] + a_data[i] * b_data[i],
                    c_copy_grad[i] + a_grad[i] * b_data[i] + a_data[i] * b_grad[i])));
    }

}

