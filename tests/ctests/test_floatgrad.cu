#include <gtest/gtest.h>
#include <iostream>

#include "float_grad.h"

// Vector comparators only for testing purposes
__host__ __device__
bool operator==(const float2& a, const float2& b) {
    return a.x == b.x && a.y == b.y;
}

__host__ __device__
bool operator==(const float3& a, const float3& b) {
    return a.x == b.x && a.y == b.y && a.z == b.z;
}

__host__ __device__
bool operator==(const float4& a, const float4& b) {
    return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
}


TEST(FloatGradTest, ScalarOperators) {
    FloatGrad<float> a(3.0f, 1.0f);
    FloatGrad<float> b(4.0f, 2.0f);
    
    FloatGrad<float> c = a + b;
    EXPECT_FLOAT_EQ(c.data, 7.0f);
    EXPECT_FLOAT_EQ(c.grad, 3.0f);

    FloatGrad<float> d = a - b;
    EXPECT_FLOAT_EQ(d.data, -1.0f);
    EXPECT_FLOAT_EQ(d.grad, -1.0f);

    FloatGrad<float> e = a * b;
    EXPECT_FLOAT_EQ(e.data, 12.0f);
    EXPECT_FLOAT_EQ(e.grad, 10.0f); // 3 * 4 + 1 * 2

    FloatGrad<float> f = a / b;
    EXPECT_FLOAT_EQ(f.data, 0.75f);
    EXPECT_FLOAT_EQ(f.grad, -0.125f); // (1 * 4 - 3 * 2) / (4 * 4)

    FloatGrad<float> g = sqrtf(b);
    EXPECT_FLOAT_EQ(g.data, 2.0f);
    EXPECT_FLOAT_EQ(g.grad, 0.5f); // 2 * 1/2 * 1/sqrt(4)

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
    EXPECT_FALSE(m.eq(n)); // Should be false since grad is different

}

TEST(FloatGradTest, ScalarCompoundOperators) {
    FloatGrad<float> a(3.0f, 1.0f);
    FloatGrad<float> b(4.0f, 2.0f);

    a += b;
    EXPECT_FLOAT_EQ(a.data, 7.0f);
    EXPECT_FLOAT_EQ(a.grad, 3.0f);

    a *= b;
    EXPECT_FLOAT_EQ(a.data, 28.0f);
    EXPECT_FLOAT_EQ(a.grad, 26.0f); // 7 * 2 + 3 * 4

    a -= b;
    EXPECT_FLOAT_EQ(a.data, 24.0f);
    EXPECT_FLOAT_EQ(a.grad, 24.0f);

    a /= b;
    EXPECT_FLOAT_EQ(a.data, 6.0f);
    EXPECT_FLOAT_EQ(a.grad, 3.0f); // (24 * 4 - 24 * 2) / (4 * 4)
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

        EXPECT_TRUE((ai + bi).eq(a[i] + b[i]));
        EXPECT_TRUE((ai - bi).eq(a[i] - b[i]));
        EXPECT_TRUE((ai * bi).eq(a[i] * b[i]));
        EXPECT_TRUE((ai / bi).eq(a[i] / b[i]));
        EXPECT_TRUE(sqrtf(bi).eq(sqrtf<float>(b[i])));

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

        EXPECT_TRUE(c[i].eq((FloatGrad<float>{c_copy_data[i], c_copy_grad[i]}
                + FloatGrad<float>{a_data[i], a_grad[i]} 
                * FloatGrad<float>{b_data[i], b_grad[i]})));
    }

}

TEST(FloatGradTest, VectorElementAccess) {
    float2 a_data = make_float2(1.0f, 2.0f);
    float2 a_grad = make_float2(0.1f, 0.2f);
    float2 b_data = make_float2(3.0f, 4.0f);
    float2 b_grad = make_float2(0.3f, 0.4f);

    FloatGrad<float2> a(a_data, a_grad);
    FloatGrad<float2> b(b_data, b_grad);

    a.x += b.x;

    EXPECT_TRUE(a.eq(FloatGrad<float2>(float2{4.0f, 2.0f}, float2{0.4f, 0.2f})));
    EXPECT_TRUE(b.eq(FloatGrad<float2>(float2{3.0f, 4.0f}, float2{0.3f, 0.4f})));

    a.y *= b.y;

    EXPECT_TRUE(a.eq(FloatGrad<float2>(float2{4.0f, 8.0f}, float2{0.4f, 1.6f})));
    EXPECT_TRUE(b.eq(FloatGrad<float2>(float2{3.0f, 4.0f}, float2{0.3f, 0.4f})));
}

TEST(FloatGradTest, VectorOperators) {

}

