#include <gtest/gtest.h>
#include <iostream>

#include "float_grad.h"
#include "test_utils.h"

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

TEST(FloatGradTest, ConstScalarArrayOperators) {
    const float a_data[10] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                            6.0f, 7.0f, 8.0f, 9.0f, 10.0f};
    const float a_grad[10] = {1.0f, 0.9f, 0.8f, 0.7f, 0.6f,
                            0.5f, 0.4f, 0.3f, 0.2f, 0.1f};
    const float b_data[10] = {2.0f, 4.0f, 6.0f, 8.0f, 10.0f,
                            12.0f, 14.0f, 16.0f, 18.0f, 20.0f};
    const float b_grad[10] = {0.5f, 0.4f, 0.3f, 0.2f, 0.1f,
                            0.05f, 0.04f, 0.03f, 0.02f, 0.01f};

    FloatGradArray<const float> a(a_data, a_grad);
    FloatGradArray<const float> b(b_data, b_grad);

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

TEST(FloatGradTest, ConstScalarArrayCompoundOperators) {
    const float a_data[10] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                            6.0f, 7.0f, 8.0f, 9.0f, 10.0f};
    const float a_grad[10] = {1.0f, 0.9f, 0.8f, 0.7f, 0.6f,
                            0.5f, 0.4f, 0.3f, 0.2f, 0.1f};
    const float b_data[10] = {2.0f, 4.0f, 6.0f, 8.0f, 10.0f,
                            12.0f, 14.0f, 16.0f, 18.0f, 20.0f};
    const float b_grad[10] = {0.5f, 0.4f, 0.3f, 0.2f, 0.1f,
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

    FloatGradArray<const float> a(a_data, a_grad);
    FloatGradArray<const float> b(b_data, b_grad);
    FloatGradArray<float> c(c_data, c_grad);

    for(int i = 0; i < 10; i++) {
        c[i] -= a[i] * b[i];

        EXPECT_TRUE(float_eq(c[i], 
                    FloatGrad<float>(c_copy_data[i] - a_data[i] * b_data[i],
                    c_copy_grad[i] - (a_grad[i] * b_data[i] + a_data[i] * b_grad[i]))));
    }

}

TEST(FloatGradTest, ArrayElementAssignment) {
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
        // volatile auto tmp = a[i] = b[i];
        FloatGradRef<float> aref(&a_data[i], &a_grad[i]);
        FloatGradRef<float> bref(&b_data[i], &b_grad[i]);
        aref = bref;

    }

    for(int i = 0; i < 10; i++) {
        EXPECT_TRUE(float_eq(a[i], b[i]));
    }


    float c_data = -5.0f;
    float c_grad = -1.0f;

    FloatGradRef<float> c(&a_data[0], &a_grad[0]);

    float d_data = 3.0f;
    float d_grad = 0.5f;

    FloatGradRef<float> d(&d_data, &d_grad);

    c = d;

}

TEST(FloatGradTest, ArrayElementAssignmentVec2) {
    float a_data[10] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                        6.0f, 7.0f, 8.0f, 9.0f, 10.0f};
    float a_grad[10] = {1.0f, 0.9f, 0.8f, 0.7f, 0.6f,
                        0.5f, 0.4f, 0.3f, 0.2f, 0.1f};
    float b_data[10] = {2.0f, 4.0f, 6.0f, 8.0f, 10.0f,
                        12.0f, 14.0f, 16.0f, 18.0f, 20.0f};
    float b_grad[10] = {0.5f, 0.4f, 0.3f, 0.2f, 0.1f,
                        0.05f, 0.04f, 0.03f, 0.02f, 0.01f};


    FloatGradArray<float2> a((float2*) a_data, (float2*) a_grad);
    FloatGradArray<float2> b((float2*) b_data, (float2*) b_grad);

    for(int i = 0; i < 5; i++) {
        a[i] = b[i];
    }

    for(int i = 0; i < 5; i++) {
        EXPECT_TRUE(float_eq(a[i], b[i]));
    }


}

TEST(FloatGradTest, ArrayOffset) {
    float a_data[10] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                        6.0f, 7.0f, 8.0f, 9.0f, 10.0f};
    float a_grad[10] = {1.0f, 0.9f, 0.8f, 0.7f, 0.6f,
                        0.5f, 0.4f, 0.3f, 0.2f, 0.1f};

    FloatGradArray<float> a(a_data, a_grad);

    int offset = 5;

    FloatGradArray<float> b = a + 5;

    EXPECT_TRUE(float_eq(b[0], a[5]));

}

