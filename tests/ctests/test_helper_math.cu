#include <gtest/gtest.h>
#include <iostream>

#include "float_grad.h"
#include "helper_math.h"

TEST(FloatGradHelperMathTest, VectorConstruct) {
    FloatGrad<float> a(3.0f, 1.0f);
    FloatGrad<float2> a2 = make_float2(a);
}

