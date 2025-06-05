#include "float_grad.h"
#include <iostream>

int test_floatgrad() {
    float x[3] = {1, 2, 0};
    float a[3] = {0.1, 0.2, 0.0};

    FloatGradArray<float> x_grad(x, a);

    x_grad[2] = x_grad[0] + x_grad[1];

    std::cout << "x_grad[2]: " << x[2] << " " << a[2] << std::endl;

    x_grad[2] = x_grad[2] + FloatGrad<float>(2);

    std::cout << "x_grad[2]: " << x[2] << " " << a[2] << std::endl;

    x_grad[2] = FloatGrad<float>(3) + x_grad[2];

    std::cout << "x_grad[2]: " << x[2] << " " << a[2] << std::endl;

    return 0;
}
