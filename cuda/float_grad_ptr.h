#ifndef FLOAT_GRAD_PTR_H
#define FLOAT_GRAD_PTR_H

#include "float_grad.h"

template <typename FloatType>
struct FloatGradArray {
    FloatType* data_ptr;
    FloatType* grad_ptr;

    __host__ __device__
    FloatGradArray(FloatType* data, FloatType* grad)
        : data_ptr(data), grad_ptr(grad) {}

    __host__ __device__
    FloatGrad<FloatType>& operator[](int index) {
        return FloatGrad<FloatType>(data_ptr[index], grad_ptr[index]);
    }

    __host__ __device__
    const FloatGrad<FloatType>& operator[](int index) const {
        return FloatGrad<FloatType>(data_ptr[index], grad_ptr[index]);
    }
};
