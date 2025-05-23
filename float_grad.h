#ifndef FLOAT_GRAD_H
#define FLOAT_GRAD_H

struct FloatGrad {
    float data, grad;

    __host__ __device__ FloatGrad() : data(0.0f), grad(0.0f) {}
    __host__ __device__ FloatGrad(float data) : data(data), grad(0.0f) {}
    __host__ __device__ FloatGrad(float data, float grad) : data(data), grad(grad) {}

    __host__ __device__
    inline FloatGrad& operator=(const FloatGrad& other) {
        data = other.data;
        grad = other.grad;
        return *this;
    }

    __host__ __device__
    inline FloatGrad& operator=(const float& scalar) {
        data = scalar;
        grad = 0.0f;
        return *this;
    }

    __host__ __device__
    inline FloatGrad& operator+=(const float& scalar) {
        data += scalar;
        return *this;
    }

    __host__ __device__
    inline FloatGrad& operator-=(const float& scalar) {
        data -= scalar;
        return *this;
    }

    __host__ __device__
    inline FloatGrad& operator*=(const float& scalar) {
        grad *= scalar;
        data *= scalar;
        return *this;
    }
    
    __host__ __device__
    inline FloatGrad& operator/=(const float& scalar) {
        grad /= scalar;
        data /= scalar;
        return *this;
    }

    __host__ __device__
    inline FloatGrad operator+(const float& scalar) const {
        return FloatGrad(data + scalar, grad);
    }

    __host__ __device__
    inline FloatGrad operator-(const float& scalar) const {
        return FloatGrad(data - scalar, grad);
    }

    __host__ __device__
    inline FloatGrad operator*(const float& scalar) const {
        return FloatGrad(data * scalar, grad * scalar);
    }

    __host__ __device__
    inline FloatGrad operator/(const float& scalar) const {
        return FloatGrad(data / scalar, grad / scalar);
    }

    __host__ __device__ 
    inline FloatGrad& operator+=(const FloatGrad& other) {
        data += other.data;
        grad += other.grad;
        return *this;
    }

    __host__ __device__
    inline FloatGrad& operator-=(const FloatGrad& other) {
        data -= other.data;
        grad -= other.grad;
        return *this;
    }

    __host__ __device__
    inline FloatGrad& operator*=(const FloatGrad& other) {
        grad = grad * other.data + data * other.grad;
        data *= other.data;
        return *this;
    }

    __host__ __device__
    inline FloatGrad& operator/=(const FloatGrad& other) {
        grad = (grad * other.data - data * other.grad) / (other.data * other.data);
        data /= other.data;
        return *this;
    }

    __host__ __device__ 
    inline FloatGrad operator+(const FloatGrad& other) const {
        return FloatGrad(data + other.data, grad + other.grad);
    }

    __host__ __device__ 
    inline FloatGrad operator-(const FloatGrad& other) const {
        return FloatGrad(data - other.data, grad - other.grad);
    }

    __host__ __device__ 
    inline FloatGrad operator*(const FloatGrad& other) const {
        return FloatGrad(data * other.data, grad * other.data + data * other.grad);
    }

    __host__ __device__ 
    inline FloatGrad operator/(const FloatGrad& other) const {
        return FloatGrad(data / other.data, (grad * other.data - data * other.grad) / (other.data * other.data));
    }
};

#endif // FLOAT_GRAD_H
