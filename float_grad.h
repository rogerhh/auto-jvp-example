#ifndef FLOAT_GRAD_H
#define FLOAT_GRAD_H

#include <math.h>

struct FloatGrad {
    float data, grad;

    __host__ __device__ FloatGrad() : data(0.0f), grad(0.0f) {}
    __host__ __device__ FloatGrad(float data) : data(data), grad(0.0f) {}
    __host__ __device__ FloatGrad(float data, float grad) : data(data), grad(grad) {}

    __host__ __device__
    inline FloatGrad& operator=(const float& scalar) {
        data = scalar;
        grad = 0.0f;
        return *this;
    }

    __host__ __device__
    inline bool operator==(const float& scalar) const {
        return data == scalar;
    }

    __host__ __device__
    inline bool operator<(const float& scalar) const {
        return data < scalar;
    }

    __host__ __device__
    inline bool operator>(const float& scalar) const {
        return data > scalar;
    }

    __host__ __device__
    inline bool operator<=(const float& scalar) const {
        return data <= scalar;
    }

    __host__ __device__
    inline bool operator>=(const float& scalar) const {
        return data >= scalar;
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
    inline FloatGrad& operator=(const FloatGrad& other) {
        data = other.data;
        grad = other.grad;
        return *this;
    }

    __host__ __device__
    inline bool operator==(const FloatGrad& other) const {
        return data == other.data;
    }

    __host__ __device__
    inline bool operator<(const FloatGrad& other) const {
        return data < other.data;
    }

    __host__ __device__
    inline bool operator>(const FloatGrad& other) const {
        return data > other.data;
    }

    __host__ __device__
    inline bool operator<=(const FloatGrad& other) const {
        return data <= other.data;
    }

    __host__ __device__
    inline bool operator>=(const FloatGrad& other) const {
        return data >= other.data;
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

    __host__ __device__
    inline operator int() const {
        return static_cast<int>(data);
    }

    __host__ __device__
    inline operator float() const {
        return data;
    }
};

FloatGrad sqrtf(const FloatGrad& x) {
    float sqrt_x = sqrtf(x.data);
    return FloatGrad(sqrt_x, x.grad / (2.0f * sqrt_x));
}

template <typename FloatType>
struct alignas(sizeof(FloatType) * 2) Float2 {
    FloatType x, y;
    __host__ __device__ Float2() : x(0), y(0) {}
    __host__ __device__ Float2(FloatType x, FloatType y) : x(x), y(y) {}
    __host__ __device__ Float2(const FloatType& scalar) : x(scalar), y(scalar) {}
};

template <typename FloatType>
struct alignas(sizeof(FloatType) * 4) Float3 {
    FloatType x, y, z;
    __host__ __device__ Float3() : x(0), y(0), z(0) {}
    __host__ __device__ Float3(FloatType x, FloatType y, FloatType z) : x(x), y(y), z(z) {}
    __host__ __device__ Float3(const FloatType& scalar) : x(scalar), y(scalar), z(scalar) {}
};

template <typename FloatType>
struct alignas(sizeof(FloatType) * 4) Float4 {
    FloatType x, y, z, w;
    __host__ __device__ Float4() : x(0), y(0), z(0), w(0) {}
    __host__ __device__ Float4(FloatType x, FloatType y, FloatType z, FloatType w) : x(x), y(y), z(z), w(w) {}
    __host__ __device__ Float4(const FloatType& scalar) : x(scalar), y(scalar), z(scalar), w(scalar) {}
};

template <typename FloatType>
Float2<FloatType> make_float2(const FloatType& x, const FloatType& y) {
    return Float2<FloatType>(x, y);
}
template <typename FloatType>
Float3<FloatType> make_float3(const FloatType& x, const FloatType& y, const FloatType& z) {
    return Float3<FloatType>(x, y, z);
}
template <typename FloatType>
Float4<FloatType> make_float4(const FloatType& x, const FloatType& y, const FloatType& z, const FloatType& w) {
    return Float4<FloatType>(x, y, z, w);
}


// struct FloatGrad2 {
//     FloatGrad x, y;
// };
// 
// template <typename T1, typename T2>
// FloatGrad2 make_float2(const T1& x, const T2& y) {
//     return FloatGrad2{FloatGrad(x), FloatGrad(y)};
// }
// 
// struct FloatGrad3 {
//     FloatGrad x, y, z;
// };
// 
// template <typename T1, typename T2, typename T3>
// FloatGrad3 make_float3(const T1& x, const T2& y, const T3& z) {
//     return FloatGrad3{FloatGrad(x), FloatGrad(y), FloatGrad(z)};
// }
// 
// struct FloatGrad4 {
//     FloatGrad x, y, z, w;
// };
// 
// template <typename T1, typename T2, typename T3, typename T4>
// FloatGrad4 make_float4(const T1& x, const T2& y, const T3& z, const T4& w) {
//     return FloatGrad4{FloatGrad(x), FloatGrad(y), FloatGrad(z), FloatGrad(w)};
// }

#endif // FLOAT_GRAD_H
