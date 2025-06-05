#ifndef FLOAT_GRAD_H
#define FLOAT_GRAD_H

#include <math.h>

// Forward declaration of FloatGradRef
template <typename FloatType> struct FloatGradRef;

template <typename FloatType>
struct FloatGrad {
    FloatType data, grad;

    __host__ __device__
    FloatGrad(FloatType data) : data(data), grad(0.0f) {}

    __host__ __device__
    FloatGrad(FloatType data, FloatType grad)
    : data(data), grad(grad) {}

    __host__ __device__
    FloatGrad& operator=(const FloatGrad& other) {
        data = other.data;
        grad = other.grad;
        return *this;
    }

    // Access functions for vector types

    __host__ __device__
    bool operator==(const FloatGrad& other) const {
        return data == other.data;
    }

    __host__ __device__
    bool operator!=(const FloatGrad& other) const {
        return data != other.data;
    }

    __host__ __device__
    bool operator<(const FloatGrad& other) const {
        return data < other.data;
    }

    __host__ __device__
    bool operator>(const FloatGrad& other) const {
        return data > other.data;
    }

    __host__ __device__
    bool operator<=(const FloatGrad& other) const {
        return data <= other.data;
    }

    __host__ __device__
    bool operator>=(const FloatGrad& other) const {
        return data >= other.data;
    }

    __host__ __device__
    FloatGrad operator+(const FloatGrad& other) const {
        return FloatGrad(data + other.data, grad + other.grad);
    }

    __host__ __device__
    FloatGrad operator-(const FloatGrad& other) const {
        return FloatGrad(data - other.data, grad - other.grad);
    }

    __host__ __device__
    FloatGrad operator*(const FloatGrad& other) const {
        return FloatGrad(data * other.data, grad * other.data + data * other.grad);
    }

    __host__ __device__
    FloatGrad operator/(const FloatGrad& other) const {
        return FloatGrad(data / other.data, (grad * other.data - data * other.grad) / (other.data * other.data));
    }

    __host__ __device__
    FloatGrad& operator+=(const FloatGrad& other) {
        data += other.data;
        grad += other.grad;
        return *this;
    }

    __host__ __device__
    FloatGrad& operator-=(const FloatGrad& other) {
        data -= other.data;
        grad -= other.grad;
        return *this;
    }

    __host__ __device__
    FloatGrad& operator*=(const FloatGrad& other) {
        grad = grad * other.data + data * other.grad;
        data *= other.data;
        return *this;
    }

    __host__ __device__
    FloatGrad& operator/=(const FloatGrad& other) {
        grad = (grad * other.data - data * other.grad) / (other.data * other.data);
        data /= other.data;
        return *this;
    }

    template <typename OtherType>
    __host__ __device__
    bool operator==(const OtherType& other) const {
        return data == FloatGrad(other).data;
    }

    template <typename OtherType>
    __host__ __device__
    bool operator!=(const OtherType& other) const {
        return data != FloatGrad(other).data;
    }

    template <typename OtherType>
    __host__ __device__
    bool operator<(const OtherType& other) const {
        return data < FloatGrad(other).data;
    }

    template <typename OtherType>
    __host__ __device__
    bool operator>(const OtherType& other) const {
        return data > FloatGrad(other).data;
    }

    template <typename OtherType>
    __host__ __device__
    bool operator<=(const OtherType& other) const {
        return data <= FloatGrad(other).data;
    }

    template <typename OtherType>
    __host__ __device__
    bool operator>=(const OtherType& other) const {
        return data >= FloatGrad(other).data;
    }

    template <typename OtherType>
    __host__ __device__
    FloatGrad operator+(const OtherType& other) const {
        return (*this) + FloatGrad(other);
    }

    template <typename OtherType>
    __host__ __device__
    FloatGrad operator-(const OtherType& other) const {
        return (*this) - FloatGrad(other);
    }

    template <typename OtherType>
    __host__ __device__
    FloatGrad operator*(const OtherType& other) const {
        return (*this) * FloatGrad(other);
    }

    template <typename OtherType>
    __host__ __device__
    FloatGrad operator/(const OtherType& other) const {
        return (*this) / FloatGrad(other);
    }

    template <typename OtherType>
    __host__ __device__
    FloatGrad& operator+=(const OtherType& other) {
        *this = (*this) + FloatGrad(other);
        return *this;
    }

    template <typename OtherType>
    __host__ __device__
    FloatGrad& operator-=(const OtherType& other) {
        *this = (*this) - FloatGrad(other);
        return *this;
    }

    template <typename OtherType>
    __host__ __device__
    FloatGrad& operator*=(const OtherType& other) {
        *this = (*this) * FloatGrad(other);
        return *this;
    }

    template <typename OtherType>
    __host__ __device__
    FloatGrad& operator/=(const OtherType& other) {
        *this = (*this) / FloatGrad(other);
        return *this;
    }

};

template <typename FloatType>
struct FloatGradRef {
    __host__ __device__
    FloatGradRef(FloatType* data, FloatType* grad)
        : data_ptr(data), grad_ptr(grad) {}

    __host__ __device__
    operator FloatGrad<FloatType>() const {
        return FloatGrad<FloatType>{*data_ptr, *grad_ptr};
    }

    __host__ __device__
    FloatGradRef& operator=(const FloatGrad<FloatType>& other) {
        *data_ptr = other.data;
        *grad_ptr = other.grad;
        return *this;
    }

    template <typename OtherType>
    __host__ __device__
    bool operator==(const OtherType& other) const {
        return FloatGrad<FloatType>(*this) == FloatGrad<FloatType>(other);
    }

    template <typename OtherType>
    __host__ __device__
    bool operator!=(const OtherType& other) const {
        return FloatGrad<FloatType>(*this) != FloatGrad<FloatType>(other);
    }

    template <typename OtherType>
    __host__ __device__
    bool operator<(const OtherType& other) const {
        return FloatGrad<FloatType>(*this) < FloatGrad<FloatType>(other);
    }

    template <typename OtherType>
    __host__ __device__
    bool operator>(const OtherType& other) const {
        return FloatGrad<FloatType>(*this) > FloatGrad<FloatType>(other);
    }

    template <typename OtherType>
    __host__ __device__
    bool operator<=(const OtherType& other) const {
        return FloatGrad<FloatType>(*this) <= FloatGrad<FloatType>(other);
    }

    template <typename OtherType>
    __host__ __device__
    bool operator>=(const OtherType& other) const {
        return FloatGrad<FloatType>(*this) >= FloatGrad<FloatType>(other);
    }

    template <typename OtherType>
    __host__ __device__
    FloatGrad<FloatType> operator+(const OtherType& other) const {
        return FloatGrad<FloatType>(*this) + FloatGrad<FloatType>(other);
    }

    template <typename OtherType>
    __host__ __device__
    FloatGrad<FloatType> operator-(const OtherType& other) const {
        return FloatGrad<FloatType>(*this) - FloatGrad<FloatType>(other);
    }

    template <typename OtherType>
    __host__ __device__
    FloatGrad<FloatType> operator*(const OtherType& other) const {
        return FloatGrad<FloatType>(*this) * FloatGrad<FloatType>(other);
    }

    template <typename OtherType>
    __host__ __device__
    FloatGrad<FloatType> operator/(const OtherType& other) const {
        return FloatGrad<FloatType>(*this) / FloatGrad<FloatType>(other);
    }

    template <typename OtherType>
    __host__ __device__
    FloatGradRef& operator+=(const OtherType& other) {
        *this = *this + FloatGrad<FloatType>(other);
        return *this;
    }

    template <typename OtherType>
    __host__ __device__
    FloatGradRef& operator-=(const OtherType& other) {
        *this = *this - FloatGrad<FloatType>(other);
        return *this;
    }

    template <typename OtherType>
    __host__ __device__
    FloatGradRef& operator*=(const OtherType& other) {
        *this = *this * FloatGrad<FloatType>(other);
        return *this;
    }

    template <typename OtherType>
    __host__ __device__
    FloatGradRef& operator/=(const OtherType& other) {
        *this = *this / FloatGrad<FloatType>(other);
        return *this;
    }

    private:
        FloatType* data_ptr;
        FloatType* grad_ptr;
};

template <typename FloatType>
__host__ __device__
FloatGrad<FloatType> sqrtf(const FloatGrad<FloatType>& x) {
    FloatType sqrt_x = sqrtf(x.data);
    return FloatGrad<FloatType>(sqrt_x, x.grad / (FloatType(2.0f) * sqrt_x));
}

template <typename FloatType>
__host__ __device__
FloatGrad<FloatType> sqrtf(const FloatGradRef<FloatType>& x) {
    return sqrtf(FloatGrad<FloatType>(x));
}

template <typename FloatType>
struct FloatGradArray {
    FloatType* data_ptr;
    FloatType* grad_ptr;

    __host__ __device__
    FloatGradArray(FloatType* data, FloatType* grad)
        : data_ptr(data), grad_ptr(grad) {}

    FloatGradRef<FloatType> operator[](int index) & {
        return FloatGradRef<FloatType>(data_ptr + index, grad_ptr + index);
    }

    FloatGrad<FloatType> operator[](int index) && {
        return FloatGrad<FloatType>{data_ptr[index], grad_ptr[index]};
    }

    FloatGrad<FloatType> operator[](int index) const & {
        return FloatGrad<FloatType>{data_ptr[index], grad_ptr[index]};
    }
};

#endif // FLOAT_GRAD_H
