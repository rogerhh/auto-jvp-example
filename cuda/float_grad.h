#ifndef FLOAT_GRAD_H
#define FLOAT_GRAD_H

#include <math.h>

// Forward declaration of FloatGradBaseRef
template <typename FloatType> struct FloatGradBase;

template <typename FloatType>
struct FloatGradBaseRef {
    FloatType* data_ptr;
    FloatType* grad_ptr;

    __host__ __device__
    FloatGradBaseRef(FloatType* data, FloatType* grad)
    : data_ptr(data), grad_ptr(grad) {}

    __host__ __device__
    operator FloatGradBase<FloatType>() const {
        return FloatGradBase<FloatType>{*data_ptr, *grad_ptr};
    }

    __host__ __device__
    FloatGradBaseRef& operator=(const FloatGradBase<FloatType>& other) {
        *data_ptr = other.data;
        *grad_ptr = other.grad;
        return *this;
    }

    __host__ __device__
    FloatGradBaseRef& operator=(const FloatGradBaseRef<FloatType>& other) {
        *data_ptr = *other.data_ptr;
        *grad_ptr = *other.grad_ptr;
        return *this;
    }

    template <typename OtherType>
    __host__ __device__
    bool operator==(const OtherType& other) const {
        return FloatGradBase<FloatType>(*this) == FloatGradBase<FloatType>(other);
    }

    template <typename OtherType>
    __host__ __device__
    bool operator!=(const OtherType& other) const {
        return FloatGradBase<FloatType>(*this) != FloatGradBase<FloatType>(other);
    }

    template <typename OtherType>
    __host__ __device__
    bool operator<(const OtherType& other) const {
        return FloatGradBase<FloatType>(*this) < FloatGradBase<FloatType>(other);
    }

    template <typename OtherType>
    __host__ __device__
    bool operator>(const OtherType& other) const {
        return FloatGradBase<FloatType>(*this) > FloatGradBase<FloatType>(other);
    }

    template <typename OtherType>
    __host__ __device__
    bool operator<=(const OtherType& other) const {
        return FloatGradBase<FloatType>(*this) <= FloatGradBase<FloatType>(other);
    }

    template <typename OtherType>
    __host__ __device__
    bool operator>=(const OtherType& other) const {
        return FloatGradBase<FloatType>(*this) >= FloatGradBase<FloatType>(other);
    }

    template <typename OtherType>
    __host__ __device__
    FloatGradBase<FloatType> operator+(const OtherType& other) const {
        return FloatGradBase<FloatType>(*this) + FloatGradBase<FloatType>(other);
    }

    template <typename OtherType>
    __host__ __device__
    FloatGradBase<FloatType> operator-(const OtherType& other) const {
        return FloatGradBase<FloatType>(*this) - FloatGradBase<FloatType>(other);
    }

    template <typename OtherType>
    __host__ __device__
    FloatGradBase<FloatType> operator*(const OtherType& other) const {
        return FloatGradBase<FloatType>(*this) * FloatGradBase<FloatType>(other);
    }

    template <typename OtherType>
    __host__ __device__
    FloatGradBase<FloatType> operator/(const OtherType& other) const {
        return FloatGradBase<FloatType>(*this) / FloatGradBase<FloatType>(other);
    }

    template <typename OtherType>
    __host__ __device__
    FloatGradBaseRef& operator+=(const OtherType& other) {
        *this = *this + FloatGradBase<FloatType>(other);
        return *this;
    }

    template <typename OtherType>
    __host__ __device__
    FloatGradBaseRef& operator-=(const OtherType& other) {
        *this = *this - FloatGradBase<FloatType>(other);
        return *this;
    }

    template <typename OtherType>
    __host__ __device__
    FloatGradBaseRef& operator*=(const OtherType& other) {
        *this = *this * FloatGradBase<FloatType>(other);
        return *this;
    }

    template <typename OtherType>
    __host__ __device__
    FloatGradBaseRef& operator/=(const OtherType& other) {
        *this = *this / FloatGradBase<FloatType>(other);
        return *this;
    }

    // Full equality check
    template <typename OtherType>
    __host__ __device__
    bool eq(const OtherType& other) const {
        return FloatGradBase<FloatType>(*this).eq(FloatGradBase<FloatType>(other));
    }
};


template <typename FloatType>
struct FloatGradBase {
    FloatType data, grad;

    __host__ __device__
    FloatGradBase(FloatType data) : data(data), grad(0.0f) {}

    __host__ __device__
    FloatGradBase(FloatType data, FloatType grad)
    : data(data), grad(grad) {}

    __host__ __device__
    FloatGradBase(const FloatGradBaseRef<FloatType>& other)
    : data(*other.data_ptr), grad(*other.grad_ptr) {}

    __host__ __device__
    FloatGradBase(const FloatGradBase<FloatType>& other)
    : data(other.data), grad(other.grad) {}

    __host__ __device__
    FloatGradBase& operator=(const FloatGradBaseRef<FloatType>& other) {
        data = *other.data_ptr;
        grad = *other.grad_ptr;
        return *this;
    }

    __host__ __device__
    FloatGradBase& operator=(const FloatGradBase& other) {
        data = other.data;
        grad = other.grad;
        return *this;
    }

    __host__ __device__
    bool operator==(const FloatGradBase& other) const {
        return data == other.data;
    }

    __host__ __device__
    bool operator!=(const FloatGradBase& other) const {
        return data != other.data;
    }

    __host__ __device__
    bool operator<(const FloatGradBase& other) const {
        return data < other.data;
    }

    __host__ __device__
    bool operator>(const FloatGradBase& other) const {
        return data > other.data;
    }

    __host__ __device__
    bool operator<=(const FloatGradBase& other) const {
        return data <= other.data;
    }

    __host__ __device__
    bool operator>=(const FloatGradBase& other) const {
        return data >= other.data;
    }

    __host__ __device__
    FloatGradBase operator+(const FloatGradBase& other) const {
        return FloatGradBase(data + other.data, grad + other.grad);
    }

    __host__ __device__
    FloatGradBase operator-(const FloatGradBase& other) const {
        return FloatGradBase(data - other.data, grad - other.grad);
    }

    __host__ __device__
    FloatGradBase operator*(const FloatGradBase& other) const {
        return FloatGradBase(data * other.data, grad * other.data + data * other.grad);
    }

    __host__ __device__
    FloatGradBase operator/(const FloatGradBase& other) const {
        return FloatGradBase(data / other.data, (grad * other.data - data * other.grad) / (other.data * other.data));
    }

    __host__ __device__
    FloatGradBase& operator+=(const FloatGradBase& other) {
        data += other.data;
        grad += other.grad;
        return *this;
    }

    __host__ __device__
    FloatGradBase& operator-=(const FloatGradBase& other) {
        data -= other.data;
        grad -= other.grad;
        return *this;
    }

    __host__ __device__
    FloatGradBase& operator*=(const FloatGradBase& other) {
        grad = grad * other.data + data * other.grad;
        data *= other.data;
        return *this;
    }

    __host__ __device__
    FloatGradBase& operator/=(const FloatGradBase& other) {
        grad = (grad * other.data - data * other.grad) / (other.data * other.data);
        data /= other.data;
        return *this;
    }

    // Full equality check
    __host__ __device__
    bool eq(const FloatGradBase& other) const {
        return data == other.data && grad == other.grad;
    }

    template <typename OtherType>
    __host__ __device__
    bool operator==(const OtherType& other) const {
        return data == FloatGradBase(other).data;
    }

    template <typename OtherType>
    __host__ __device__
    bool operator!=(const OtherType& other) const {
        return data != FloatGradBase(other).data;
    }

    template <typename OtherType>
    __host__ __device__
    bool operator<(const OtherType& other) const {
        return data < FloatGradBase(other).data;
    }

    template <typename OtherType>
    __host__ __device__
    bool operator>(const OtherType& other) const {
        return data > FloatGradBase(other).data;
    }

    template <typename OtherType>
    __host__ __device__
    bool operator<=(const OtherType& other) const {
        return data <= FloatGradBase(other).data;
    }

    template <typename OtherType>
    __host__ __device__
    bool operator>=(const OtherType& other) const {
        return data >= FloatGradBase(other).data;
    }

    template <typename OtherType>
    __host__ __device__
    FloatGradBase operator+(const OtherType& other) const {
        return (*this) + FloatGradBase(other);
    }

    template <typename OtherType>
    __host__ __device__
    FloatGradBase operator-(const OtherType& other) const {
        return (*this) - FloatGradBase(other);
    }

    template <typename OtherType>
    __host__ __device__
    FloatGradBase operator*(const OtherType& other) const {
        return (*this) * FloatGradBase(other);
    }

    template <typename OtherType>
    __host__ __device__
    FloatGradBase operator/(const OtherType& other) const {
        return (*this) / FloatGradBase(other);
    }

    template <typename OtherType>
    __host__ __device__
    FloatGradBase& operator+=(const OtherType& other) {
        *this = (*this) + FloatGradBase(other);
        return *this;
    }

    template <typename OtherType>
    __host__ __device__
    FloatGradBase& operator-=(const OtherType& other) {
        *this = (*this) - FloatGradBase(other);
        return *this;
    }

    template <typename OtherType>
    __host__ __device__
    FloatGradBase& operator*=(const OtherType& other) {
        *this = (*this) * FloatGradBase(other);
        return *this;
    }

    template <typename OtherType>
    __host__ __device__
    FloatGradBase& operator/=(const OtherType& other) {
        *this = (*this) / FloatGradBase(other);
        return *this;
    }

    // Full equality check
    template <typename OtherType>
    __host__ __device__
    bool eq(const OtherType& other) const {
        return this->eq(FloatGradBase<FloatType>(other));
    }

};

template <typename FloatType>
__host__ __device__
FloatGradBase<FloatType> sqrtf(const FloatGradBase<FloatType>& x) {
    FloatType sqrt_x = sqrtf(x.data);
    return FloatGradBase<FloatType>(sqrt_x, x.grad / (FloatType(2.0f) * sqrt_x));
}

template <typename FloatType>
__host__ __device__
FloatGradBase<FloatType> sqrtf(const FloatGradBaseRef<FloatType>& x) {
    return sqrtf(FloatGradBase<FloatType>(x));
}

//////////////////////////////////////////////////////////
/// Access functions for entries in vector types
//////////////////////////////////////////////////////////

template <typename FloatType>
struct FloatGradRef : FloatGradBaseRef<FloatType> {
    using FloatGradBaseRef<FloatType>::FloatGradBaseRef;

    __host__ __device__
    FloatGradRef& operator=(const FloatGradBase<FloatType>& other) {
        *(this->data_ptr) = other.data;
        *(this->grad_ptr) = other.grad;
        return *this;
    }

    __host__ __device__
    FloatGradRef& operator=(const FloatGradBaseRef<FloatType>& other) {
        *(this->data_ptr) = *other.data_ptr;
        *(this->grad_ptr) = *other.grad_ptr;
        return *this;
    }
};

template <>
struct FloatGradRef<float2> : FloatGradBaseRef<float2> {
    template <typename... Args>
    __host__ __device__
    FloatGradRef<float2>(Args&&... args) 
    : FloatGradBaseRef<float2>(std::forward<Args>(args)...),
      x(&data_ptr->x, &grad_ptr->x), y(&data_ptr->y, &grad_ptr->y) {}

    __host__ __device__
    FloatGradRef& operator=(const FloatGradBase<float2>& other) {
        *(this->data_ptr) = other.data;
        *(this->grad_ptr) = other.grad;
        return *this;
    }

    __host__ __device__
    FloatGradRef& operator=(const FloatGradBaseRef<float2>& other) {
        *(this->data_ptr) = *other.data_ptr;
        *(this->grad_ptr) = *other.grad_ptr;
        return *this;
    }

    FloatGradBaseRef<float> x;
    FloatGradBaseRef<float> y;
};

template <>
struct FloatGradRef<float3> : FloatGradBaseRef<float3> {
    template <typename... Args>
    __host__ __device__
    FloatGradRef<float3>(Args&&... args) 
    : FloatGradBaseRef<float3>(std::forward<Args>(args)...),
      x(&data_ptr->x, &grad_ptr->x), 
      y(&data_ptr->y, &grad_ptr->y),
      z(&data_ptr->z, &grad_ptr->z) {}

    __host__ __device__
    FloatGradRef& operator=(const FloatGradBase<float3>& other) {
        *(this->data_ptr) = other.data;
        *(this->grad_ptr) = other.grad;
        return *this;
    }

    __host__ __device__
    FloatGradRef& operator=(const FloatGradBaseRef<float3>& other) {
        *(this->data_ptr) = *other.data_ptr;
        *(this->grad_ptr) = *other.grad_ptr;
        return *this;
    }

    FloatGradBaseRef<float> x;
    FloatGradBaseRef<float> y;
    FloatGradBaseRef<float> z;
};

template <>
struct FloatGradRef<float4> : FloatGradBaseRef<float4> {
    template <typename... Args>
    __host__ __device__
    FloatGradRef<float4>(Args&&... args) 
    : FloatGradBaseRef<float4>(std::forward<Args>(args)...),
      x(&data_ptr->x, &grad_ptr->x), 
      y(&data_ptr->y, &grad_ptr->y),
      z(&data_ptr->z, &grad_ptr->z),
      w(&data_ptr->w, &grad_ptr->w) {}

    __host__ __device__
    FloatGradRef& operator=(const FloatGradBase<float4>& other) {
        *(this->data_ptr) = other.data;
        *(this->grad_ptr) = other.grad;
        return *this;
    }

    __host__ __device__
    FloatGradRef& operator=(const FloatGradBaseRef<float4>& other) {
        *(this->data_ptr) = *other.data_ptr;
        *(this->grad_ptr) = *other.grad_ptr;
        return *this;
    }

    FloatGradBaseRef<float> x;
    FloatGradBaseRef<float> y;
    FloatGradBaseRef<float> z;
    FloatGradBaseRef<float> w;
};

template <typename FloatType>
struct FloatGrad : FloatGradBase<FloatType> {
    using FloatGradBase<FloatType>::FloatGradBase;

    __host__ __device__
    FloatGrad(const FloatGradBase<FloatType>& other)
    : FloatGradBase<FloatType>(other.data, other.grad) {}

    __host__ __device__
    FloatGrad& operator=(const FloatGradBase<FloatType>& other) {
        this->data = other.data;
        this->grad = other.grad;
        return *this;
    }

    __host__ __device__
    FloatGrad& operator=(const FloatGradBaseRef<FloatType>& other) {
        this->data = *other.data_ptr;
        this->grad = *other.grad_ptr;
        return *this;
    }
};

template <>
struct FloatGrad<float2> : FloatGradBase<float2> {
    template <typename... Args>
    __host__ __device__
    FloatGrad<float2>(Args&&... args) 
    : FloatGradBase<float2>(std::forward<Args>(args)...),
      x(&data.x, &grad.x), y(&data.y, &grad.y) {}

    __host__ __device__
    FloatGrad(const FloatGradBase<float2>& other)
    : FloatGradBase<float2>(other.data, other.grad),
      x(&data.x, &grad.x), y(&data.y, &grad.y) {}

    __host__ __device__
    FloatGrad& operator=(const FloatGradBase<float2>& other) {
        this->data = other.data;
        this->grad = other.grad;
        return *this;
    }

    __host__ __device__
    FloatGrad& operator=(const FloatGradBaseRef<float2>& other) {
        this->data = *other.data_ptr;
        this->grad = *other.grad_ptr;
        return *this;
    }

    FloatGradBaseRef<float> x;
    FloatGradBaseRef<float> y;
};

template <>
struct FloatGrad<float3> : FloatGradBase<float3> {
    template <typename... Args>
    __host__ __device__
    FloatGrad<float3>(Args&&... args) 
    : FloatGradBase<float3>(std::forward<Args>(args)...),
      x(&data.x, &grad.x), 
      y(&data.y, &grad.y),
      z(&data.z, &grad.z) {}

    __host__ __device__
    FloatGrad(const FloatGradBase<float3>& other)
    : FloatGradBase<float3>(other.data, other.grad),
      x(&data.x, &grad.x), 
      y(&data.y, &grad.y),
      z(&data.z, &grad.z) {}

    __host__ __device__
    FloatGrad& operator=(const FloatGradBase<float3>& other) {
        this->data = other.data;
        this->grad = other.grad;
        return *this;
    }

    __host__ __device__
    FloatGrad& operator=(const FloatGradBaseRef<float3>& other) {
        this->data = *other.data_ptr;
        this->grad = *other.grad_ptr;
        return *this;
    }

    FloatGradBaseRef<float> x;
    FloatGradBaseRef<float> y;
    FloatGradBaseRef<float> z;
};

template <>
struct FloatGrad<float4> : FloatGradBase<float4> {
    template <typename... Args>
    __host__ __device__
    FloatGrad<float4>(Args&&... args) 
    : FloatGradBase<float4>(std::forward<Args>(args)...),
      x(&data.x, &grad.x), 
      y(&data.y, &grad.y),
      z(&data.z, &grad.z),
      w(&data.w, &grad.w) {}

    __host__ __device__
    FloatGrad(const FloatGradBase<float4>& other)
    : FloatGradBase<float4>(other.data, other.grad),
      x(&data.x, &grad.x), 
      y(&data.y, &grad.y),
      z(&data.z, &grad.z),
      w(&data.w, &grad.w) {}

    __host__ __device__
    FloatGrad& operator=(const FloatGradBase<float4>& other) {
        this->data = other.data;
        this->grad = other.grad;
        return *this;
    }

    __host__ __device__
    FloatGrad& operator=(const FloatGradBaseRef<float4>& other) {
        this->data = *other.data_ptr;
        this->grad = *other.grad_ptr;
        return *this;
    }

    FloatGradBaseRef<float> x;
    FloatGradBaseRef<float> y;
    FloatGradBaseRef<float> z;
    FloatGradBaseRef<float> w;
};

//////////////////////////////////////////////////////////
/// Container
//////////////////////////////////////////////////////////

template <typename FloatType>
struct FloatGradArray {
    FloatType* data_ptr;
    FloatType* grad_ptr;

    __host__ __device__
    FloatGradArray(FloatType* data, FloatType* grad)
        : data_ptr(data), grad_ptr(grad) {}

    FloatGradBaseRef<FloatType> operator[](int index) {
        return FloatGradBaseRef<FloatType>(data_ptr + index, grad_ptr + index);
    }

    FloatGradBase<FloatType> operator[](int index) const {
        return FloatGradBase<FloatType>{data_ptr[index], grad_ptr[index]};
    }
};

#endif // FLOAT_GRAD_H
