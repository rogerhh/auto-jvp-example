#ifndef FLOAT_GRAD_H
#define FLOAT_GRAD_H

#include <type_traits>

template <typename FloatType>
struct FloatGrad;

template <typename FloatType>
struct FloatGradRef {
    // Delete default constructor to prevent uninitialized usage
    FloatGradRef() = delete;

    __host__ __device__
    FloatGradRef(FloatType* data_ptr, FloatType* grad_ptr)
        : data_ptr_(data_ptr), grad_ptr_(grad_ptr) {}

    __host__ __device__
    FloatGradRef(const FloatGradRef& other)
        : data_ptr_(other.data_ptr_), grad_ptr_(other.grad_ptr_) {}

    __host__ __device__
    FloatGradRef& operator=(const FloatGradRef& other) {
        if (this != &other) {
            *data_ptr_ = *other.data_ptr_;
            *grad_ptr_ = *other.grad_ptr_;
        }
        return *this;
    }

    __host__ __device__
    FloatGradRef& operator=(const FloatGrad<FloatType>& other) {
        *data_ptr_ = other.data_;
        *grad_ptr_ = other.grad_;
        return *this;
    }

    __host__ __device__
    FloatType& data() {
        return *data_ptr_;
    }
    __host__ __device__
    const FloatType& data() const {
        return *data_ptr_;
    }
    __host__ __device__
    FloatType& grad() {
        return *grad_ptr_;
    }
    __host__ __device__
    const FloatType& grad() const {
        return *grad_ptr_;
    }

    FloatType* data_ptr_;
    FloatType* grad_ptr_;

};

template <typename FloatType>
struct FloatGrad {
    __host__ __device__
    FloatGrad(const FloatType& data, const FloatType& grad)
        : data_(data), grad_(grad) {}

    __host__ __device__
    FloatGrad(const FloatGrad& other)
        : data_(other.data_), grad_(other.grad_) {}

    __host__ __device__
    FloatGrad(const FloatGradRef<FloatType>& ref)
        : data_(*ref.data_ptr_), grad_(*ref.grad_ptr_) {}

    __host__ __device__
    FloatGrad& operator=(const FloatGrad& other) {
        if (this != &other) {
            data_ = other.data_;
            grad_ = other.grad_;
        }
        return *this;
    }

    __host__ __device__
    FloatGrad& operator=(const FloatGradRef<FloatType>& ref) {
        data_ = *ref.data_ptr_;
        grad_ = *ref.grad_ptr_;
        return *this;
    }

    __host__ __device__
    FloatType& data() {
        return data_;
    }
    __host__ __device__
    const FloatType& data() const {
        return data_;
    }
    __host__ __device__
    FloatType& grad() {
        return grad_;
    }
    __host__ __device__
    const FloatType& grad() const {
        return grad_;
    }

    FloatType data_;
    FloatType grad_;
};


//////////////////////////////////////////////////////////////////////////////
/// Advanced template checking using SFINAE
//////////////////////////////////////////////////////////////////////////////

template <typename T>
struct is_float_grad : std::false_type {};

template <typename FloatType>
struct is_float_grad<FloatGradRef<FloatType>> : std::true_type {};

template <typename FloatType>
struct is_float_grad<FloatGrad<FloatType>> : std::true_type {};

template <typename T>
decltype(auto) get_data(const T& t) {
    if constexpr (is_float_grad<T>::value) {
        return t.data();
    } else {
        return t;
    }
}

//////////////////////////////////////////////////////////////////////////////
/// Operator overloads
/// Overload all operators for any FloatGrad type
//////////////////////////////////////////////////////////////////////////////

/// Comparison operators

template <typename T1, typename T2>
__host__ __device__
std::enable_if_t<is_float_grad<T1>::value 
                 || is_float_grad<T2>::value,
                 bool>
operator==(const T1& a, const T2& b) {
    return get_data<T1>(a) == get_data<T2>(b);
}

template <typename T1, typename T2>
__host__ __device__
std::enable_if_t<is_float_grad<T1>::value 
                 || is_float_grad<T2>::value,
                 bool>
operator!=(const T1& a, const T2& b) {
    return get_data<T1>(a) != get_data<T2>(b);
}

template <typename T1, typename T2>
__host__ __device__
std::enable_if_t<is_float_grad<T1>::value 
                 || is_float_grad<T2>::value,
                 bool>
operator<(const T1& a, const T2& b) {
    return get_data<T1>(a) < get_data<T2>(b);
}

/// Arithmetic operators

template <typename T1, typename T2,
          typename F1=decltype(get_data(std::declval<T1>())),
          typename F2=decltype(get_data(std::declval<T2>())),
          typename F3=decltype(std::declval<F1>() + std::declval<F2>()),
          typename = std::enable_if_t<is_float_grad<T1>::value
                                      || is_float_grad<T2>::value>>
FloatGrad<F3> operator+(const T1& a, const T2& b) {
    F3 data = get_data<T1>(a) + get_data<T2>(b);
    F3 grad;
    if constexpr (is_float_grad<T1>::value 
        && is_float_grad<T2>::value) {
        grad = a.grad() + b.grad();
    } else if constexpr (is_float_grad<T1>::value) {
        grad = a.grad();
    } else {
        grad = b.grad();
    }
    return FloatGrad<F3>(data, grad);
}

template <typename T1, typename T2,
          typename F1=decltype(get_data(std::declval<T1>())),
          typename F2=decltype(get_data(std::declval<T2>())),
          typename F3=decltype(std::declval<F1>() * std::declval<F2>()),
          typename = std::enable_if_t<is_float_grad<T1>::value
                                      || is_float_grad<T2>::value>>
FloatGrad<F3> operator*(const T1& a, const T2& b) {
    F3 data = get_data<T1>(a) * get_data<T2>(b);
    F3 grad;
    if constexpr (is_float_grad<T1>::value 
        && is_float_grad<T2>::value) {
        grad = a.data() + b.data() + a.grad() * b.data();
    } else if constexpr (is_float_grad<T1>::value) {
        grad = a.grad() * b.data();
    } else {
        grad = a.data() * b.grad();
    }
    return FloatGrad<F3>(data, grad);
}

#include "float_grad_float2.h"


#endif // FLOAT_GRAD_H

