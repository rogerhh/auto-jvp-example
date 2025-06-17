#ifndef FLOAT_GRAD_H
#define FLOAT_GRAD_H

#include <type_traits>
#include <iostream>

template <typename FloatType>
struct FloatGrad;

template <typename FloatType>
struct FloatGradRef;

/////////////////////////////////////////////////////////////////////////////
/// SFINAE utility
/////////////////////////////////////////////////////////////////////////////

template <typename T>
struct always_false : std::false_type {};

template <typename T>
struct is_float_grad : std::false_type {};
template <typename FloatType>
struct is_float_grad<FloatGradRef<FloatType>> : std::true_type {};
template <typename FloatType>
struct is_float_grad<FloatGrad<FloatType>> : std::true_type {};

template <typename T>
struct is_float_grad_ref : std::false_type {};
template <typename FloatType>
struct is_float_grad_ref<FloatGradRef<FloatType>> : std::true_type {};

template <typename T>
struct is_float_grad_val : std::false_type {};
template <typename FloatType>
struct is_float_grad_val<FloatGrad<FloatType>> : std::true_type {};

template <typename T>
struct extract_float_grad_type { using type = T; };
template <typename FloatType>
struct extract_float_grad_type<FloatGrad<FloatType>> { using type = FloatType; };
template <typename FloatType>
struct extract_float_grad_type<FloatGradRef<FloatType>> { using type = FloatType; };

/////////////////////////////////////////////////////////////////////////////
/// Class definitions
/////////////////////////////////////////////////////////////////////////////

template <typename FloatType>
struct FloatGradRef {

    FloatType* data_ptr_;
    FloatType* grad_ptr_;

    // Delete default constructor to prevent uninitialized usage
    FloatGradRef() = delete;

    // Base constructor
    __host__ __device__
    FloatGradRef(FloatType* data_ptr, FloatType* grad_ptr)
        : data_ptr_(data_ptr), grad_ptr_(grad_ptr) {}

    // Constructing from reference type
    template <typename U = FloatType, 
              typename = std::enable_if_t<std::is_same_v<std::remove_const_t<U>, 
                                                         std::remove_const_t<FloatType>>>>
    __host__ __device__
    FloatGradRef(const FloatGradRef<U>& other)
        : data_ptr_(other.data_ptr_), grad_ptr_(other.grad_ptr_) {}
    
    // Constructing reference type from value type disabled

    // Assigning from reference type
    template <typename U = FloatType, 
              typename = std::enable_if_t<std::is_same_v<std::remove_const_t<U>, 
                                                         std::remove_const_t<FloatType>>>>
    __host__ __device__
    FloatGradRef& operator=(const FloatGradRef<U>& other) {
        *data_ptr_ = *other.data_ptr_;
        *grad_ptr_ = *other.grad_ptr_;
        return *this;
    }

    // Assigning from value type
    template <typename U = FloatType, 
              typename = std::enable_if_t<std::is_same_v<std::remove_const_t<U>, 
                                                         std::remove_const_t<FloatType>>>>
    __host__ __device__
    FloatGradRef& operator=(const FloatGrad<U>& other) {
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
};

template <typename FloatType>
struct FloatGrad {

    FloatType data_;
    FloatType grad_;

    __host__ __device__
    FloatGrad(const FloatType& data, const FloatType& grad)
        : data_(data), grad_(grad) {}

    template <typename U = FloatType, 
              typename = std::enable_if_t<std::is_same_v<std::remove_const_t<U>, 
                                                         std::remove_const_t<FloatType>>>>
    __host__ __device__
    FloatGrad(const FloatGrad<U>& other)
        : data_(other.data_), grad_(other.grad_) {}

    template <typename U = FloatType, 
              typename = std::enable_if_t<std::is_same_v<std::remove_const_t<U>, 
                                                         std::remove_const_t<FloatType>>>>
    __host__ __device__
    FloatGrad(const FloatGradRef<U>& ref)
        : data_(*ref.data_ptr_), grad_(*ref.grad_ptr_) {}

    template <typename U = FloatType, std::enable_if_t<std::is_same_v<std::remove_const_t<U>, 
                                                       float>, int> = 0>
    __host__ __device__
    FloatGrad(const FloatType& data)
        : data_(data), grad_(0.0f) {}

    template <typename U = FloatType, 
              typename = std::enable_if_t<std::is_same_v<std::remove_const_t<U>, 
                                                         std::remove_const_t<FloatType>>>>
    __host__ __device__
    FloatGrad& operator=(const FloatGrad<U>& other) {
        data_ = other.data_;
        grad_ = other.grad_;
        return *this;
    }

    template <typename U = FloatType, 
              typename = std::enable_if_t<std::is_same_v<std::remove_const_t<U>, 
                                                         std::remove_const_t<FloatType>>>>
    __host__ __device__
    FloatGrad& operator=(const FloatGradRef<U>& ref) {
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
};

//////////////////////////////////////////////////////////////////////////////
/// Container
//////////////////////////////////////////////////////////////////////////////

template <typename FloatType>
struct FloatGradArray {
    FloatType* data_arr_;
    FloatType* grad_arr_;

    __host__ __device__
    FloatGradArray(FloatType* data, FloatType* grad)
        : data_arr_(data), grad_arr_(grad) {}

    __host__ __device__
    FloatGradRef<FloatType> operator[](int index) {
        return FloatGradRef<FloatType>(data_arr_ + index, grad_arr_ + index);
    }

    __host__ __device__
    FloatGrad<FloatType> operator[](int index) const {
        return FloatGrad<FloatType>(data_arr_[index], grad_arr_[index]);
    }
};


//////////////////////////////////////////////////////////////////////////////
/// Advanced template checking using SFINAE
//////////////////////////////////////////////////////////////////////////////

template <typename T>
__host__ __device__
decltype(auto) get_data(const T& t) {
    if constexpr (is_float_grad<T>::value) {
        return t.data();
    } else {
        return t;
    }
}


template <typename T>
__host__ __device__
decltype(auto) get_grad(const T& t) {
    if constexpr (is_float_grad<T>::value) {
        return t.grad();
    }
    else if constexpr (std::is_same_v<std::remove_cv_t<T>, float>) {
        return 0.0f; // Default gradient for non-Floatgrad floats
    }
    else {
        static_assert(always_false<T>::value, "Unsupported type for get_grad");
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

template <typename T1, typename T2>
__host__ __device__
std::enable_if_t<is_float_grad<T1>::value 
                 || is_float_grad<T2>::value,
                 bool>
operator<=(const T1& a, const T2& b) {
    return get_data<T1>(a) <= get_data<T2>(b);
}

template <typename T1, typename T2>
__host__ __device__
std::enable_if_t<is_float_grad<T1>::value 
                 || is_float_grad<T2>::value,
                 bool>
operator>(const T1& a, const T2& b) {
    return get_data<T1>(a) > get_data<T2>(b);
}

template <typename T1, typename T2>
__host__ __device__
std::enable_if_t<is_float_grad<T1>::value 
                 || is_float_grad<T2>::value,
                 bool>
operator>=(const T1& a, const T2& b) {
    return get_data<T1>(a) >= get_data<T2>(b);
}

/// Arithmetic operators

template <typename T1, typename T2,
          typename = std::enable_if_t<is_float_grad<T1>::value
                                      || is_float_grad<T2>::value>>
__host__ __device__
auto operator+(const T1& a, const T2& b) {
    auto data = get_data<T1>(a) + get_data<T2>(b);
    decltype(data) grad;
    if constexpr (is_float_grad<T1>::value 
        && is_float_grad<T2>::value) {
        grad = a.grad() + b.grad();
    } else if constexpr (is_float_grad<T1>::value) {
        grad = a.grad();
    } else {
        grad = b.grad();
    }
    return FloatGrad<decltype(data)>(data, grad);
}

template <typename T1, typename T2,
          typename = std::enable_if_t<is_float_grad<T1>::value
                                      || is_float_grad<T2>::value>>
__host__ __device__
auto operator-(const T1& a, const T2& b) {
    auto data = get_data<T1>(a) - get_data<T2>(b);
    decltype(data) grad;
    if constexpr (is_float_grad<T1>::value 
        && is_float_grad<T2>::value) {
        grad = a.grad() - b.grad();
    } else if constexpr (is_float_grad<T1>::value) {
        grad = a.grad();
    } else {
        grad = -b.grad();
    }
    return FloatGrad<decltype(data)>(data, grad);
}

template <typename T1, typename T2,
          typename = std::enable_if_t<is_float_grad<T1>::value
                                      || is_float_grad<T2>::value>>
__host__ __device__
auto operator*(const T1& a, const T2& b) {
    auto data = get_data<T1>(a) * get_data<T2>(b);
    decltype(data) grad;
    if constexpr (is_float_grad<T1>::value 
        && is_float_grad<T2>::value) {
        grad = get_data(a) * get_grad(b) + get_grad(a) * get_data(b);
    } else if constexpr (is_float_grad<T1>::value) {
        grad = get_grad(a) * get_data(b);
    } else {
        grad = get_data(a) * get_grad(b);
    }
    return FloatGrad<decltype(data)>(data, grad);
}

template <typename T1, typename T2,
          typename = std::enable_if_t<is_float_grad<T1>::value
                                      || is_float_grad<T2>::value>>
__host__ __device__
auto operator/(const T1& a, const T2& b) {
    auto data = get_data<T1>(a) / get_data<T2>(b);
    decltype(data) grad;
    if constexpr (is_float_grad<T1>::value 
        && is_float_grad<T2>::value) {
        grad = (get_grad(a) * get_data(b) - get_data(a) * get_grad(b)) 
                / (get_data(b) * get_data(b));
    } else if constexpr (is_float_grad<T1>::value) {
        grad = get_grad(a) / get_data(b);
    } else {
        grad = -get_data(a) * get_grad(b) / (get_data(b) * get_data(b));
    }
    return FloatGrad<decltype(data)>(data, grad);
}

/// Compound assignment operators

template <typename T, typename OtherType>
__host__ __device__
std::enable_if_t<is_float_grad_val<T>::value, T&>
operator+=(T& t, const OtherType& other) {
    t = t + other;
    return t;
}

template <typename T, typename OtherType>
__host__ __device__
std::enable_if_t<is_float_grad_ref<T>::value, T>
operator+=(T t, const OtherType& other) {
    t = t + other;
    return t;
}

template <typename T, typename OtherType>
__host__ __device__
std::enable_if_t<is_float_grad_val<T>::value, T&>
operator-=(T& t, const OtherType& other) {
    t = t - other;
    return t;
}

template <typename T, typename OtherType>
__host__ __device__
std::enable_if_t<is_float_grad_ref<T>::value, T>
operator-=(T t, const OtherType& other) {
    t = t - other;
    return t;
}

template <typename T, typename OtherType>
__host__ __device__
std::enable_if_t<is_float_grad_val<T>::value, T&>
operator*=(T& t, const OtherType& other) {
    t = t * other;
    return t;
}

template <typename T, typename OtherType>
__host__ __device__
std::enable_if_t<is_float_grad_ref<T>::value, T>
operator*=(T t, const OtherType& other) {
    t = t * other;
    return t;
}

template <typename T, typename OtherType>
__host__ __device__
std::enable_if_t<is_float_grad_val<T>::value, T&>
operator/=(T& t, const OtherType& other) {
    t = t / other;
    return t;
}

template <typename T, typename OtherType>
__host__ __device__
std::enable_if_t<is_float_grad_ref<T>::value, T>
operator/=(T t, const OtherType& other) {
    t = t / other;
    return t;
}

#include "cuda/float_grad_float.h"
#include "cuda/float_grad_float2.h"
#include "cuda/float_grad_const_float2.h"
#include "cuda/float_grad_float3.h"
#include "cuda/float_grad_const_float3.h"
#include "cuda/float_grad_float4.h"
#include "cuda/float_grad_const_float4.h"


#endif // FLOAT_GRAD_H

