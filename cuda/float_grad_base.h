#ifndef FLOAT_GRAD_BASE_H
#define FLOAT_GRAD_BASE_H

#include <type_traits>
#include <iostream>

template <typename FloatType>
struct FloatGradBase;

template <typename FloatType>
struct FloatGradRefBase;

template <typename FloatType>
struct FloatGradArray;

/////////////////////////////////////////////////////////////////////////////
/// SFINAE utility
/////////////////////////////////////////////////////////////////////////////

template <typename T>
struct always_false : std::false_type {};

std::false_type is_float_grad_impl(const void*);
template <typename T>
std::true_type is_float_grad_impl(const FloatGradBase<T>*);
template <typename T>
std::true_type is_float_grad_impl(const FloatGradRefBase<T>*);
template <typename T>
std::true_type is_float_grad_impl(const FloatGradArray<T>*);
template <typename T>
using is_float_grad = decltype(is_float_grad_impl(std::declval<T*>()));

std::false_type is_float_grad_val_impl(const void*);
template <typename T>
std::true_type is_float_grad_val_impl(const FloatGradBase<T>*);
template <typename T>
using is_float_grad_val = decltype(is_float_grad_val_impl(std::declval<T*>()));

std::false_type is_float_grad_ref_impl(const void*);
template <typename T>
std::true_type is_float_grad_ref_impl(const FloatGradRefBase<T>*);
template <typename T>
using is_float_grad_ref = decltype(is_float_grad_ref_impl(std::declval<T*>()));

std::false_type is_float_grad_array_impl(const void*);
template <typename T>
std::true_type is_float_grad_array_impl(const FloatGradArray<T>*);
template <typename T>
using is_float_grad_array = decltype(is_float_grad_array_impl(std::declval<T*>()));

/////////////////////////////////////////////////////////////////////////////
/// Class definitions
/////////////////////////////////////////////////////////////////////////////

template <typename FloatType>
struct FloatGradRefBase {
    FloatType* const data_ptr_;
    FloatType* const grad_ptr_;

    // Virtual destructor for pure abstract base class
    virtual ~FloatGradRefBase() = 0;

    // Delete default constructor to prevent uninitialized usage
    FloatGradRefBase() = delete;

    // Base constructor
    __host__ __device__
    FloatGradRefBase(FloatType* data_ptr, FloatType* grad_ptr)
        : data_ptr_(data_ptr), grad_ptr_(grad_ptr) {}

    // Copy constructor
    __host__ __device__
    FloatGradRefBase(const FloatGradRefBase& other)
        : data_ptr_(other.data_ptr_), grad_ptr_(other.grad_ptr_) {}

    // Constructing from reference type
    template <typename U = FloatType, 
              typename = std::enable_if_t<std::is_same_v<std::remove_const_t<U>, 
                                                         std::remove_const_t<FloatType>>>>
    __host__ __device__
    FloatGradRefBase(const FloatGradRefBase<U>& other)
        : data_ptr_(other.data_ptr_), grad_ptr_(other.grad_ptr_) {}

    // Constructing reference type from value type disabled
    // Use the FloatGrad.ref() function

    // Default assignment operator
    __host__ __device__
    FloatGradRefBase& operator=(const FloatGradRefBase& other) {
        this->data() = other.data();
        this->grad() = other.grad();
        return *this;
    }

    template <typename OtherType>
    __host__ __device__
    FloatGradRefBase& operator=(const OtherType& other);

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
struct FloatGradBase {
    FloatType data_;
    FloatType grad_;

    // Virtual destructor for pure abstract base class
    // virtual ~FloatGradBase() = 0;
    ~FloatGradBase();

    // Copy constructor
    template <typename U = FloatType, 
              typename = std::enable_if_t<std::is_same_v<std::remove_const_t<U>, 
                                                         std::remove_const_t<FloatType>>>>
    __host__ __device__
    FloatGradBase(const FloatGradBase<U>& other)
        : data_(other.data_), grad_(other.grad_) {}

    template <typename T1, typename T2,
              typename = std::enable_if_t<std::is_same<std::decay_t<T1>, 
                                                       std::decay_t<FloatType>>::value
                                          && std::is_same<std::decay_t<T2>, 
                                                          std::decay_t<FloatType>>::value>>
    __host__ __device__
    FloatGradBase(const T1& data, const T2& grad)
        : data_(data), grad_(grad) {}

    __host__ __device__
    FloatGradBase(const FloatGradBase<FloatType>& other)
        : data_(other.data_), grad_(other.grad_) {}

    // Constructors for the FloatType
    // Need to explicitly disable the two-argument constructor
    template <typename... Args,
    std::enable_if_t<!(sizeof...(Args) == 2 &&
                       std::conjunction_v<std::is_same<std::decay_t<Args>, 
                                                       std::decay_t<FloatType>>...>), int> = 0>
    __host__ __device__
    FloatGradBase(Args&&... args);

    template <typename OtherType>
    __host__ __device__
    FloatGradBase& operator=(const OtherType& other);

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

/// Default class specializations

template <typename FloatType>
struct FloatGradRef : public FloatGradRefBase<FloatType> {
    using FloatGradRefBase<FloatType>::FloatGradRefBase;
    
    // All assignment operators
    template <typename OtherType>
    __host__ __device__
    FloatGradRef& operator=(const OtherType& other) {
        FloatGradRefBase<FloatType>::operator=(other);
        return *this;
    }
};

template <typename FloatType>
struct FloatGrad : public FloatGradBase<FloatType> {
    using FloatGradBase<FloatType>::FloatGradBase;

    __host__ __device__
    FloatGrad& operator=(const FloatGrad& other) {
        FloatGradBase<FloatType>::operator=(other);
        return *this;
    }

    template <typename OtherType>
    __host__ __device__
    FloatGrad& operator=(const OtherType& other) {
        FloatGradBase<FloatType>::operator=(other);
        return *this;
    }
};

//////////////////////////////////////////////////////////////////////////////
/// Container
//////////////////////////////////////////////////////////////////////////////
#include "cuda/float_grad_array.h"


//////////////////////////////////////////////////////////////////////////////
/// Access functions. This is the main function to overload between
/// FloatGrad and non-FloatGrad types.
//////////////////////////////////////////////////////////////////////////////

template <typename T>
inline __host__ __device__
decltype(auto) get_data(const T& t) {
    if constexpr (is_float_grad<T>::value) {
        return t.data();
    } else {
        return t;
    }
}

template <typename T>
inline __host__ __device__
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
/// Template constructor implementations
//////////////////////////////////////////////////////////////////////////////

template <typename FloatType>
FloatGradRefBase<FloatType>::~FloatGradRefBase() = default; // Default destructor
template <typename FloatType>
FloatGradBase<FloatType>::~FloatGradBase() = default; // Default destructor

// Note: This needs to be marked __noinline__ to prevent incorrect compiler optimizations
template <typename FloatType>
template <typename... Args,
std::enable_if_t<!(sizeof...(Args) == 2 &&
                   std::conjunction_v<std::is_same<std::decay_t<Args>, 
                                                   std::decay_t<FloatType>>...>), int>>
__noinline__ __host__ __device__
FloatGradBase<FloatType>::FloatGradBase(Args&&... args)
: data_(get_data(std::forward<Args>(args))...), grad_(get_grad(std::forward<Args>(args))...) {}

template <typename FloatType>
template <typename OtherType>
__host__ __device__
FloatGradRefBase<FloatType>& FloatGradRefBase<FloatType>::operator=(const OtherType& other) {
    this->data() = get_data(other);
    this->grad() = get_grad(other);
    return *this;
}

template <typename FloatType>
template <typename OtherType>
__host__ __device__
FloatGradBase<FloatType>& FloatGradBase<FloatType>::operator=(const OtherType& other) {
    this->data() = get_data(other);
    this->grad() = get_grad(other);
    return *this;
}

//////////////////////////////////////////////////////////////////////////////
/// Operator overloads
//////////////////////////////////////////////////////////////////////////////

template <typename T1, typename T2>
__host__ __device__
std::enable_if_t<(is_float_grad<T1>::value || is_float_grad<T2>::value)
                  && (!is_float_grad_array<T1>::value && !is_float_grad_array<T2>::value),
                  bool>
operator==(const T1& a, const T2& b) {
    return get_data<T1>(a) == get_data<T2>(b);
}

template <typename T1, typename T2>
__host__ __device__
std::enable_if_t<(is_float_grad<T1>::value || is_float_grad<T2>::value)
                  && (!is_float_grad_array<T1>::value && !is_float_grad_array<T2>::value),
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
auto operator+(T1 a, T2 b) {
    // TODO: This is ugly. Fix later
    if constexpr (is_float_grad_array<T1>::value 
                  || is_float_grad_array<T2>::value) {
        if constexpr (is_float_grad_array<T1>::value) {
            static_assert(!is_float_grad<T2>::value, 
                          "Cannot add FloatGradArray to FloatGrad");
            return T1(a.data_ptr() + b, a.grad_ptr() + b);
        } else {
            static_assert(!is_float_grad<T1>::value, 
                          "Cannot add FloatGrad to FloatGradArray");
            return T2(b.data_ptr() + a, b.grad_ptr() + a);
        }
    } 
    else { 
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

/// Compound assignment operators. Need to separate val and ref types due to 
/// array indexing returning r-value references.

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


#endif // FLOAT_GRAD_BASE_H

