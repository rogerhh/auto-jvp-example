#ifndef FLOAT_GRAD_FLOAT_H
#define FLOAT_GRAD_FLOAT_H

#include <type_traits>

template <typename T>
using is_float_type = std::is_same<std::decay_t<decltype(get_data(std::declval<T>()))>, float>;

//////////////////////////////////////////////////////////////////////////////
/// Additional builtin functions
/// Need to be very careful about these functions so that they don't get 
/// picked for scalar types like int, double, etc.
//////////////////////////////////////////////////////////////////////////////

template <typename T>
__host__ __device__
inline std::enable_if_t<is_float_type<T>::value
                        && is_float_grad<T>::value, FloatGrad<float>>
operator+(T x) {
    return FloatGrad<float>(x);
}

template <typename T>
__host__ __device__
inline std::enable_if_t<is_float_type<T>::value
                        && is_float_grad<T>::value, FloatGrad<float>>
operator-(T x) {
    return FloatGrad<float>(-get_data(x), -get_grad(x));
}

template <typename T>
__host__ __device__
inline std::enable_if_t<is_float_type<T>::value
                        && is_float_grad<T>::value, FloatGrad<float>>
fabs(T x) {
    return x >= 0.0f ? FloatGrad<float>(x) : FloatGrad<float>(-x);
}

template <typename T>
__host__ __device__
inline std::enable_if_t<is_float_type<T>::value
                        && is_float_grad<T>::value, FloatGrad<float>>
sqrtf(T a) {
    float data = sqrtf(get_data(a));
    float grad = get_grad(a) / (2.0f * data);
    return FloatGrad<float>(data, grad);
}

template <typename T>
__host__ __device__
inline std::enable_if_t<is_float_type<T>::value
                        && is_float_grad<T>::value, FloatGrad<float>>
floorf(T x) {
    return FloatGrad<float>(floorf(get_data(x)), 0.0f);
}

template <typename T>
__host__ __device__
inline std::enable_if_t<is_float_type<T>::value
                        && is_float_grad<T>::value, FloatGrad<float>>
ceilf(T x) {
    return FloatGrad<float>(ceilf(get_data(x)), 0.0f);
}

template <typename T>
__host__ __device__
inline std::enable_if_t<is_float_type<T>::value
                        && is_float_grad<T>::value, FloatGrad<float>>
roundf(T x) {
    return FloatGrad<float>(roundf(get_data(x)), 0.0f);
}

template <typename T>
__host__ __device__
inline std::enable_if_t<is_float_type<T>::value
                        && is_float_grad<T>::value, FloatGrad<float>>
truncf(T x) {
    return x >= 0.0f ? floorf(x) : ceilf(x);
}

template <typename T1, typename T2>
inline __host__ __device__
std::enable_if_t<is_float_type<T1>::value && is_float_type<T2>::value
                 && (is_float_grad<T1>::value || is_float_grad<T2>::value), FloatGrad<float>>
fmodf(const T1 x, const T2 y) {
    return x - y * truncf(x / y);
}

template <typename T>
__host__ __device__
inline std::enable_if_t<is_float_type<T>::value
                        && is_float_grad<T>::value, FloatGrad<float>>
expf(T x) {
    float data = expf(get_data(x));
    float grad = data * get_grad(x);
    return FloatGrad<float>(data, grad);
}

#endif // FLOAT_GRAD_FLOAT_H
