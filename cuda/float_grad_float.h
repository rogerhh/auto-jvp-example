#ifndef FLOAT_GRAD_FLOAT_H
#define FLOAT_GRAD_FLOAT_H

// template <> struct FloatGrad<float>;
// template <> struct FloatGradRef<float>;

template <typename T> struct is_float_type : std::false_type {};
template <> struct is_float_type<float> : std::true_type {};
template <> struct is_float_type<FloatGrad<float>> : std::true_type {};
template <> struct is_float_type<FloatGradRef<float>> : std::true_type {};

template <> struct is_float_type<const float> : std::true_type {};
template <> struct is_float_type<FloatGrad<const float>> : std::true_type {};
template <> struct is_float_type<FloatGradRef<const float>> : std::true_type {};

//////////////////////////////////////////////////////////////////////////////
/// Additional builtin functions
//////////////////////////////////////////////////////////////////////////////

template <typename T>
__host__ __device__
inline std::enable_if_t<is_float_type<T>::value, FloatGrad<float>>
operator+(T x) {
    return FloatGrad<float>(x);
}

template <typename T>
__host__ __device__
inline std::enable_if_t<is_float_type<T>::value, FloatGrad<float>>
operator-(T x) {
    return FloatGrad<float>(-get_data(x), -get_grad(x));
}

template <typename T>
__host__ __device__
inline std::enable_if_t<is_float_type<T>::value, FloatGrad<float>>
fabs(T x) {
    return x >= 0.0f ? x : -x;
}

template <typename T>
__host__ __device__
inline std::enable_if_t<is_float_type<T>::value, FloatGrad<float>>
sqrtf(T a) {
    float data = sqrtf(get_data(a));
    float grad = get_grad(a) / (2.0f * data);
    return FloatGrad<float>(data, grad);
}

template <typename T>
__host__ __device__
inline std::enable_if_t<is_float_type<T>::value, FloatGrad<float>>
floorf(T x) {
    return FloatGrad<float>(floorf(get_data(x)), 0.0f);
}

template <typename T>
__host__ __device__
inline std::enable_if_t<is_float_type<T>::value, FloatGrad<float>>
ceilf(T x) {
    return FloatGrad<float>(ceilf(get_data(x)), 0.0f);
}

template <typename T>
__host__ __device__
inline std::enable_if_t<is_float_type<T>::value, FloatGrad<float>>
roundf(T x) {
    return FloatGrad<float>(roundf(get_data(x)), 0.0f);
}

template <typename T>
__host__ __device__
inline std::enable_if_t<is_float_type<T>::value, FloatGrad<float>>
truncf(T x) {
    return x >= 0.0f ? floorf(x) : ceilf(x);
}

template <typename T1, typename T2>
inline __host__ __device__
std::enable_if_t<is_float_type<T1>::value && is_float_type<T2>::value, FloatGrad<float>>
fmodf(const T1 x, const T2 y) {
    return x - y * truncf(x / y);
}

#endif // FLOAT_GRAD_FLOAT_H
