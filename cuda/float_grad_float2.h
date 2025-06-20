#ifndef FLOAT_GRAD_FLOAT2_H
#define FLOAT_GRAD_FLOAT2_H

template <typename T>
using is_float2_type = std::is_same<std::decay_t<decltype(get_data(std::declval<T>()))>, float2>;

template <>
struct FloatGradRef<float2> : FloatGradRefBase<float2> {
    // All constructors
    template <typename... Args>
    __host__ __device__
    FloatGradRef(Args&&... args)
    : FloatGradRefBase<float2>(std::forward<Args>(args)...),
      x(&data().x, &grad().x), 
      y(&data().y, &grad().y) {}

    // All assignment operators
    template <typename OtherType>
    __host__ __device__
    FloatGradRef& operator=(const OtherType& other) {
        FloatGradRefBase<float2>::operator=(other);
        return *this;
    }

    FloatGradRef<float> x;
    FloatGradRef<float> y;
};

template <>
struct FloatGrad<float2> : FloatGradBase<float2> {
    // All constructors
    template <typename... Args>
    __host__ __device__
    FloatGrad(Args&&... args)
    : FloatGradBase<float2>(std::forward<Args>(args)...),
      x(&data().x, &grad().x), 
      y(&data().y, &grad().y) {}

    // All assignment operators
    template <typename OtherType>
    __host__ __device__
    FloatGrad& operator=(const OtherType& other) {
        FloatGradBase<float2>::operator=(other);
        return *this;
    }

    FloatGradRef<float> x;
    FloatGradRef<float> y;
};

template <>
struct FloatGradRef<const float2> : FloatGradRefBase<const float2> {
    template <typename... Args>
    __host__ __device__
    FloatGradRef(Args&&... args)
    : FloatGradRefBase<const float2>(std::forward<Args>(args)...),
      x(&data().x, &grad().x), 
      y(&data().y, &grad().y) {}

    FloatGradRef<const float> x;
    FloatGradRef<const float> y;
};

template <>
struct FloatGrad<const float2> : FloatGradBase<const float2> {
    template <typename... Args>
    __host__ __device__
    FloatGrad(Args&&... args)
    : FloatGradBase<const float2>(std::forward<Args>(args)...),
      x(&data().x, &grad().x), 
      y(&data().y, &grad().y) {}

    FloatGradRef<const float> x;
    FloatGradRef<const float> y;
};

template <typename T1, typename T2,
          typename = std::enable_if_t<is_float_type<T1>::value
                                      && is_float_type<T2>::value>>
__host__ __device__
inline FloatGrad<float2> make_float2(const T1& x, const T2& y) {
    return FloatGrad<float2>(float2{get_data(x), get_data(y)}, 
                             float2{get_grad(x), get_grad(y)});
}

#endif // FLOAT_GRAD_FLOAT2_H
