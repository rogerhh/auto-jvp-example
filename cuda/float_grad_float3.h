#ifndef FLOAT_GRAD_FLOAT3_H
#define FLOAT_GRAD_FLOAT3_H

template <typename T>
using is_float3_type = std::is_same<std::decay_t<decltype(get_data(std::declval<T>()))>, float3>;

template <>
struct FloatGradRef<float3> : FloatGradRefBase<float3> {
    // All constructors
    template <typename... Args>
    __host__ __device__
    FloatGradRef(Args&&... args)
    : FloatGradRefBase<float3>(std::forward<Args>(args)...),
      x(&data().x, &grad().x), 
      y(&data().y, &grad().y),
      z(&data().z, &grad().z) {}

    // All assignment operators
    template <typename OtherType>
    __host__ __device__
    FloatGradRef& operator=(const OtherType& other) {
        FloatGradRefBase<float3>::operator=(other);
        return *this;
    }

    FloatGradRef<float> x;
    FloatGradRef<float> y;
    FloatGradRef<float> z;
};

template <>
struct FloatGrad<float3> : FloatGradBase<float3> {
    // All constructors
    template <typename... Args>
    __host__ __device__
    FloatGrad(Args&&... args)
    : FloatGradBase<float3>(std::forward<Args>(args)...),
      x(&data().x, &grad().x), 
      y(&data().y, &grad().y),
      z(&data().z, &grad().z) {}

    // All assignment operators
    template <typename OtherType>
    __host__ __device__
    FloatGrad& operator=(const OtherType& other) {
        FloatGradBase<float3>::operator=(other);
        return *this;
    }

    FloatGradRef<float> x;
    FloatGradRef<float> y;
    FloatGradRef<float> z;
};

template <>
struct FloatGradRef<const float3> : FloatGradRefBase<const float3> {
    template <typename... Args>
    __host__ __device__
    FloatGradRef(Args&&... args)
    : FloatGradRefBase<const float3>(std::forward<Args>(args)...),
      x(&data().x, &grad().x), 
      y(&data().y, &grad().y),
      z(&data().z, &grad().z) {}

    FloatGradRef<const float> x;
    FloatGradRef<const float> y;
    FloatGradRef<const float> z;
};

template <>
struct FloatGrad<const float3> : FloatGradBase<const float3> {
    template <typename... Args>
    __host__ __device__
    FloatGrad(Args&&... args)
    : FloatGradBase<const float3>(std::forward<Args>(args)...),
      x(&data().x, &grad().x), 
      y(&data().y, &grad().y),
      z(&data().z, &grad().z) {}

    FloatGradRef<const float> x;
    FloatGradRef<const float> y;
    FloatGradRef<const float> z;
};

template <typename T1, typename T2, typename T3,
          typename = std::enable_if_t<is_float_type<T1>::value
                                      && is_float_type<T2>::value
                                      && is_float_type<T3>::value>>
__host__ __device__
inline FloatGrad<float3> make_float3(const T1& x, const T2& y, const T3& z) {
    return FloatGrad<float3>(float3{get_data(x), get_data(y), get_data(z)}, 
                             float3{get_grad(x), get_grad(y), get_grad(z)});
}

#endif // FLOAT_GRAD_FLOAT3_H
