#ifndef FLOAT_GRAD_FLOAT4_H
#define FLOAT_GRAD_FLOAT4_H

template <typename T>
using is_float4_type = std::is_same<std::decay_t<decltype(get_data(std::declval<T>()))>, float4>;

template <>
struct FloatGradRef<float4> : FloatGradRefBase<float4> {
    // All constructors
    template <typename... Args>
    __host__ __device__
    FloatGradRef(Args&&... args)
    : FloatGradRefBase<float4>(std::forward<Args>(args)...),
      x(&data().x, &grad().x), 
      y(&data().y, &grad().y),
      z(&data().z, &grad().z),
      w(&data().w, &grad().w) {}

    // All assignment operators
    template <typename OtherType>
    __host__ __device__
    FloatGradRef& operator=(const OtherType& other) {
        FloatGradRefBase<float4>::operator=(other);
        return *this;
    }

    FloatGradRef<float> x;
    FloatGradRef<float> y;
    FloatGradRef<float> z;
    FloatGradRef<float> w;
};

template <>
struct FloatGrad<float4> : FloatGradBase<float4> {
    // All constructors
    template <typename... Args>
    __host__ __device__
    FloatGrad(Args&&... args)
    : FloatGradBase<float4>(std::forward<Args>(args)...),
      x(&data().x, &grad().x), 
      y(&data().y, &grad().y),
      z(&data().z, &grad().z),
      w(&data().w, &grad().w) {}

    // All assignment operators
    template <typename OtherType>
    __host__ __device__
    FloatGrad& operator=(const OtherType& other) {
        FloatGradBase<float4>::operator=(other);
        return *this;
    }

    FloatGradRef<float> x;
    FloatGradRef<float> y;
    FloatGradRef<float> z;
    FloatGradRef<float> w;
};

template <>
struct FloatGradRef<const float4> : FloatGradRefBase<const float4> {
    template <typename... Args>
    __host__ __device__
    FloatGradRef(Args&&... args)
    : FloatGradRefBase<const float4>(std::forward<Args>(args)...),
      x(&data().x, &grad().x), 
      y(&data().y, &grad().y),
      z(&data().z, &grad().z),
      w(&data().w, &grad().w) {}

    FloatGradRef<const float> x;
    FloatGradRef<const float> y;
    FloatGradRef<const float> z;
    FloatGradRef<const float> w;
};

template <>
struct FloatGrad<const float4> : FloatGradBase<const float4> {
    template <typename... Args>
    __host__ __device__
    FloatGrad(Args&&... args)
    : FloatGradBase<const float4>(std::forward<Args>(args)...),
      x(&data().x, &grad().x), 
      y(&data().y, &grad().y),
      z(&data().z, &grad().z),
      w(&data().w, &grad().w) {}

    FloatGradRef<const float> x;
    FloatGradRef<const float> y;
    FloatGradRef<const float> z;
    FloatGradRef<const float> w;
};

template <typename T1, typename T2, typename T3, typename T4,
          typename = std::enable_if_t<is_float_type<T1>::value
                                      && is_float_type<T2>::value
                                      && is_float_type<T3>::value
                                      && is_float_type<T4>::value>>
__host__ __device__
inline FloatGrad<float4> make_float4(const T1& x, const T2& y, const T3& z, const T4& w) {
    return FloatGrad<float4>(float4{get_data(x), get_data(y), get_data(z), get_data(w)}, 
                             float4{get_grad(x), get_grad(y), get_grad(z), get_grad(w)});
}

#endif // FLOAT_GRAD_FLOAT4_H
