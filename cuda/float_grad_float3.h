#ifndef FLOAT_GRAD_FLOAT3_H
#define FLOAT_GRAD_FLOAT3_H

template <> struct FloatGrad<float3>;
template <> struct FloatGradRef<float3>;

template <typename T> struct is_float3_type : std::false_type {};
template <> struct is_float3_type<float3> : std::true_type {};
template <> struct is_float3_type<FloatGrad<float3>> : std::true_type {};
template <> struct is_float3_type<FloatGradRef<float3>> : std::true_type {};

template <>
struct FloatGradRef<float3> {
    // Delete default constructor to prevent uninitialized usage
    FloatGradRef() = delete;

    __host__ __device__
    FloatGradRef(float3* data_ptr, float3* grad_ptr);

    // Constructing from reference type
    template <typename U = float3, 
              typename = std::enable_if_t<std::is_same_v<std::remove_const_t<U>, 
                                                         std::remove_const_t<float3>>>>
    __host__ __device__
    FloatGradRef(const FloatGradRef<U>& other);

    template <typename OtherType>
    __host__ __device__
    FloatGradRef& operator=(const OtherType& other);

    __host__ __device__
    float3& data() {
        return *data_ptr_;
    }
    __host__ __device__
    const float3& data() const {
        return *data_ptr_;
    }
    __host__ __device__
    float3& grad() {
        return *grad_ptr_;
    }
    __host__ __device__
    const float3& grad() const {
        return *grad_ptr_;
    }

    float3* data_ptr_;
    float3* grad_ptr_;

    FloatGradRef<float> x;
    FloatGradRef<float> y;
    FloatGradRef<float> z;
};

template <>
struct FloatGrad<float3> {
    __host__ __device__
    FloatGrad(const float3& data);

    __host__ __device__
    FloatGrad(const float3& data, const float3& grad);

    template <typename U = float3, 
              typename = std::enable_if_t<std::is_same_v<std::remove_const_t<U>, 
                                                         std::remove_const_t<float3>>>>
    __host__ __device__
    FloatGrad(const FloatGrad<U>& other);

    template <typename U = float3, 
              typename = std::enable_if_t<std::is_same_v<std::remove_const_t<U>, 
                                                         std::remove_const_t<float3>>>>
    __host__ __device__
    FloatGrad(const FloatGradRef<U>& ref);

    template <typename OtherType>
    __host__ __device__
    FloatGrad& operator=(const OtherType& other);

    __host__ __device__
    float3& data() {
        return data_;
    }
    __host__ __device__
    const float3& data() const {
        return data_;
    }
    __host__ __device__
    float3& grad() {
        return grad_;
    }
    __host__ __device__
    const float3& grad() const {
        return grad_;
    }

    float3 data_;
    float3 grad_;

    FloatGradRef<float> x;
    FloatGradRef<float> y;
    FloatGradRef<float> z;
};

__host__ __device__
inline FloatGradRef<float3>::FloatGradRef(float3* data_ptr, float3* grad_ptr)
    : data_ptr_(data_ptr), grad_ptr_(grad_ptr),
      x(&data_ptr_->x, &grad_ptr_->x),
      y(&data_ptr_->y, &grad_ptr_->y),
      z(&data_ptr_->z, &grad_ptr_->z) {}

template <typename U, typename>
__host__ __device__
inline FloatGradRef<float3>::FloatGradRef(const FloatGradRef<U>& other)
    : data_ptr_(other.data_ptr_), grad_ptr_(other.grad_ptr_),
      x(&data_ptr_->x, &grad_ptr_->x),
      y(&data_ptr_->y, &grad_ptr_->y),
      z(&data_ptr_->z, &grad_ptr_->z) {}

template <typename OtherType>
__host__ __device__
inline FloatGradRef<float3>& FloatGradRef<float3>::operator=(const OtherType& other) {
    *data_ptr_ = get_data(other);
    *grad_ptr_ = get_grad(other);
    return *this;
}

__host__ __device__
inline FloatGrad<float3>::FloatGrad(const float3& data)
    : data_(data), grad_(float3{0.0f, 0.0f}),
      x(&data_.x, &grad_.x), 
      y(&data_.y, &grad_.y),
      z(&data_.z, &grad_.z) {}

__host__ __device__
inline FloatGrad<float3>::FloatGrad(const float3& data, const float3& grad)
    : data_(data), grad_(grad),
      x(&data_.x, &grad_.x), 
      y(&data_.y, &grad_.y),
      z(&data_.z, &grad_.z) {}

template <typename U, typename>
__host__ __device__
inline FloatGrad<float3>::FloatGrad(const FloatGrad<U>& other)
    : data_(other.data_), grad_(other.grad_),
      x(&data_.x, &grad_.x), 
      y(&data_.y, &grad_.y),
      z(&data_.z, &grad_.z) {}

template <typename U, typename>
__host__ __device__
inline FloatGrad<float3>::FloatGrad(const FloatGradRef<U>& ref)
    : data_(*ref.data_ptr_), grad_(*ref.grad_ptr_),
      x(&data_.x, &grad_.x), 
      y(&data_.y, &grad_.y),
      z(&data_.z, &grad_.z) {}

template <typename OtherType>
__host__ __device__
inline FloatGrad<float3>& FloatGrad<float3>::operator=(const OtherType& other) {
    this->data() = get_data(other);
    this->grad() = get_grad(other);
    return *this;
}

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
