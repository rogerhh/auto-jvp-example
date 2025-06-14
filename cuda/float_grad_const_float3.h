#ifndef FLOAT_GRAD_CONST_FLOAT3_H
#define FLOAT_GRAD_CONST_FLOAT3_H

template <> struct FloatGrad<const float3>;
template <> struct FloatGradRef<const float3>;

template <> struct is_float3_type<const float3> : std::true_type {};
template <> struct is_float3_type<FloatGrad<const float3>> : std::true_type {};
template <> struct is_float3_type<FloatGradRef<const float3>> : std::true_type {};

template <>
struct FloatGradRef<const float3> {
    // Delete default constructor to prevent uninitialized usage
    FloatGradRef() = delete;

    __host__ __device__
    FloatGradRef(const float3* data_ptr, const float3* grad_ptr);

    // Constructing from reference type
    template <typename U = const float3, 
              typename = std::enable_if_t<std::is_same_v<std::remove_const_t<U>, 
                                                         std::remove_const_t<float3>>>>
    __host__ __device__
    FloatGradRef(const FloatGradRef<U>& other);

    __host__ __device__
    const float3& data() const {
        return *data_ptr_;
    }
    __host__ __device__
    const float3& grad() const {
        return *grad_ptr_;
    }

    const float3* data_ptr_;
    const float3* grad_ptr_;

    FloatGradRef<const float> x;
    FloatGradRef<const float> y;
    FloatGradRef<const float> z;
};

template <>
struct FloatGrad<const float3> {
    __host__ __device__
    FloatGrad(const float3& data);

    __host__ __device__
    FloatGrad(const float3& data, const float3& grad);

    template <typename U = const float3, 
              typename = std::enable_if_t<std::is_same_v<std::remove_const_t<U>, 
                                                         std::remove_const_t<float3>>>>
    __host__ __device__
    FloatGrad(const FloatGrad<U>& other);

    template <typename U = const float3, 
              typename = std::enable_if_t<std::is_same_v<std::remove_const_t<U>, 
                                                         std::remove_const_t<float3>>>>
    __host__ __device__
    FloatGrad(const FloatGradRef<U>& ref);

    __host__ __device__
    const float3& data() const {
        return data_;
    }
    __host__ __device__
    const float3& grad() const {
        return grad_;
    }

    const float3 data_;
    const float3 grad_;

    FloatGradRef<const float> x;
    FloatGradRef<const float> y;
    FloatGradRef<const float> z;
};

__host__ __device__
inline FloatGradRef<const float3>::FloatGradRef(const float3* data_ptr, const float3* grad_ptr)
    : data_ptr_(data_ptr), grad_ptr_(grad_ptr),
      x(&data_ptr_->x, &grad_ptr_->x),
      y(&data_ptr_->y, &grad_ptr_->y),
      z(&data_ptr_->z, &grad_ptr_->z) {}

template <typename U, typename>
__host__ __device__
inline FloatGradRef<const float3>::FloatGradRef(const FloatGradRef<U>& other)
    : data_ptr_(other.data_ptr_), grad_ptr_(other.grad_ptr_),
      x(&data_ptr_->x, &grad_ptr_->x),
      y(&data_ptr_->y, &grad_ptr_->y),
      z(&data_ptr_->z, &grad_ptr_->z) {}

__host__ __device__
inline FloatGrad<const float3>::FloatGrad(const float3& data)
    : data_(data), grad_(float3{0.0f, 0.0f}),
      x(&data_.x, &grad_.x), 
      y(&data_.y, &grad_.y),
      z(&data_.z, &grad_.z) {}

__host__ __device__
inline FloatGrad<const float3>::FloatGrad(const float3& data, const float3& grad)
    : data_(data), grad_(grad),
      x(&data_.x, &grad_.x), 
      y(&data_.y, &grad_.y),
      z(&data_.z, &grad_.z) {}

template <typename U, typename>
__host__ __device__
inline FloatGrad<const float3>::FloatGrad(const FloatGrad<U>& other)
    : data_(other.data_), grad_(other.grad_),
      x(&data_.x, &grad_.x), 
      y(&data_.y, &grad_.y),
      z(&data_.z, &grad_.z) {}

template <typename U, typename>
__host__ __device__
inline FloatGrad<const float3>::FloatGrad(const FloatGradRef<U>& ref)
    : data_(*ref.data_ptr_), grad_(*ref.grad_ptr_),
      x(&data_.x, &grad_.x), 
      y(&data_.y, &grad_.y),
      z(&data_.z, &grad_.z) {}

#endif // FLOAT_GRAD_CONST_FLOAT3_H
