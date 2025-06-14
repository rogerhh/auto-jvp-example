#ifndef FLOAT_GRAD_CONST_FLOAT4_H
#define FLOAT_GRAD_CONST_FLOAT4_H

template <> struct FloatGrad<const float4>;
template <> struct FloatGradRef<const float4>;

template <> struct is_float4_type<const float4> : std::true_type {};
template <> struct is_float4_type<FloatGrad<const float4>> : std::true_type {};
template <> struct is_float4_type<FloatGradRef<const float4>> : std::true_type {};

template <>
struct FloatGradRef<const float4> {
    // Delete default constructor to prevent uninitialized usage
    FloatGradRef() = delete;

    __host__ __device__
    FloatGradRef(const float4* data_ptr, const float4* grad_ptr);

    // Constructing from reference type
    template <typename U = const float4, 
              typename = std::enable_if_t<std::is_same_v<std::remove_const_t<U>, 
                                                         std::remove_const_t<float4>>>>
    __host__ __device__
    FloatGradRef(const FloatGradRef<U>& other);

    __host__ __device__
    const float4& data() const {
        return *data_ptr_;
    }
    __host__ __device__
    const float4& grad() const {
        return *grad_ptr_;
    }

    const float4* data_ptr_;
    const float4* grad_ptr_;

    FloatGradRef<const float> x;
    FloatGradRef<const float> y;
    FloatGradRef<const float> z;
    FloatGradRef<const float> w;
};

template <>
struct FloatGrad<const float4> {
    __host__ __device__
    FloatGrad(const float4& data);

    __host__ __device__
    FloatGrad(const float4& data, const float4& grad);

    template <typename U = const float4, 
              typename = std::enable_if_t<std::is_same_v<std::remove_const_t<U>, 
                                                         std::remove_const_t<float4>>>>
    __host__ __device__
    FloatGrad(const FloatGrad<U>& other);

    template <typename U = const float4, 
              typename = std::enable_if_t<std::is_same_v<std::remove_const_t<U>, 
                                                         std::remove_const_t<float4>>>>
    __host__ __device__
    FloatGrad(const FloatGradRef<U>& ref);

    __host__ __device__
    const float4& data() const {
        return data_;
    }
    __host__ __device__
    const float4& grad() const {
        return grad_;
    }

    const float4 data_;
    const float4 grad_;

    FloatGradRef<const float> x;
    FloatGradRef<const float> y;
    FloatGradRef<const float> z;
    FloatGradRef<const float> w;
};

__host__ __device__
inline FloatGradRef<const float4>::FloatGradRef(const float4* data_ptr, const float4* grad_ptr)
    : data_ptr_(data_ptr), grad_ptr_(grad_ptr),
      x(&data_ptr_->x, &grad_ptr_->x),
      y(&data_ptr_->y, &grad_ptr_->y),
      z(&data_ptr_->z, &grad_ptr_->z),
      w(&data_ptr_->w, &grad_ptr_->w) {}

template <typename U, typename>
__host__ __device__
inline FloatGradRef<const float4>::FloatGradRef(const FloatGradRef<U>& other)
    : data_ptr_(other.data_ptr_), grad_ptr_(other.grad_ptr_),
      x(&data_ptr_->x, &grad_ptr_->x),
      y(&data_ptr_->y, &grad_ptr_->y),
      z(&data_ptr_->z, &grad_ptr_->z),
      w(&data_ptr_->w, &grad_ptr_->w) {}

__host__ __device__
inline FloatGrad<const float4>::FloatGrad(const float4& data)
    : data_(data), grad_(float4{0.0f, 0.0f}),
      x(&data_.x, &grad_.x), 
      y(&data_.y, &grad_.y),
      z(&data_.z, &grad_.z),
      w(&data_.w, &grad_.w) {}

__host__ __device__
inline FloatGrad<const float4>::FloatGrad(const float4& data, const float4& grad)
    : data_(data), grad_(grad),
      x(&data_.x, &grad_.x), 
      y(&data_.y, &grad_.y),
      z(&data_.z, &grad_.z),
      w(&data_.w, &grad_.w) {}

template <typename U, typename>
__host__ __device__
inline FloatGrad<const float4>::FloatGrad(const FloatGrad<U>& other)
    : data_(other.data_), grad_(other.grad_),
      x(&data_.x, &grad_.x), 
      y(&data_.y, &grad_.y),
      z(&data_.z, &grad_.z),
      w(&data_.w, &grad_.w) {}

template <typename U, typename>
__host__ __device__
inline FloatGrad<const float4>::FloatGrad(const FloatGradRef<U>& ref)
    : data_(*ref.data_ptr_), grad_(*ref.grad_ptr_),
      x(&data_.x, &grad_.x), 
      y(&data_.y, &grad_.y),
      z(&data_.z, &grad_.z),
      w(&data_.w, &grad_.w) {}

#endif // FLOAT_GRAD_CONST_FLOAT4_H
