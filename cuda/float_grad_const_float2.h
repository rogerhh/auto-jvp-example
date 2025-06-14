#ifndef FLOAT_GRAD_CONST_FLOAT2_H
#define FLOAT_GRAD_CONST_FLOAT2_H

template <> struct FloatGrad<const float2>;
template <> struct FloatGradRef<const float2>;

template <> struct is_float2_type<const float2> : std::true_type {};
template <> struct is_float2_type<FloatGrad<const float2>> : std::true_type {};
template <> struct is_float2_type<FloatGradRef<const float2>> : std::true_type {};

template <>
struct FloatGradRef<const float2> {
    // Delete default constructor to prevent uninitialized usage
    FloatGradRef() = delete;

    __host__ __device__
    FloatGradRef(const float2* data_ptr, const float2* grad_ptr);

    // Constructing from reference type
    template <typename U = const float2, 
              typename = std::enable_if_t<std::is_same_v<std::remove_const_t<U>, 
                                                         std::remove_const_t<float2>>>>
    __host__ __device__
    FloatGradRef(const FloatGradRef<U>& other);

    __host__ __device__
    const float2& data() const {
        return *data_ptr_;
    }
    __host__ __device__
    const float2& grad() const {
        return *grad_ptr_;
    }

    const float2* data_ptr_;
    const float2* grad_ptr_;

    FloatGradRef<const float> x;
    FloatGradRef<const float> y;
};

template <>
struct FloatGrad<const float2> {
    __host__ __device__
    FloatGrad(const float2& data);

    __host__ __device__
    FloatGrad(const float2& data, const float2& grad);

    template <typename U = const float2, 
              typename = std::enable_if_t<std::is_same_v<std::remove_const_t<U>, 
                                                         std::remove_const_t<float2>>>>
    __host__ __device__
    FloatGrad(const FloatGrad<U>& other);

    template <typename U = const float2, 
              typename = std::enable_if_t<std::is_same_v<std::remove_const_t<U>, 
                                                         std::remove_const_t<float2>>>>
    __host__ __device__
    FloatGrad(const FloatGradRef<U>& ref);

    __host__ __device__
    const float2& data() const {
        return data_;
    }
    __host__ __device__
    const float2& grad() const {
        return grad_;
    }

    const float2 data_;
    const float2 grad_;

    FloatGradRef<const float> x;
    FloatGradRef<const float> y;
};

__host__ __device__
inline FloatGradRef<const float2>::FloatGradRef(const float2* data_ptr, const float2* grad_ptr)
    : data_ptr_(data_ptr), grad_ptr_(grad_ptr),
      x(&data_ptr_->x, &grad_ptr_->x),
      y(&data_ptr_->y, &grad_ptr_->y) {}

template <typename U, typename>
__host__ __device__
inline FloatGradRef<const float2>::FloatGradRef(const FloatGradRef<U>& other)
    : data_ptr_(other.data_ptr_), grad_ptr_(other.grad_ptr_),
      x(&data_ptr_->x, &grad_ptr_->x),
      y(&data_ptr_->y, &grad_ptr_->y) {}

__host__ __device__
inline FloatGrad<const float2>::FloatGrad(const float2& data)
    : data_(data), grad_(float2{0.0f, 0.0f}),
      x(&data_.x, &grad_.x), y(&data_.y, &grad_.y) {}

__host__ __device__
inline FloatGrad<const float2>::FloatGrad(const float2& data, const float2& grad)
    : data_(data), grad_(grad),
      x(&data_.x, &grad_.x), y(&data_.y, &grad_.y) {}

template <typename U, typename>
__host__ __device__
inline FloatGrad<const float2>::FloatGrad(const FloatGrad<U>& other)
    : data_(other.data_), grad_(other.grad_),
      x(&data_.x, &grad_.x), y(&data_.y, &grad_.y) {}

template <typename U, typename>
__host__ __device__
inline FloatGrad<const float2>::FloatGrad(const FloatGradRef<U>& ref)
    : data_(*ref.data_ptr_), grad_(*ref.grad_ptr_),
      x(&data_.x, &grad_.x), y(&data_.y, &grad_.y) {}

#endif // FLOAT_GRAD_CONST_FLOAT2_H
