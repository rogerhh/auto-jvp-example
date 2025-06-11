#ifndef FLOAT_GRAD_FLOAT2_H
#define FLOAT_GRAD_FLOAT2_H

template <> struct FloatGrad<float2>;
template <> struct FloatGradRef<float2>;

template <typename T> struct is_float2_type : std::false_type {};
template <> struct is_float2_type<float2> : std::true_type {};
template <> struct is_float2_type<FloatGrad<float2>> : std::true_type {};
template <> struct is_float2_type<FloatGradRef<float2>> : std::true_type {};

template <>
struct FloatGradRef<float2> {
    // Delete default constructor to prevent uninitialized usage
    FloatGradRef() = delete;

    __host__ __device__
    FloatGradRef(float2* data_ptr, float2* grad_ptr);

    __host__ __device__
    FloatGradRef(const FloatGradRef& other);

    __host__ __device__
    FloatGradRef& operator=(const FloatGradRef& other) {
        if (this != &other) {
            *data_ptr_ = *other.data_ptr_;
            *grad_ptr_ = *other.grad_ptr_;
        }
        return *this;
    }

    __host__ __device__
    FloatGradRef& operator=(const FloatGrad<float2>& other);

    __host__ __device__
    float2& data() {
        return *data_ptr_;
    }
    __host__ __device__
    const float2& data() const {
        return *data_ptr_;
    }
    __host__ __device__
    float2& grad() {
        return *grad_ptr_;
    }
    __host__ __device__
    const float2& grad() const {
        return *grad_ptr_;
    }

    float2* data_ptr_;
    float2* grad_ptr_;

    FloatGradRef<float> x;
    FloatGradRef<float> y;
};

template <>
struct FloatGrad<float2> {
    __host__ __device__
    FloatGrad(const float2& data);

    __host__ __device__
    FloatGrad(const float2& data, const float2& grad);

    __host__ __device__
    FloatGrad(const FloatGrad& other);

    __host__ __device__
    FloatGrad(const FloatGradRef<float2>& ref);

    __host__ __device__
    FloatGrad& operator=(const FloatGrad& other) {
        if (this != &other) {
            data_ = other.data_;
            grad_ = other.grad_;
        }
        return *this;
    }

    __host__ __device__
    FloatGrad& operator=(const FloatGradRef<float2>& ref) {
        data_ = *ref.data_ptr_;
        grad_ = *ref.grad_ptr_;
        return *this;
    }

    __host__ __device__
    float2& data() {
        return data_;
    }
    __host__ __device__
    const float2& data() const {
        return data_;
    }
    __host__ __device__
    float2& grad() {
        return grad_;
    }
    __host__ __device__
    const float2& grad() const {
        return grad_;
    }

    float2 data_;
    float2 grad_;

    FloatGradRef<float> x;
    FloatGradRef<float> y;
};

__host__ __device__
inline FloatGradRef<float2>::FloatGradRef(float2* data_ptr, float2* grad_ptr)
    : data_ptr_(data_ptr), grad_ptr_(grad_ptr),
      x(&data_ptr_->x, &grad_ptr_->x),
      y(&data_ptr_->y, &grad_ptr_->y) {}

__host__ __device__
inline FloatGradRef<float2>::FloatGradRef(const FloatGradRef<float2>& other)
    : data_ptr_(other.data_ptr_), grad_ptr_(other.grad_ptr_),
      x(&data_ptr_->x, &grad_ptr_->x),
      y(&data_ptr_->y, &grad_ptr_->y) {}

__host__ __device__
inline FloatGradRef<float2>& FloatGradRef<float2>::operator=(const FloatGrad<float2>& other) {
    *data_ptr_ = other.data_;
    *grad_ptr_ = other.grad_;
    return *this;
}

__host__ __device__
inline FloatGrad<float2>::FloatGrad(const float2& data)
    : data_(data), grad_(float2{0.0f, 0.0f}),
      x(&data_.x, &grad_.x), y(&data_.y, &grad_.y) {}

__host__ __device__
inline FloatGrad<float2>::FloatGrad(const float2& data, const float2& grad)
    : data_(data), grad_(grad),
      x(&data_.x, &grad_.x), y(&data_.y, &grad_.y) {}

__host__ __device__
inline FloatGrad<float2>::FloatGrad(const FloatGrad<float2>& other)
    : data_(other.data_), grad_(other.grad_),
      x(&data_.x, &grad_.x), y(&data_.y, &grad_.y) {}

__host__ __device__
inline FloatGrad<float2>::FloatGrad(const FloatGradRef<float2>& ref)
    : data_(*ref.data_ptr_), grad_(*ref.grad_ptr_),
      x(&data_.x, &grad_.x), y(&data_.y, &grad_.y) {}

template <typename T1, typename T2,
          typename = std::enable_if_t<is_float_type<T1>::value
                                      && is_float_type<T2>::value>>
__host__ __device__
inline FloatGrad<float2> make_float2(const T1& x, const T2& y) {
    return FloatGrad<float2>(float2{get_data(x), get_data(y)}, 
                             float2{get_grad(x), get_grad(y)});
}

#endif // FLOAT_GRAD_FLOAT2_H
