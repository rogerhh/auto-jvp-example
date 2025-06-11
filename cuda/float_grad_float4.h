#ifndef FLOAT_GRAD_FLOAT4_H
#define FLOAT_GRAD_FLOAT4_H

template <> struct FloatGrad<float4>;
template <> struct FloatGradRef<float4>;

template <typename T> struct is_float4_type : std::false_type {};
template <> struct is_float4_type<float4> : std::true_type {};
template <> struct is_float4_type<FloatGrad<float4>> : std::true_type {};
template <> struct is_float4_type<FloatGradRef<float4>> : std::true_type {};

template <>
struct FloatGradRef<float4> {
    // Delete default constructor to prevent uninitialized usage
    FloatGradRef() = delete;

    __host__ __device__
    FloatGradRef(float4* data_ptr, float4* grad_ptr);

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
    FloatGradRef& operator=(const FloatGrad<float4>& other);

    __host__ __device__
    float4& data() {
        return *data_ptr_;
    }
    __host__ __device__
    const float4& data() const {
        return *data_ptr_;
    }
    __host__ __device__
    float4& grad() {
        return *grad_ptr_;
    }
    __host__ __device__
    const float4& grad() const {
        return *grad_ptr_;
    }

    float4* data_ptr_;
    float4* grad_ptr_;

    FloatGradRef<float> x;
    FloatGradRef<float> y;
    FloatGradRef<float> z;
    FloatGradRef<float> w;
};

template <>
struct FloatGrad<float4> {
    __host__ __device__
    FloatGrad(const float4& data);

    __host__ __device__
    FloatGrad(const float4& data, const float4& grad);

    __host__ __device__
    FloatGrad(const FloatGrad& other);

    __host__ __device__
    FloatGrad(const FloatGradRef<float4>& ref);

    __host__ __device__
    FloatGrad& operator=(const FloatGrad& other) {
        if (this != &other) {
            data_ = other.data_;
            grad_ = other.grad_;
        }
        return *this;
    }

    __host__ __device__
    FloatGrad& operator=(const FloatGradRef<float4>& ref) {
        data_ = *ref.data_ptr_;
        grad_ = *ref.grad_ptr_;
        return *this;
    }

    __host__ __device__
    float4& data() {
        return data_;
    }
    __host__ __device__
    const float4& data() const {
        return data_;
    }
    __host__ __device__
    float4& grad() {
        return grad_;
    }
    __host__ __device__
    const float4& grad() const {
        return grad_;
    }

    float4 data_;
    float4 grad_;

    FloatGradRef<float> x;
    FloatGradRef<float> y;
    FloatGradRef<float> z;
    FloatGradRef<float> w;
};

__host__ __device__
inline FloatGradRef<float4>::FloatGradRef(float4* data_ptr, float4* grad_ptr)
    : data_ptr_(data_ptr), grad_ptr_(grad_ptr),
      x(&data_ptr_->x, &grad_ptr_->x),
      y(&data_ptr_->y, &grad_ptr_->y),
      z(&data_ptr_->z, &grad_ptr_->z),
      w(&data_ptr_->w, &grad_ptr_->w) {}

__host__ __device__
inline FloatGradRef<float4>::FloatGradRef(const FloatGradRef<float4>& other)
    : data_ptr_(other.data_ptr_), grad_ptr_(other.grad_ptr_),
      x(&data_ptr_->x, &grad_ptr_->x),
      y(&data_ptr_->y, &grad_ptr_->y),
      z(&data_ptr_->z, &grad_ptr_->z),
      w(&data_ptr_->w, &grad_ptr_->w) {}

__host__ __device__
inline FloatGradRef<float4>& FloatGradRef<float4>::operator=(const FloatGrad<float4>& other) {
    *data_ptr_ = other.data_;
    *grad_ptr_ = other.grad_;
    return *this;
}

__host__ __device__
inline FloatGrad<float4>::FloatGrad(const float4& data)
    : data_(data), grad_(float4{0.0f, 0.0f, 0.0f, 0.0f}),
      x(&data_.x, &grad_.x), y(&data_.y, &grad_.y), 
      z(&data_.z, &grad_.z), w(&data_.w, &grad_.w) {}

__host__ __device__
inline FloatGrad<float4>::FloatGrad(const float4& data, const float4& grad)
    : data_(data), grad_(grad),
      x(&data_.x, &grad_.x), y(&data_.y, &grad_.y), 
      z(&data_.z, &grad_.z), w(&data_.w, &grad_.w) {}

__host__ __device__
inline FloatGrad<float4>::FloatGrad(const FloatGrad<float4>& other)
    : data_(other.data_), grad_(other.grad_),
      x(&data_.x, &grad_.x), y(&data_.y, &grad_.y), 
      z(&data_.z, &grad_.z), w(&data_.w, &grad_.w) {}

__host__ __device__
inline FloatGrad<float4>::FloatGrad(const FloatGradRef<float4>& ref)
    : data_(*ref.data_ptr_), grad_(*ref.grad_ptr_),
      x(&data_.x, &grad_.x), y(&data_.y, &grad_.y), 
      z(&data_.z, &grad_.z), w(&data_.w, &grad_.w) {}

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
