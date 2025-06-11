#ifndef FLOAT_GRAD_FLOAT_H
#define FLOAT_GRAD_FLOAT_H

template <> struct FloatGrad<float>;
template <> struct FloatGradRef<float>;

template <typename T> struct is_float_type : std::false_type {};
template <> struct is_float_type<float> : std::true_type {};
template <> struct is_float_type<FloatGrad<float>> : std::true_type {};
template <> struct is_float_type<FloatGradRef<float>> : std::true_type {};

template <>
struct FloatGradRef<float> {
    // Delete default constructor to prevent uninitialized usage
    FloatGradRef() = delete;

    __host__ __device__
    FloatGradRef(float* data_ptr, float* grad_ptr);

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
    FloatGradRef& operator=(const FloatGrad<float>& other);

    __host__ __device__
    float& data() {
        return *data_ptr_;
    }
    __host__ __device__
    const float& data() const {
        return *data_ptr_;
    }
    __host__ __device__
    float& grad() {
        return *grad_ptr_;
    }
    __host__ __device__
    const float& grad() const {
        return *grad_ptr_;
    }

    float* data_ptr_;
    float* grad_ptr_;
};

template <>
struct FloatGrad<float> {
    __host__ __device__
    FloatGrad(const float& data);

    __host__ __device__
    FloatGrad(const float& data, const float& grad);

    __host__ __device__
    FloatGrad(const FloatGrad& other);

    __host__ __device__
    FloatGrad(const FloatGradRef<float>& ref);

    __host__ __device__
    FloatGrad& operator=(const FloatGrad& other) {
        if (this != &other) {
            data_ = other.data_;
            grad_ = other.grad_;
        }
        return *this;
    }

    __host__ __device__
    FloatGrad& operator=(const FloatGradRef<float>& ref) {
        data_ = *ref.data_ptr_;
        grad_ = *ref.grad_ptr_;
        return *this;
    }

    __host__ __device__
    float& data() {
        return data_;
    }
    __host__ __device__
    const float& data() const {
        return data_;
    }
    __host__ __device__
    float& grad() {
        return grad_;
    }
    __host__ __device__
    const float& grad() const {
        return grad_;
    }

    float data_;
    float grad_;
};

__host__ __device__
inline FloatGradRef<float>::FloatGradRef(float* data_ptr, float* grad_ptr)
    : data_ptr_(data_ptr), grad_ptr_(grad_ptr) {}

__host__ __device__
inline FloatGradRef<float>::FloatGradRef(const FloatGradRef<float>& other)
    : data_ptr_(other.data_ptr_), grad_ptr_(other.grad_ptr_) {}

__host__ __device__
inline FloatGradRef<float>& FloatGradRef<float>::operator=(const FloatGrad<float>& other) {
    *data_ptr_ = other.data_;
    *grad_ptr_ = other.grad_;
    return *this;
}

__host__ __device__
inline FloatGrad<float>::FloatGrad(const float& data)
    : data_(data), grad_(0.0f) {}

__host__ __device__
inline FloatGrad<float>::FloatGrad(const float& data, const float& grad)
    : data_(data), grad_(grad) {}

__host__ __device__
inline FloatGrad<float>::FloatGrad(const FloatGrad<float>& other)
    : data_(other.data_), grad_(other.grad_) {}

__host__ __device__
inline FloatGrad<float>::FloatGrad(const FloatGradRef<float>& ref)
    : data_(*ref.data_ptr_), grad_(*ref.grad_ptr_) {}


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
