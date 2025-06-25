#ifndef FLOAT_GRAD_ARRAY_H
#define FLOAT_GRAD_ARRAY_H

std::false_type is_float_grad_array_impl(const void*);
template <typename T>
std::true_type is_float_grad_array_impl(const FloatGradArray<T>*);
template <typename T>
using is_float_grad_array = decltype(is_float_grad_array_impl(std::declval<T*>()));

template <typename FloatType>
struct FloatGradArray {
    FloatType* data_arr_;
    FloatType* grad_arr_;

    __host__ __device__
    FloatGradArray(FloatType* data, FloatType* grad)
        : data_arr_(data), grad_arr_(grad) {}

    __host__ __device__
    FloatGradRef<FloatType> operator[](int index) {
        return FloatGradRef<FloatType>(data_arr_ + index, grad_arr_ + index);
    }

    __host__ __device__
    FloatGradRef<const FloatType> operator[](int index) const {
        return FloatGradRef<const FloatType>(data_arr_ + index, grad_arr_ + index);
    }

    template <typename CastType>
    __host__ __device__
    FloatGradArray<CastType> cast() const {
        return FloatGradArray<CastType>(
            reinterpret_cast<CastType*>(data_arr_),
            reinterpret_cast<CastType*>(grad_arr_)
        );
    }

    __host__ __device__
    FloatGradArray operator+(int offset) const {
        return FloatGradArray(data_arr_ + offset, grad_arr_ + offset);
    }


    __host__ __device__
    FloatType* data_ptr() {
        return data_arr_;
    }
    __host__ __device__
    const FloatType* data_ptr() const {
        return data_arr_;
    }
    __host__ __device__
    FloatType* grad_ptr() {
        return grad_arr_;
    }
    __host__ __device__
    const FloatType* grad_ptr() const {
        return grad_arr_;
    }
};

template <typename CastType, typename T>
__host__ __device__
auto cast(T a) {
    if constexpr (is_float_grad_array<T>::value) {
        return a.template cast<CastType>();
    } else {
        return reinterpret_cast<CastType*>(a);
    }
}

#endif // FLOAT_GRAD_ARRAY_H
