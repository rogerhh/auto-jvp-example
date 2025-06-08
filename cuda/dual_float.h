#ifndef DUAL_FLOAT_H
#define DUAL_FLOAT_H

#include <type_traits>

// This is the base template for a dual number representation.
template <typename FloatType>
struct FloatGrad {
    struct Val;
    struct Ref;

    struct Val {
        FloatType data;
        FloatType grad;
    };
    struct Ref {
        FloatType* data_ptr;
        FloatType* grad_ptr;
    }
}

template <typename FloatType, bool IsRef=true>
struct DualFloat {
    using ValueType = typename std::conditional<IsRef, 
          FloatGrad<FloatType>::Ref, FloatGrad<FloatType>::Val>::type;
    ValueType val;
};

#endif // DUAL_FLOAT_H

