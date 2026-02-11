#pragma once

#include <vectra/vectra.hpp>

namespace rivoli
{

template <typename FP, vectra::SIMDLevel level>
FORCE_INLINE vectra::Vectratype<FP, level> distanceGreatCircle(
    vectra::Vectratype<FP, level> thetaOne,
    vectra::Vectratype<FP, level> phiOne  ,
    vectra::Vectratype<FP, level> thetaTwo,
    vectra::Vectratype<FP, level> phiTwo)
{
    using vct = vectra::Vectratype<FP, level>;

    return vct::acos(vct::sin(thetaOne) * vct::sin(thetaTwo) + 
                     vct::cos(thetaOne) * vct::cos(thetaTwo) * vct::cos(phiOne - phiTwo));
}

namespace experimental
{

template <typename FP, vectra::SIMDLevel level>
FORCE_INLINE  typename vectra::ComputeBackend<FP, level>::type distanceGreatCircle(
    const typename vectra::ComputeBackend<FP, level>::type& thetaOne,
    const typename vectra::ComputeBackend<FP, level>::type& phiOne,
    const typename vectra::ComputeBackend<FP, level>::type& thetaTwo,
    const typename vectra::ComputeBackend<FP, level>::type& phiTwo)
{
    using backend = vectra::ComputeBackend<FP, level>;

    return backend::acos(backend::add(
           backend::mul (backend::sin(thetaOne), backend::sin(thetaTwo)),
           backend::mul (backend::cos(thetaOne), backend::mul(backend::cos(thetaTwo), backend::cos(backend::sub(phiOne, phiTwo))))));
}

}

}