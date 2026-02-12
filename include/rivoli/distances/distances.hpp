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

template <typename FP, vectra::SIMDLevel level>
FORCE_INLINE vectra::Vectratype<FP, level> distanceGreatCircleBilateral(
    vectra::Vectratype<FP, level> thetaOne,
    vectra::Vectratype<FP, level> phiOne  ,
    vectra::Vectratype<FP, level> thetaTwo,
    vectra::Vectratype<FP, level> phiTwo)
{
    using vct = vectra::Vectratype<FP, level>;

    vct distanceOne = rivoli::greatCircleDistance(thetaOne, phiOne, thetaTwo, phiTwo);
    vct distanceTwo = rivoli::greatCircleDistance(thetaOne, phiOne, thetaTwo, vct::two_pi() - phiTwo);

    return vct::min(distanceOne, distanceTwo);
}

template <typename FP, vectra::SIMDLevel level>
FORCE_INLINE vectra::Vectratype<FP, level> distance1Rx2SEuclidean(
    vectra::Vectratype<FP, level> thetaAOne,
    vectra::Vectratype<FP, level> thetaBOne,
    vectra::Vectratype<FP, level> phiBOne  ,
    vectra::Vectratype<FP, level> thetaATwo,
    vectra::Vectratype<FP, level> thetaBTwo,
    vectra::Vectratype<FP, level> phiBTwo)
{
    using vct = vectra::Vectratype<FP, level>;

    vct distanceR1 = thetaAOne - thetaATwo;
    vct distanceS2 = rivoli::distanceGreatCircle(thetaBOne, phiBOne, thetaBTwo, phiBTwo);

    return vct::sqrt(distanceR1 * distanceR1 + distanceS2 * distanceS2);
}

template <typename FP, vectra::SIMDLevel level>
FORCE_INLINE vectra::Vectratype<FP, level> distance1Rx2SBilateralEuclidean(
    vectra::Vectratype<FP, level> thetaAOne,
    vectra::Vectratype<FP, level> thetaBOne,
    vectra::Vectratype<FP, level> phiBOne  ,
    vectra::Vectratype<FP, level> thetaATwo,
    vectra::Vectratype<FP, level> thetaBTwo,
    vectra::Vectratype<FP, level> phiBTwo)
{
    using vct = vectra::Vectratype<FP, level>;

    vct distanceOne = rivoli::distance1Rx2SEuclidean(thetaAOne, thetaBOne, phiBOne,
                                                     thetaATwo, thetaBTwo, phiBTwo);
    vct distanceTwo = rivoli::distance1Rx2SEuclidean(thetaAOne, thetaBOne, phiBOne,
                                     thetaATwo, thetaBTwo, vct::two_pi() - phiBTwo);

    return vct::min(distanceOne, distanceTwo);
}

template <typename FP, vectra::SIMDLevel level>
FORCE_INLINE vectra::Vectratype<FP, level> distance1Rx2SReciprocalEuclidean(
    vectra::Vectratype<FP, level> thetaAOne,
    vectra::Vectratype<FP, level> thetaBOne,
    vectra::Vectratype<FP, level> phiBOne  ,
    vectra::Vectratype<FP, level> thetaATwo,
    vectra::Vectratype<FP, level> thetaBTwo,
    vectra::Vectratype<FP, level> phiBTwo)
{
    using vct = vectra::Vectratype<FP, level>;

    vct distanceOne   = rivoli::distance1Rx2SEuclidean(thetaAOne, thetaBOne, phiBOne,
                                                       thetaATwo, thetaBTwo, phiBTwo);
    vct distanceTwo   = rivoli::distance1Rx2SEuclidean(thetaBOne, thetaAOne, vct::two_pi() - phiBOne,
                                                       thetaATwo, thetaBTwo, phiBTwo);
    vct distanceThree = rivoli::distance1Rx2SEuclidean(thetaAOne, thetaBOne, phiBOne,
                                                       thetaBTwo, thetaATwo, vct::two_pi() - phiBTwo);
    vct distanceFour  = rivoli::distance1Rx2SEuclidean(thetaBOne, thetaAOne, vct::two_pi() - phiBOne,
                                                       thetaBTwo, thetaATwo, vct::two_pi() - phiBTwo);

    return vct::min(vct::min(distanceOne, distanceTwo), vct::min(distanceThree, distanceFour));
}

template <typename FP, vectra::SIMDLevel level>
FORCE_INLINE vectra::Vectratype<FP, level> distance1Rx2SBilateralReciprocalEuclidean(
    vectra::Vectratype<FP, level> thetaAOne,
    vectra::Vectratype<FP, level> thetaBOne,
    vectra::Vectratype<FP, level> phiBOne  ,
    vectra::Vectratype<FP, level> thetaATwo,
    vectra::Vectratype<FP, level> thetaBTwo,
    vectra::Vectratype<FP, level> phiBTwo)
{
    using vct = vectra::Vectratype<FP, level>;

    vct distanceOne = rivoli::distance1Rx2SReciprocalEuclidean(thetaAOne, thetaBOne, phiBOne,
                                                     thetaATwo, thetaBTwo, phiBTwo);
    vct distanceTwo = rivoli::distance1Rx2SReciprocalEuclidean(thetaAOne, thetaBOne, phiBOne,
                                                               thetaATwo, thetaBTwo, vct::two_pi() - phiBTwo);

    return vct::min(distanceOne, distanceTwo);
}


namespace experimental
{

template <typename FP, vectra::SIMDLevel level>
FORCE_INLINE  typename vectra::ComputeBackend<FP, level>::type distanceGreatCircle(
    const typename vectra::ComputeBackend<FP, level>::type& thetaOne,
    const typename vectra::ComputeBackend<FP, level>::type& phiOne  ,
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