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

    return vct::acos(vct::cos(thetaOne) * vct::cos(thetaTwo) +
                     vct::sin(thetaOne) * vct::sin(thetaTwo) * vct::cos(phiOne - phiTwo));
}

template <typename FP, vectra::SIMDLevel level>
FORCE_INLINE vectra::Vectratype<FP, level> distanceGreatCircleBilateral(
    vectra::Vectratype<FP, level> thetaOne,
    vectra::Vectratype<FP, level> phiOne  ,
    vectra::Vectratype<FP, level> thetaTwo,
    vectra::Vectratype<FP, level> phiTwo)
{
    using vct = vectra::Vectratype<FP, level>;

    vct distanceOne = rivoli::distanceGreatCircle(thetaOne, phiOne, thetaTwo, phiTwo);
    vct distanceTwo = rivoli::distanceGreatCircle(thetaOne, phiOne, thetaTwo, vct::two_pi() - phiTwo);

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

template <typename FP, vectra::SIMDLevel level>
FORCE_INLINE vectra::Vectratype<FP, level> distance1Rx2SReciprocalRusinkiewiczEuclidean(
    vectra::Vectratype<FP, level> thetaHOne,
    vectra::Vectratype<FP, level> thetaDOne,
    vectra::Vectratype<FP, level> phiDOne  ,
    vectra::Vectratype<FP, level> thetaHTwo,
    vectra::Vectratype<FP, level> thetaDTwo,
    vectra::Vectratype<FP, level> phiDTwo)
{
    using vct = vectra::Vectratype<FP, level>;

    vct distanceOne   = rivoli::distance1Rx2SEuclidean(thetaHOne, thetaDOne, phiDOne,
                                                       thetaHTwo, thetaDTwo, phiDTwo);
    vct distanceTwo   = rivoli::distance1Rx2SEuclidean(thetaHOne, thetaDOne, phiDOne + vct::pi(),
                                                       thetaHTwo, thetaDTwo, phiDTwo);

    return vct::min(distanceOne, distanceTwo);
}

template <typename FP, vectra::SIMDLevel level>
FORCE_INLINE vectra::Vectratype<FP, level> distance2Sx2SEuclidean(
    vectra::Vectratype<FP, level> thetaAOne,
    vectra::Vectratype<FP, level> phiAOne  ,
    vectra::Vectratype<FP, level> thetaBOne,
    vectra::Vectratype<FP, level> phiBOne  ,
    vectra::Vectratype<FP, level> thetaATwo,
    vectra::Vectratype<FP, level> phiATwo  ,
    vectra::Vectratype<FP, level> thetaBTwo,
    vectra::Vectratype<FP, level> phiBTwo)
{
    using vct = vectra::Vectratype<FP, level>;

    vct distanceS1 = rivoli::distanceGreatCircle(thetaAOne, phiAOne, thetaATwo, phiATwo);
    vct distanceS2 = rivoli::distanceGreatCircle(thetaBOne, phiBOne, thetaBTwo, phiBTwo);

    return vct::sqrt(distanceS1 * distanceS1 + distanceS2 * distanceS2);
}

template <typename FP, vectra::SIMDLevel level>
FORCE_INLINE vectra::Vectratype<FP, level> distance2Sx2SBilateralEuclidean(
    vectra::Vectratype<FP, level> thetaAOne,
    vectra::Vectratype<FP, level> phiAOne  ,
    vectra::Vectratype<FP, level> thetaBOne,
    vectra::Vectratype<FP, level> phiBOne  ,
    vectra::Vectratype<FP, level> thetaATwo,
    vectra::Vectratype<FP, level> phiATwo  ,
    vectra::Vectratype<FP, level> thetaBTwo,
    vectra::Vectratype<FP, level> phiBTwo)
{
    using vct = vectra::Vectratype<FP, level>;

    vct distanceS1One = rivoli::distanceGreatCircle(thetaAOne, phiAOne, thetaATwo, phiATwo);
    vct distanceS1Two = rivoli::distanceGreatCircle(thetaAOne, phiAOne, thetaATwo, vct::two_pi() - phiATwo);
    vct distanceS2One = rivoli::distanceGreatCircle(thetaBOne, phiBOne, thetaBTwo, phiBTwo);
    vct distanceS2Two = rivoli::distanceGreatCircle(thetaBOne, phiBOne, thetaBTwo, vct::two_pi() - phiBTwo);

    vct distanceS1 = vct::min(distanceS1One, distanceS1Two);
    vct distanceS2 = vct::min(distanceS2One, distanceS2Two);

    return vct::sqrt(distanceS1 * distanceS1 + distanceS2 * distanceS2);
}

template <typename FP, vectra::SIMDLevel level>
FORCE_INLINE vectra::Vectratype<FP, level> distance2Sx2SReciprocalEuclidean(
    vectra::Vectratype<FP, level> thetaAOne,
    vectra::Vectratype<FP, level> phiAOne  ,
    vectra::Vectratype<FP, level> thetaBOne,
    vectra::Vectratype<FP, level> phiBOne  ,
    vectra::Vectratype<FP, level> thetaATwo,
    vectra::Vectratype<FP, level> phiATwo  ,
    vectra::Vectratype<FP, level> thetaBTwo,
    vectra::Vectratype<FP, level> phiBTwo)
{
    using vct = vectra::Vectratype<FP, level>;

    vct distanceS1One = rivoli::distanceGreatCircle(thetaAOne, phiAOne, thetaATwo, phiATwo);
    vct distanceS2One = rivoli::distanceGreatCircle(thetaBOne, phiBOne, thetaBTwo, phiBTwo);
    vct distanceOne   = vct::sqrt(distanceS1One * distanceS1One + distanceS2One * distanceS2One);

    vct distanceS1Two = rivoli::distanceGreatCircle(thetaBOne, phiBOne, thetaATwo, phiATwo);
    vct distanceS2Two = rivoli::distanceGreatCircle(thetaAOne, phiAOne, thetaBTwo, phiBTwo);
    vct distanceTwo   = vct::sqrt(distanceS1Two * distanceS1Two + distanceS2Two * distanceS2Two);

    return vct::min(distanceOne, distanceTwo);
}

template <typename FP, vectra::SIMDLevel level>
FORCE_INLINE vectra::Vectratype<FP, level> distance2Sx2SBilateralReciprocalEuclidean(
    vectra::Vectratype<FP, level> thetaAOne,
    vectra::Vectratype<FP, level> phiAOne  ,
    vectra::Vectratype<FP, level> thetaBOne,
    vectra::Vectratype<FP, level> phiBOne  ,
    vectra::Vectratype<FP, level> thetaATwo,
    vectra::Vectratype<FP, level> phiATwo  ,
    vectra::Vectratype<FP, level> thetaBTwo,
    vectra::Vectratype<FP, level> phiBTwo)
{
    using vct = vectra::Vectratype<FP, level>;

    vct distanceOne   = rivoli::distance2Sx2SReciprocalEuclidean(thetaAOne, phiAOne, thetaBOne, phiBOne, 
                                                                 thetaATwo, phiATwo, thetaBTwo, phiBTwo);
    vct distanceTwo   = rivoli::distance2Sx2SReciprocalEuclidean(thetaAOne, vct::two_pi() - phiAOne, thetaBOne, phiBOne,
                                                                 thetaATwo, phiATwo, thetaBTwo, phiBTwo);
    vct distanceThree = rivoli::distance2Sx2SReciprocalEuclidean(thetaAOne, phiAOne, thetaBOne, vct::two_pi() - phiBOne,
                                                                 thetaATwo, phiATwo, thetaBTwo, phiBTwo);
    vct distanceFour  = rivoli::distance2Sx2SReciprocalEuclidean(thetaAOne, vct::two_pi() - phiAOne, thetaBOne, vct::two_pi() - phiBOne,
                                                                 thetaATwo, phiATwo, thetaBTwo, phiBTwo);

    return vct::min(vct::min(distanceOne, distanceTwo), vct::min(distanceThree, distanceFour));
}

template <typename FP, vectra::SIMDLevel level>
FORCE_INLINE vectra::Vectratype<FP, level> distance2Sx2SReciprocalRusinkiewiczEuclidean(
    vectra::Vectratype<FP, level> thetaHOne,
    vectra::Vectratype<FP, level> phiHOne  ,
    vectra::Vectratype<FP, level> thetaDOne,
    vectra::Vectratype<FP, level> phiDOne  ,
    vectra::Vectratype<FP, level> thetaHTwo,
    vectra::Vectratype<FP, level> phiHTwo  ,
    vectra::Vectratype<FP, level> thetaDTwo,
    vectra::Vectratype<FP, level> phiDTwo)
{
    using vct = vectra::Vectratype<FP, level>;

    vct distanceH = rivoli::distanceGreatCircle(thetaHOne, phiHOne, thetaHTwo, phiHTwo);

    vct distanceDOne = rivoli::distanceGreatCircle(thetaDOne, phiDOne, thetaDTwo, phiDTwo);
    vct distanceDTwo = rivoli::distanceGreatCircle(thetaDOne, phiDOne, thetaDTwo, phiDTwo + vct::pi());
    vct distanceD    = vct::min(distanceDOne, distanceDTwo);

    return vct::sqrt(distanceH * distanceH + distanceD * distanceD);
}

template <typename FP, vectra::SIMDLevel level>
FORCE_INLINE vectra::Vectratype<FP, level> distance2Sx2SBilateralReciprocalRusinkiewiczEuclidean(
    vectra::Vectratype<FP, level> thetaHOne,
    vectra::Vectratype<FP, level> phiHOne  ,
    vectra::Vectratype<FP, level> thetaDOne,
    vectra::Vectratype<FP, level> phiDOne  ,
    vectra::Vectratype<FP, level> thetaHTwo,
    vectra::Vectratype<FP, level> phiHTwo  ,
    vectra::Vectratype<FP, level> thetaDTwo,
    vectra::Vectratype<FP, level> phiDTwo)
{
    using vct = vectra::Vectratype<FP, level>;

    vct distanceOne = rivoli::distance2Sx2SReciprocalRusinkiewiczEuclidean(
        thetaHOne, phiHOne, thetaDOne, phiDOne,
		thetaHTwo, phiHTwo, thetaDTwo, phiDTwo);

    vct distanceTwo = rivoli::distance2Sx2SReciprocalRusinkiewiczEuclidean(
        thetaHOne, phiHOne, thetaDOne, phiDOne,
        thetaHTwo, phiHTwo, thetaDTwo, phiDTwo + vct::half_pi());

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
