#pragma once

#include <vectra/vectra.hpp>


namespace rivoli {

template <typename FP, vectra::SIMDLevel level>
void sphericalToRusinkiewicz(
    vectra::Vectratype<FP, level> thetaI,
    vectra::Vectratype<FP, level> thetaO,
    vectra::Vectratype<FP, level> deltaPhi)
{
    using vct = vectra::Vectratype<FP, level>;
}

template <typename FP, vectra::SIMDLevel level>
void sphericalToRusinkiewicz(
    vectra::Vectratype<FP, level> thetaI,
    vectra::Vectratype<FP, level> phiI,
    vectra::Vectratype<FP, level> thetaO,
    vectra::Vectratype<FP, level> phiO)
{
    using vct = vectra::Vectratype<FP, level>;
}

template <typename FP, vectra::SIMDLevel level>
void rusinkiewiczToSpherical(
    vectra::Vectratype<FP, level> thetaH,
    vectra::Vectratype<FP, level> thetaD,
    vectra::Vectratype<FP, level> phiD)
{
    using vct = vectra::Vectratype<FP, level>;
}

template <typename FP, vectra::SIMDLevel level>
void rusinkiewiczToSpherical(
    vectra::Vectratype<FP, level> thetaH,
    vectra::Vectratype<FP, level> phiH,
    vectra::Vectratype<FP, level> thetaD,
    vectra::Vectratype<FP, level> phiD)
{
    using vct = vectra::Vectratype<FP, level>;
}

/*
 * 10.1109/TPAMI.2006.170
 */
template <typename FP, vectra::SIMDLevel level>
void rusinkiewiczToZickler(
    vectra::Vectratype<FP, level> thetaH,
    vectra::Vectratype<FP, level> thetaD,
    vectra::Vectratype<FP, level> phiD)
{
    using vct = vectra::Vectratype<FP, level>;

    vct twoPhiD = 2. * phiD;

    vct u = vct::sin(thetaH) * vct::cos(twoPhiD);
    vct v = vct::sin(thetaH) * vct::sin(twoPhiD);
    vct w = 2. * phiD / vct::pi;
}

}
