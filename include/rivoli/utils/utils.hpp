#pragma once

#include <vectra/vectra.hpp>


namespace rivoli {

template <typename FP, vectra::SIMDLevel level>
void sphericalToRusinkiewicz(
    vectra::Vectratype<FP, level> thetaI,
    vectra::Vectratype<FP, level> thetaO,
    vectra::Vectratype<FP, level> deltaPhi)
{

}

template <typename FP, vectra::SIMDLevel level>
void sphericalToRusinkiewicz(
    vectra::Vectratype<FP, level> thetaI,
    vectra::Vectratype<FP, level> phiI,
    vectra::Vectratype<FP, level> thetaO,
    vectra::Vectratype<FP, level> phiO)
{

}

template <typename FP, vectra::SIMDLevel level>
void rusinkiewiczToSpherical(
    vectra::Vectratype<FP, level> thetaH,
    vectra::Vectratype<FP, level> thetaD,
    vectra::Vectratype<FP, level> phiD)
{

}

template <typename FP, vectra::SIMDLevel level>
void rusinkiewiczToSpherical(
    vectra::Vectratype<FP, level> thetaH,
    vectra::Vectratype<FP, level> phiH,
    vectra::Vectratype<FP, level> thetaD,
    vectra::Vectratype<FP, level> phiD)
{

}

}
