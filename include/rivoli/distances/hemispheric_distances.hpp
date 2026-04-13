#pragma once


#include <vectra/vectra.hpp>

namespace rivoli
{


template <typename FP, vectra::SIMDLevel level>
struct HemisphericDistances
{
    using vct = vectra::Vectratype<FP, level>;

    vct hemisphericDistanceOne;
    vct hemisphericDistanceTwo;
};

template <typename FP>
struct HemisphericDistancesScalar {
    FP hemisphericDistanceOne;
    FP hemisphericDistanceTwo;
};

}
