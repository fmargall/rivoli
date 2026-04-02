#pragma once

#include <rivoli/topologies/topologies.hpp>

#include <vectra/vectra.hpp>


namespace rivoli {

enum class CoordinateSystem {
	Spherical,
	Rusinkiewicz
};

template <size_t dimension, bool bilateral, bool reciprocal, CoordinateSystem coordSystem, typename FP, vectra::SIMDLevel level>
struct TopologyTraits;

template <size_t dimension, bool bilateral, bool reciprocal, CoordinateSystem coordSystem, typename FP, vectra::SIMDLevel level>
using TopologySelector = TopologyTraits<dimension, bilateral, reciprocal, coordSystem, FP, level>::type;


template <typename FP, vectra::SIMDLevel level>
struct TopologyTraits<2, false, false, CoordinateSystem::Spherical, FP, level> {
	using type = Topology2S<FP, level>;
};

template <typename FP, vectra::SIMDLevel level>
struct TopologyTraits<2, true, false, CoordinateSystem::Spherical, FP, level> {
	using type = Topology2SBilateral<FP, level>;
};


template <typename FP, vectra::SIMDLevel level>
struct TopologyTraits<4, false, false, CoordinateSystem::Spherical, FP, level> {
	using type = Topology2Sx2SEuclidean<FP, level>;
};

template <typename FP, vectra::SIMDLevel level>
struct TopologyTraits<4, true, false, CoordinateSystem::Spherical, FP, level> {
	using type = Topology2Sx2SBilateralEuclidean<FP, level>;
};

template <typename FP, vectra::SIMDLevel level>
struct TopologyTraits<4, false, true, CoordinateSystem::Spherical, FP, level> {
	using type = Topology2Sx2SReciprocalEuclidean<FP, level>;
};

template <typename FP, vectra::SIMDLevel level>
struct TopologyTraits<4, true, true, CoordinateSystem::Spherical, FP, level> {
	using type = Topology2Sx2SBilateralReciprocalEuclidean<FP, level>;
};

}
