#pragma once

#include <rivoli/topologies/topologies.hpp>

#include <vectra/vectra.hpp>


namespace rivoli {

enum class CoordinateSystem {
	Spherical,
	Rusinkiewicz
};

template <size_t dimension, bool bilateral, bool reciprocal, bool isotropy, CoordinateSystem coordSystem, typename FP, vectra::SIMDLevel level>
struct TopologyTraits;

template <size_t dimension, bool bilateral, bool reciprocal, bool isotropy, CoordinateSystem coordSystem, typename FP, vectra::SIMDLevel level>
using TopologySelector = TopologyTraits<dimension, bilateral, reciprocal, isotropy, coordSystem, FP, level>::type;

template <typename FP, vectra::SIMDLevel level>
struct TopologyTraits<2, false, false, false, CoordinateSystem::Spherical, FP, level> {
	using type = Topology2S<FP, level>;
};

template <typename FP, vectra::SIMDLevel level>
struct TopologyTraits<2, false, false, true, CoordinateSystem::Spherical, FP, level> {
	// Forcing isotropy for a 2D topology,
	// does not change the topology itself
	using type = Topology2S<FP, level>;
};

template <typename FP, vectra::SIMDLevel level>
struct TopologyTraits<2, true, false, false, CoordinateSystem::Spherical, FP, level> {
	using type = Topology2SBilateral<FP, level>;
};

template <typename FP, vectra::SIMDLevel level>
struct TopologyTraits<2, true, false, true, CoordinateSystem::Spherical, FP, level> {
	// Forcing isotropy for a 2D topology,
	// does not change the topology itself
	using type = Topology2SBilateral<FP, level>;
};


template <typename FP, vectra::SIMDLevel level>
struct TopologyTraits<3, false, false, false, CoordinateSystem::Spherical, FP, level> {
	using type = Topology1Rx2SEuclidean<FP, level>;
};

template <typename FP, vectra::SIMDLevel level>
struct TopologyTraits<3, false, false, true, CoordinateSystem::Spherical, FP, level> {
	using type = Topology1Rx2SIsotropicEuclidean<FP, level>;
};

template <typename FP, vectra::SIMDLevel level>
struct TopologyTraits<3, true, false, false, CoordinateSystem::Spherical, FP, level> {
	using type = Topology1Rx2SBilateralEuclidean<FP, level>;
};

template <typename FP, vectra::SIMDLevel level>
struct TopologyTraits<3, true, false, true, CoordinateSystem::Spherical, FP, level> {
	using type = Topology1Rx2SBilateralIsotropicEuclidean<FP, level>;
};

template <typename FP, vectra::SIMDLevel level>
struct TopologyTraits<3, false, true, false, CoordinateSystem::Spherical, FP, level> {
	using type = Topology1Rx2SReciprocalEuclidean<FP, level>;
};

template <typename FP, vectra::SIMDLevel level>
struct TopologyTraits<3, false, true, true, CoordinateSystem::Spherical, FP, level> {
	using type = Topology1Rx2SReciprocalIsotropicEuclidean<FP, level>;
};

template <typename FP, vectra::SIMDLevel level>
struct TopologyTraits<3, true, true, false, CoordinateSystem::Spherical, FP, level> {
	using type = Topology1Rx2SBilateralReciprocalEuclidean<FP, level>;
};

template <typename FP, vectra::SIMDLevel level>
struct TopologyTraits<3, true, true, true, CoordinateSystem::Spherical, FP, level> {
	using type = Topology1Rx2SBilateralReciprocalIsotropicEuclidean<FP, level>;
};

template <typename FP, vectra::SIMDLevel level>
struct TopologyTraits<3, false, false, false, CoordinateSystem::Rusinkiewicz, FP, level> {
	// Fallback to the usual 4D topology, since the
	// Rusinkiewicz system is also defined as 1Rx2S
	using type = Topology1Rx2SEuclidean<FP, level>;
};

template <typename FP, vectra::SIMDLevel level>
struct TopologyTraits<3, false, false, true, CoordinateSystem::Rusinkiewicz, FP, level> {
	// Fallback to the usual 4D topology, since the
	// Rusinkiewicz system is also defined as 1Rx2S
	using type = Topology1Rx2SIsotropicEuclidean<FP, level>;
};

template <typename FP, vectra::SIMDLevel level>
struct TopologyTraits <3, false, true, false, CoordinateSystem::Rusinkiewicz, FP, level> {
	using type = Topology1Rx2SReciprocalRusinkiewiczEuclidean<FP, level>;
};

template <typename FP, vectra::SIMDLevel level>
struct TopologyTraits <3, false, true, true, CoordinateSystem::Rusinkiewicz, FP, level> {
	using type = Topology1Rx2SReciprocalRusinkiewiczIsotropicEuclidean<FP, level>;
};

template <typename FP, vectra::SIMDLevel level>
struct TopologyTraits <3, true, false, false, CoordinateSystem::Rusinkiewicz, FP, level> {
	// Rusinkiewicz parameterisation cannot intrinsically force
	// symmetry, without forcing also the Helmholtz reciprocity
	static_assert(false, "Rusinkiewicz parameterisation cannot be bilateral symmetrical without being also reciprocal");
	using type = Topology1Rx2SBilateralReciprocalRusinkiewiczEuclidean<FP, level>;
};

template <typename FP, vectra::SIMDLevel level>
struct TopologyTraits <3, true, false, true, CoordinateSystem::Rusinkiewicz, FP, level> {
	// Rusinkiewicz parameterisation cannot intrinsically force
	// symmetry, without forcing also the Helmholtz reciprocity
	static_assert(false, "Rusinkiewicz parameterisation cannot be bilateral symmetrical without being also reciprocal");
	using type = Topology1Rx2SBilateralReciprocalRusinkiewiczIsotropicEuclidean<FP, level>;
};

template <typename FP, vectra::SIMDLevel level>
struct TopologyTraits <3, true, true, false, CoordinateSystem::Rusinkiewicz, FP, level> {
	using type = Topology1Rx2SBilateralReciprocalRusinkiewiczEuclidean<FP, level>;
};

template <typename FP, vectra::SIMDLevel level>
struct TopologyTraits <3, true, true, true, CoordinateSystem::Rusinkiewicz, FP, level> {
	using type = Topology1Rx2SBilateralReciprocalRusinkiewiczIsotropicEuclidean<FP, level>;
};


template <typename FP, vectra::SIMDLevel level>
struct TopologyTraits<4, false, false, false, CoordinateSystem::Spherical, FP, level> {
	using type = Topology2Sx2SEuclidean<FP, level>;
};

template <typename FP, vectra::SIMDLevel level>
struct TopologyTraits<4, false, false, true, CoordinateSystem::Spherical, FP, level> {
	// Forcing isotropy for a 4-dimensional topology, by definition, is not possible
	static_assert(false, "Isotropy cannot be enforced for 4-dimensional topologies.");
	using type = Topology2Sx2SEuclidean<FP, level>;
};

template <typename FP, vectra::SIMDLevel level>
struct TopologyTraits<4, true, false, false, CoordinateSystem::Spherical, FP, level> {
	using type = Topology2Sx2SBilateralEuclidean<FP, level>;
};

template <typename FP, vectra::SIMDLevel level>
struct TopologyTraits<4, true, false, true, CoordinateSystem::Spherical, FP, level> {
	// Forcing isotropy for a 4-dimensional topology, by definition, is not possible
	static_assert(false, "Isotropy cannot be enforced for 4-dimensional topologies.");
	using type = Topology2Sx2SBilateralEuclidean<FP, level>;
};

template <typename FP, vectra::SIMDLevel level>
struct TopologyTraits<4, false, true, false, CoordinateSystem::Spherical, FP, level> {
	using type = Topology2Sx2SReciprocalEuclidean<FP, level>;
};

template <typename FP, vectra::SIMDLevel level>
struct TopologyTraits<4, false, true, true, CoordinateSystem::Spherical, FP, level> {
	// Forcing isotropy for a 4-dimensional topology, by definition, is not possible
	static_assert(false, "Isotropy cannot be enforced for 4-dimensional topologies.");
	using type = Topology2Sx2SReciprocalEuclidean<FP, level>;
};

template <typename FP, vectra::SIMDLevel level>
struct TopologyTraits<4, true, true, false, CoordinateSystem::Spherical, FP, level> {
	using type = Topology2Sx2SBilateralReciprocalEuclidean<FP, level>;
};

template <typename FP, vectra::SIMDLevel level>
struct TopologyTraits<4, true, true, true, CoordinateSystem::Spherical, FP, level> {
	// Forcing isotropy for a 4-dimensional topology, by definition, is not possible
	static_assert(false, "Isotropy cannot be enforced for 4-dimensional topologies.");
	using type = Topology2Sx2SBilateralReciprocalEuclidean<FP, level>;
};


template <typename FP, vectra::SIMDLevel level>
struct TopologyTraits<4, false, false, false, CoordinateSystem::Rusinkiewicz, FP, level> {
	// Fallback to the usual 4D topology, since the
	// Rusinkiewicz system is also defined as 2Sx2S
	using type = Topology2Sx2SEuclidean<FP, level>;
};

template <typename FP, vectra::SIMDLevel level>
struct TopologyTraits<4, false, false, true, CoordinateSystem::Rusinkiewicz, FP, level> {
	// Forcing isotropy for a 4-dimensional topology, by definition, is not possible
	static_assert(false, "Isotropy cannot be enforced for 4-dimensional topologies.");
	// Fallback to the usual 4D topology, since the
	// Rusinkiewicz system is also defined as 2Sx2S
	using type = Topology2Sx2SEuclidean<FP, level>;
};

template <typename FP, vectra::SIMDLevel level>
struct TopologyTraits<4, false, true, false, CoordinateSystem::Rusinkiewicz, FP, level> {
	using type = Topology2Sx2SReciprocalRusinkiewiczEuclidean<FP, level>;
};

template <typename FP, vectra::SIMDLevel level>
struct TopologyTraits<4, false, true, true, CoordinateSystem::Rusinkiewicz, FP, level> {
	// Forcing isotropy for a 4-dimensional topology, by definition, is not possible
	static_assert(false, "Isotropy cannot be enforced for 4-dimensional topologies.");
	using type = Topology2Sx2SReciprocalRusinkiewiczEuclidean<FP, level>;
};

template <typename FP, vectra::SIMDLevel level>
struct TopologyTraits<4, true, false, false, CoordinateSystem::Rusinkiewicz, FP, level> {
	// Rusinkiewicz parameterisation cannot intrinsically force
	// symmetry, without forcing also the Helmholtz reciprocity
	static_assert(false, "Rusinkiewicz parameterisation cannot be bilateral symmetrical without being also reciprocal");
	using type = Topology2Sx2SBilateralReciprocalRusinkiewiczEuclidean<FP, level>;

};

template <typename FP, vectra::SIMDLevel level>
struct TopologyTraits<4, true, false, true, CoordinateSystem::Rusinkiewicz, FP, level> {
	// Forcing isotropy for a 4-dimensional topology, by definition, is not possible
	static_assert(false, "Isotropy cannot be enforced for 4-dimensional topologies.");
	// Rusinkiewicz parameterisation cannot intrinsically force
	// symmetry, without forcing also the Helmholtz reciprocity
	static_assert(false, "Rusinkiewicz parameterisation cannot be bilateral symmetrical without being also reciprocal");
	using type = Topology2Sx2SBilateralReciprocalRusinkiewiczEuclidean<FP, level>;

};

template <typename FP, vectra::SIMDLevel level>
struct TopologyTraits<4, true, true, false, CoordinateSystem::Rusinkiewicz, FP, level> {
	using type = Topology2Sx2SBilateralReciprocalRusinkiewiczEuclidean<FP, level>;
};

template <typename FP, vectra::SIMDLevel level>
struct TopologyTraits<4, true, true, true, CoordinateSystem::Rusinkiewicz, FP, level> {
	// Forcing isotropy for a 4-dimensional topology, by definition, is not possible
	static_assert(false, "Isotropy cannot be enforced for 4-dimensional topologies.");
	using type = Topology2Sx2SBilateralReciprocalRusinkiewiczEuclidean<FP, level>;
};

}
