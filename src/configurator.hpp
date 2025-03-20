#pragma once

#include <memory>
#include <string>
#include <vector>

#include "coordinate.hpp"
#include "interpolator.hpp"
#include "topology.hpp"

// Forward declaration
template <typename FloatingPrecision>
class RBFInterpolator;

class MainConfig {
public:
	MainConfig(const std::string& configFilePath);

	std::string getFloatingPointPrecision() const;

protected:
	size_t m_nbDimensions;

	bool m_forceReciprocity = true;
	bool m_forceBilateralSymmetry = true;

	std::string m_kernel = "linear";
	std::string m_inputFilePath;
	std::string m_locationRBFFilePath;
	std::string m_floatingPointPrecision = "FP32";

	std::string m_outputFormat;
	std::string m_outputFilePath;

	float m_regularisationParameter = 0.f;
	float m_nonNegativityCorrectionParameter = 1.0f;

	bool m_parallelComputing = true;
};

template <typename FloatingPrecision>
class RuntimeConfig : public MainConfig {
public:
	RuntimeConfig(const MainConfig& mainConfig);

private:
	friend class RBFInterpolator<FloatingPrecision>;

	static std::unique_ptr<Topology<FloatingPrecision>> initTopology(RuntimeConfig<FloatingPrecision>& runtimeConfig);

	// Class containing the distance function related to the BRDF topology,
	// accessed using m_topology.getDistance(bidirectionOne, bidirectionTwo)
	std::unique_ptr<Topology<FloatingPrecision>> m_topology;

	// Input data, with their coordinates and values
	std::vector<std::unique_ptr<Coordinate>> m_coordinates;
	std::vector<FloatingPrecision> m_values;

	// Output RBF coordinates, that can be pointed to m_coordinates in
	// the case where no constraint is applied to the RBF location, or
	// to a chosen set of directions in the other case.
	std::vector<std::unique_ptr<Coordinate>> m_coordinatesRBF;

	// Coefficients of the RBF. Can be computed later by the interpolator
	std::vector<FloatingPrecision> m_coefficients;
};