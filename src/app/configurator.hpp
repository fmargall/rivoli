#pragma once

#include <memory>
#include <string>
#include <vector>

#include "coordinate.hpp"
#include "export.hpp"
#include "interpolator.hpp"
#include "topology.hpp"

// Forward declarations
template <typename FloatingPrecision>
class RBFInterpolator;

class MainConfig {
public:
	MainConfig(const std::string& configFilePath);

	// Virtual destructor
	virtual ~MainConfig() = default;

	std::string getFloatingPointPrecision() const;

protected:
	template <typename FloatingPrecision>
	friend void initRBFCoeffsFile(const std::string& outputFilePath,
							      const RuntimeConfig<FloatingPrecision>& runtimeConfig,
								  const bool& isRGB);

	size_t m_nbClusters;
	size_t m_nbDimensions;

	bool m_forceReciprocity = true;
	bool m_forceBilateralSymmetry = true;

	std::string m_kernel = "linear";
	std::string m_inputFilePath;
	std::string m_locationRBFFilePath;
	std::string m_floatingPointPrecision = "FP32";

	std::string m_locationRBFSampler;
	size_t      m_numberRBF;

	std::string m_inputParameterisation = "spherical";
	std::string m_forceGrazingAnglesNullFunctionHemisphereOne = "none";
	std::string m_forceGrazingAnglesNullFunctionHemisphereTwo = "none";

	float m_maximumThetaInputFirstHemisphere  = glm::half_pi<float>();
	float m_maximumThetaInputSecondHemisphere = glm::half_pi<float>();

	std::string m_outputFormat;
	std::string m_outputFilePath;

	float m_thresholdCoef = 0.f;
	float m_regularisationParameter = 0.f;
	float m_nonNegativityCorrectionParameter = 1.0f;

	bool m_parallelComputing = true;
	bool m_uniqueLocationRBF = true;
	bool m_forceSingleThreadWriting = false;
};

template <typename FloatingPrecision>
class RuntimeConfig : public MainConfig {
public:
	// Constructor from a parent class instance
	RuntimeConfig(const MainConfig& mainConfig);
	// Copy constructor
	RuntimeConfig(const RuntimeConfig& other);

private:
	friend class RBFInterpolator<FloatingPrecision>;

	// We declare these functions as friends using a different template parameter name,
	// (FloatingPrecisionType) in order to avoid shadowing the class template parameter
	// (FloatingPrecision). This ensures compatibility with both MSVC, and GCC / Clang
	// compilers, the latter refusing template shadowing.
	template <typename FloatingPrecisionType>
	friend void initRBFCoeffsFile(const std::string& outputFilePath,
								  const RuntimeConfig<FloatingPrecisionType>& runtimeConfig,
								  const bool& isRGB);

	template <typename FloatingPrecisionType, typename DataType>
	friend void writeToRBFCoeffs(const std::string& outputFilePath,
						         const RuntimeConfig<FloatingPrecisionType>& runtimeConfig,
							     const std::vector<DataType>& inputData,
							     const size_t& clusterID);

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