#pragma once

#include <memory>
#include <string>
#include <vector>

#include "coordinate.hpp"
#include "interpolator.hpp"

// Forward declaration
template <typename FloatingPrecision>
class RBFInterpolator;

class MainConfig {
public:
	MainConfig(const std::string& configFilePath);

	std::string getFloatingPointPrecision() const;

protected:
	std::string m_kernel;
	std::string m_inputFilePath;
	std::string m_locationRBFFilePath;
	std::string m_floatingPointPrecision;
};

template <typename FloatingPrecision>
class RuntimeConfig : public MainConfig {
public:
	RuntimeConfig(const MainConfig& mainConfig);

private:
	friend class RBFInterpolator<FloatingPrecision>;

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