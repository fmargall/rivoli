#pragma once
#include <functional>
#include <memory>
#include <vector>

#include "configurator.hpp"
#include "coordinate.hpp"
#include "topology.hpp"

template <typename FloatingPrecision>
class RuntimeConfig;

template <typename FloatingPrecision>
class RBFInterpolator {
public:
	RBFInterpolator(RuntimeConfig<FloatingPrecision>& runtimeConfig);

	FloatingPrecision interpolate(const std::unique_ptr<Coordinate>& coordinate) const;

	void reduceInterpolator(const FloatingPrecision& threshold = static_cast<FloatingPrecision>(0));

	std::vector<FloatingPrecision> getCoefficients() const;

private:
	std::unique_ptr<Topology<FloatingPrecision>> m_topology;

	std::function<FloatingPrecision(FloatingPrecision)> m_kernel;
	std::function<FloatingPrecision(FloatingPrecision)> m_grazingAngleFctHemisphereOne;
	std::function<FloatingPrecision(FloatingPrecision)> m_grazingAngleFctHemisphereTwo;

	std::vector<FloatingPrecision> m_coefficients;
	std::vector<std::unique_ptr<Coordinate>> m_coordinates;

	// Used to insure non-negativity interpolation
	FloatingPrecision m_nonNegativityCorrectionParameter;

	// Can speed up the interpolation when set to a value greater than 0
	// Caution: this will not reduce the number of coefficients. It will
	// only pass some of them during the computation, but won't suppress
	// them from the list of coefficients. For deleting them completely,
	// use the reduceInterpolator function.
	FloatingPrecision m_thresholdCoefficient = static_cast<FloatingPrecision>(0);
};