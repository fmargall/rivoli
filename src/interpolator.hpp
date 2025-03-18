#pragma once
#include <functional>
#include <memory>
#include <vector>

#include "configurator.hpp"
#include "coordinate.hpp"
#include "topology.hpp"

// Forward declaration
template <typename FloatingPrecision>
class RuntimeConfig;

template <typename FloatingPrecision>
class RBFInterpolator {
public:
	RBFInterpolator(RuntimeConfig<FloatingPrecision>& runtimeConfig);

private:
	std::unique_ptr<Topology<FloatingPrecision>> m_topology;

	std::function<FloatingPrecision(FloatingPrecision)> m_kernel;

	std::vector<FloatingPrecision> m_coefficients;
	std::vector<std::unique_ptr<Coordinate>> m_coordinates;
};