#pragma once

#include <memory>
#include <vector>

#include "configurator.hpp"
#include "coordinate.hpp"

// Forward declaration
template <typename FloatingPrecision>
class RuntimeConfig;

template <typename FloatingPrecision>
class RBFInterpolator {
public:
	RBFInterpolator(RuntimeConfig<FloatingPrecision>& runtimeConfig);

private:
	std::vector<FloatingPrecision> m_coefficients;
	std::vector<std::unique_ptr<Coordinate>> m_coordinates;
};