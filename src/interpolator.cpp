#include <Eigen/Dense>
#include <glm/glm.hpp>

#include "interpolator.hpp"
#include "logger.hpp"

template <typename FloatingPrecision>
RBFInterpolator<FloatingPrecision>::RBFInterpolator(RuntimeConfig<FloatingPrecision>& runtimeConfig) {

}

// Explicit instantiation
template class RBFInterpolator<float>;
template class RBFInterpolator<double>;