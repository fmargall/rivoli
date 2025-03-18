#include <Eigen/Dense>
#include <glm/glm.hpp>

#include "interpolator.hpp"
#include "logger.hpp"

template <typename FloatingPrecision>
RBFInterpolator<FloatingPrecision>::RBFInterpolator(RuntimeConfig<FloatingPrecision>& runtimeConfig) {

	// Initialising the kernel function
	if (runtimeConfig.m_kernel == "linear")
		m_kernel = [](FloatingPrecision arg) { return arg; };
	else
		LOG_CRITICAL("Invalid kernel function: " + runtimeConfig.m_kernel);

	// Initialising the RBF kernel matrix
	Eigen::MatrixXd distanceMatrix = Eigen::MatrixXd::Zero(runtimeConfig.m_coordinates.size(),
		                                                   runtimeConfig.m_coordinatesRBF.size());

}

// Explicit instantiation
template class RBFInterpolator<float>;
template class RBFInterpolator<double>;