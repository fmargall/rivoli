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

	if (runtimeConfig.m_coordinates.size() != runtimeConfig.m_coordinatesRBF.size()) {
		// The distance matrix will not be a square matrix, so each
		// coefficient needs to be computed with a two-nested loop.
		LOG_DEBUG("Distance matrix is not square. Solving using canonical equations.");

		for (size_t rowID = 0; rowID < runtimeConfig.m_coordinates.size(); rowID++) {
			for (size_t colID = 0; colID < runtimeConfig.m_coordinatesRBF.size(); colID++) {
				//FloatingPrecision distance = m_topology.getDistance(runtimeConfig.m_coordinates[rowID],
				//													runtimeConfig.m_coordinatesRBF[colID]);

				//distanceMatrix(rowID, colID) = m_kernel(distance);
			}
		}
	}

}

// Explicit instantiation
template class RBFInterpolator<float>;
template class RBFInterpolator<double>;