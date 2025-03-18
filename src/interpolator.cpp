#include <Eigen/Dense>
#include <glm/glm.hpp>

#include "interpolator.hpp"
#include "logger.hpp"

template <typename FloatingPrecision>
RBFInterpolator<FloatingPrecision>::RBFInterpolator(RuntimeConfig<FloatingPrecision>& runtimeConfig) {

	// Initialising the topology
	m_topology = RuntimeConfig<FloatingPrecision>::initTopology(runtimeConfig);

	// Initialising the kernel function
	if (runtimeConfig.m_kernel == "linear")
		m_kernel = [](FloatingPrecision arg) { return arg; };
	else
		LOG_CRITICAL("Invalid kernel function: " + runtimeConfig.m_kernel);

	// Initialising the RBF kernel matrix
	Eigen::MatrixXd distanceMatrix = Eigen::MatrixXd::Zero(runtimeConfig.m_coordinates.size(),
		                                                   runtimeConfig.m_coordinatesRBF.size());

	if (runtimeConfig.m_locationRBFFilePath != "none") {
		// The distance matrix will not be a square matrix, so each
		// coefficient needs to be computed with a two-nested loop.
		LOG_DEBUG("Computing rectangular kernel-distance matrix.");

		size_t iterationNumber = 0;
		for (size_t rowID = 0; rowID < runtimeConfig.m_coordinates.size(); rowID++) {
			for (size_t colID = 0; colID < runtimeConfig.m_coordinatesRBF.size(); colID++) {
				FloatingPrecision distance = m_topology->getDistance(*(runtimeConfig.m_coordinates[rowID]), 
															         *(runtimeConfig.m_coordinatesRBF[colID]));
				distanceMatrix(rowID, colID) = m_kernel(distance);

				logger.displayProgressBar(++iterationNumber, runtimeConfig.m_coordinates.size() * (runtimeConfig.m_coordinatesRBF.size() - 1));
			}
		}
	}
	else {
		// The distance matrix will be square and symmetrical matrix. 
		// We can accelerate its computation by using this property.
		LOG_DEBUG("Computing square kernel-distance matrix.");

		size_t iterationNumber = 0;
		for (size_t rowID = 0; rowID < runtimeConfig.m_coordinates.size(); rowID++) {
			// Matrix is symmetrical, so we only need to compute the upper triangle
			for (size_t colID = 0; colID <= rowID; colID++) {
				FloatingPrecision distance = m_topology->getDistance(*(runtimeConfig.m_coordinates[rowID]),
																	 *(runtimeConfig.m_coordinatesRBF[colID]));
				distanceMatrix(rowID, colID) = m_kernel(distance);
			}
		}
	}

	LOG_DEBUG("Kernel-distance matrix computed.");

	// Used to ensure that the result is non-negative (this
	// will be corrected when computing the interpolation).
	m_nonNegativityCorrectionParameter = static_cast<FloatingPrecision>(runtimeConfig.m_nonNegativityCorrectionParameter);
	FloatingPrecision power = static_cast<FloatingPrecision>(1 / m_nonNegativityCorrectionParameter);

	// Initialisation of the result
	Eigen::VectorXd resultVector(runtimeConfig.m_coordinates.size());
	for (size_t i = 0; i < runtimeConfig.m_coordinates.size(); i++) {
		resultVector(i) = glm::pow(runtimeConfig.m_values[i], power);
	}

	if (runtimeConfig.m_locationRBFFilePath == "none") {
		LOG_DEBUG("Computing RBF interpolator coefficients using a square matrix...");
		// Adding a Tikhonov regularisation parameter if needed
		distanceMatrix += runtimeConfig.m_regularisationParameter * Eigen::MatrixXd::Identity(runtimeConfig.m_coordinates.size(), runtimeConfig.m_coordinates.size());
		// Solving the linear system for a square matrix
		Eigen::FullPivLU<Eigen::MatrixXd> luDecomposition(distanceMatrix);
		Eigen::VectorXd weightsVector = luDecomposition.solve(resultVector);
		// Saving computed coefficients
		m_coefficients = std::vector<FloatingPrecision>(weightsVector.data(), weightsVector.data() + weightsVector.size());

	}
	else {
		LOG_DEBUG("Computing RBF interpolator coefficients using a non-square matrix...");
		// Computing the pseudo-inverse of the matrix
		Eigen::MatrixXd pseudoInverse = distanceMatrix.transpose() * distanceMatrix;
		// Adding a Tikhonov regularisation parameter if needed
		pseudoInverse += runtimeConfig.m_regularisationParameter * Eigen::MatrixXd::Identity(runtimeConfig.m_coordinatesRBF.size(), runtimeConfig.m_coordinatesRBF.size());
		// Solving the linear system for a non-square matrix
		Eigen::FullPivLU<Eigen::MatrixXd> luDecomposition(pseudoInverse);
		Eigen::VectorXd weightsVector = luDecomposition.solve(distanceMatrix.transpose() * resultVector);
		// Saving computed coefficients
		m_coefficients = std::vector<FloatingPrecision>(weightsVector.data(), weightsVector.data() + weightsVector.size());
	}
	
	LOG_DEBUG("Computation of the RBF coefficients completed.");

	LOG_DEBUG("RBF interpolator initialisation completed.");

}

template <typename FloatingPrecision>
FloatingPrecision RBFInterpolator<FloatingPrecision>::interpolate(const std::unique_ptr<Coordinate>& coordinate) const 
{
	FloatingPrecision result = 0;
	for (size_t i = 0; i < m_coordinates.size(); i++) {
		FloatingPrecision distance = m_topology->getDistance(*m_coordinates[i], *coordinate);
		result += m_coefficients[i] * m_kernel(distance);
	}

	// Non-negativity correction made if required
	return glm::pow(result, m_nonNegativityCorrectionParameter);
}

// Explicit instantiation
template class RBFInterpolator<float>;
template class RBFInterpolator<double>;