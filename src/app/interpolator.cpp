#include <fstream>

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

	// Initialising grazing angle function for hemisphere one
	if      (runtimeConfig.m_forceGrazingAnglesNullFunctionHemisphereOne == "none")
		m_grazingAngleFctHemisphereOne = [](FloatingPrecision arg) { return static_cast<FloatingPrecision>(1.); };
	else if (runtimeConfig.m_forceGrazingAnglesNullFunctionHemisphereOne == "cosine")
		m_grazingAngleFctHemisphereOne = [](FloatingPrecision arg) { return glm::cos(arg); };
	else if (runtimeConfig.m_forceGrazingAnglesNullFunctionHemisphereOne == "linear")
		m_grazingAngleFctHemisphereOne = [](FloatingPrecision arg) { return static_cast<FloatingPrecision>(1.) - arg / glm::half_pi<FloatingPrecision>(); };
	else
		LOG_CRITICAL("Invalid grazing angle function for hemisphere one: " 
			+ runtimeConfig.m_forceGrazingAnglesNullFunctionHemisphereOne);
	LOG_VERBOSE("Grazing angle function for hemisphere one initialised to " 
		    + runtimeConfig.m_forceGrazingAnglesNullFunctionHemisphereOne);

	// Initialising grazing angle function for hemisphere two
	if      (runtimeConfig.m_forceGrazingAnglesNullFunctionHemisphereTwo == "none")
		m_grazingAngleFctHemisphereTwo = [](FloatingPrecision arg) { return static_cast<FloatingPrecision>(1.); };
	else if (runtimeConfig.m_forceGrazingAnglesNullFunctionHemisphereTwo == "cosine")
		m_grazingAngleFctHemisphereTwo = [](FloatingPrecision arg) { return glm::cos(arg); };
	else if (runtimeConfig.m_forceGrazingAnglesNullFunctionHemisphereTwo == "linear")
		m_grazingAngleFctHemisphereTwo = [](FloatingPrecision arg) { return static_cast<FloatingPrecision>(1.) - arg / glm::half_pi<FloatingPrecision>(); };
	else
		LOG_CRITICAL("Invalid grazing angle function for hemisphere two: " 
			+ runtimeConfig.m_forceGrazingAnglesNullFunctionHemisphereTwo);
	LOG_VERBOSE("Grazing angle function for hemisphere two initialised to "
		    + runtimeConfig.m_forceGrazingAnglesNullFunctionHemisphereTwo);

	// Adding grazing coordinates if needed
	if (runtimeConfig.m_forceGrazingAnglesNullFunctionHemisphereOne != "none" && runtimeConfig.m_coordinates.size() != 0) {
		runtimeConfig.m_values.push_back(static_cast<FloatingPrecision>(0));
		runtimeConfig.m_coordinates.push_back(std::make_unique<GrazingCoordinate>(1));
		LOG_VERBOSE("Grazing coordinate over hermisphere 1 added to the input data.");
	}
	if (runtimeConfig.m_forceGrazingAnglesNullFunctionHemisphereTwo != "none" && runtimeConfig.m_coordinates.size() != 0) {
		if (runtimeConfig.m_topology->getDimension() == 2)
			LOG_CRITICAL("Grazing coordinates can be set only for topologies > 2D.");
		runtimeConfig.m_values.push_back(static_cast<FloatingPrecision>(0));
		runtimeConfig.m_coordinates.push_back(std::make_unique<GrazingCoordinate>(2));
		LOG_VERBOSE("Grazing coordinate over hermisphere 2 added to the input data.");
	}

	// Initialising the RBF kernel matrix
	Eigen::MatrixXd distanceMatrix = Eigen::MatrixXd::Zero(runtimeConfig.m_coordinates.size(),
		                                                   runtimeConfig.m_coordinatesRBF.size());

	if (runtimeConfig.m_coordinatesRBF.size() > runtimeConfig.m_coordinates.size())
		LOG_WARN("Underdetermined problem: ", runtimeConfig.m_coordinates.size(), " data",
			     " points for ", runtimeConfig.m_coordinatesRBF.size(), " RBF kernels.");

	if (runtimeConfig.m_locationRBFFilePath != "none") {
		// The distance matrix will not be a square matrix, so each
		// coefficient needs to be computed with a two-nested loop.
		LOG_DEBUG("Computing rectangular kernel-distance matrix.");

		size_t iterationNumber = 0;
		for (size_t rowID = 0; rowID < runtimeConfig.m_coordinates.size(); rowID++) {
			for (size_t colID = 0; colID < runtimeConfig.m_coordinatesRBF.size(); colID++) {
				FloatingPrecision distance = m_topology->getDistance(*runtimeConfig.m_coordinates[rowID], 
															         *runtimeConfig.m_coordinatesRBF[colID]);
				distanceMatrix(rowID, colID) = m_kernel(distance);

				if (logger.level >= LogLevel::DEBUG)
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
				FloatingPrecision distance = m_topology->getDistance(*runtimeConfig.m_coordinates[rowID],
																	 *runtimeConfig.m_coordinatesRBF[colID]);
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
		// Adding the non negative correction over the input values:
		resultVector(i) = glm::pow(runtimeConfig.m_values[i], power);
		// Adding grazing angle correction for the first hemisphere:
		if (runtimeConfig.m_forceGrazingAnglesNullFunctionHemisphereOne != "none") {
			if      (typeid(*runtimeConfig.m_coordinates[i]) == typeid(Coordinate2D<FloatingPrecision>)) {
				Coordinate2D<FloatingPrecision>& coord = dynamic_cast<Coordinate2D<FloatingPrecision>&>(*runtimeConfig.m_coordinates[i]);
				resultVector(i) *= m_grazingAngleFctHemisphereOne(coord.getTheta());
			}
			else if (typeid(*runtimeConfig.m_coordinates[i]) == typeid(Coordinate3DSpherical<FloatingPrecision>)) {
				Coordinate3DSpherical<FloatingPrecision>& coord = dynamic_cast<Coordinate3DSpherical<FloatingPrecision>&>(*runtimeConfig.m_coordinates[i]);
				resultVector(i) *= m_grazingAngleFctHemisphereOne(coord.getThetaI());
			}
			else if (typeid(*runtimeConfig.m_coordinates[i]) == typeid(Coordinate3DRusinkiewicz<FloatingPrecision>)) {
				Coordinate3DRusinkiewicz<FloatingPrecision>& coord = dynamic_cast<Coordinate3DRusinkiewicz<FloatingPrecision>&>(*runtimeConfig.m_coordinates[i]);
				resultVector(i) *= m_grazingAngleFctHemisphereOne(coord.getThetaH());
			}
			else if (typeid(*runtimeConfig.m_coordinates[i]) == typeid(GrazingCoordinate))
				resultVector(i) = static_cast<FloatingPrecision>(0);
			else
				LOG_CRITICAL("Unsupported Coordinate type for hemisphere one grazing angle correction.");
		}
		// Adding grazing angle correction for the second hemisphere:
		if (runtimeConfig.m_forceGrazingAnglesNullFunctionHemisphereTwo != "none") {
			if      (typeid(*runtimeConfig.m_coordinates[i]) == typeid(Coordinate3DSpherical<FloatingPrecision>)) {
				Coordinate3DSpherical<FloatingPrecision>& coord = dynamic_cast<Coordinate3DSpherical<FloatingPrecision>&>(*runtimeConfig.m_coordinates[i]);
				resultVector(i) *= m_grazingAngleFctHemisphereTwo(coord.getThetaO());
			}
			else if (typeid(*runtimeConfig.m_coordinates[i]) == typeid(Coordinate3DRusinkiewicz<FloatingPrecision>)) {
				Coordinate3DRusinkiewicz<FloatingPrecision>& coord = dynamic_cast<Coordinate3DRusinkiewicz<FloatingPrecision>&>(*runtimeConfig.m_coordinates[i]);
				resultVector(i) *= m_grazingAngleFctHemisphereTwo(coord.getThetaD());
			}
			else if (typeid(*runtimeConfig.m_coordinates[i]) == typeid(GrazingCoordinate))
				resultVector(i) = static_cast<FloatingPrecision>(0);
			else
				LOG_CRITICAL("Unsupported Coordinate type for hemisphere two grazing angle correction.");
		}
	}

	LOG_DEBUG("Result vector initialised.");

	if (runtimeConfig.m_locationRBFFilePath == "none") {
		LOG_DEBUG("Computing RBF interpolator coefficients using a square matrix...");
		// Adding a Tikhonov regularisation parameter if needed
		distanceMatrix += runtimeConfig.m_regularisationParameter * Eigen::MatrixXd::Identity(runtimeConfig.m_coordinates.size(), runtimeConfig.m_coordinates.size());
		// Solving the linear system for a square matrix
		Eigen::FullPivLU<Eigen::MatrixXd> luDecomposition(distanceMatrix);
		Eigen::VectorXd coefsVector = luDecomposition.solve(resultVector);
		// Saving computed coefficients
		m_coefficients = std::vector<FloatingPrecision>(coefsVector.data(), coefsVector.data() + coefsVector.size());

	}
	else {
		LOG_DEBUG("Computing RBF interpolator coefficients using a non-square matrix...");
		// Computing the pseudo-inverse of the matrix
		Eigen::MatrixXd pseudoInverse = distanceMatrix.transpose() * distanceMatrix;
		// Adding a Tikhonov regularisation parameter if needed
		pseudoInverse += runtimeConfig.m_regularisationParameter * Eigen::MatrixXd::Identity(runtimeConfig.m_coordinatesRBF.size(), runtimeConfig.m_coordinatesRBF.size());
		// Solving the linear system for a non-square matrix
		Eigen::FullPivLU<Eigen::MatrixXd> luDecomposition(pseudoInverse);
		Eigen::VectorXd coefsVector = luDecomposition.solve(distanceMatrix.transpose() * resultVector);
		// Saving computed coefficients
		m_coefficients = std::vector<FloatingPrecision>(coefsVector.data(), coefsVector.data() + coefsVector.size());
	}

	// Saving the coefficients in DEBUG mode
	if (logger.level >= LogLevel::DEBUG) {
		std::ofstream outFile("weights.dat");
		
		if (outFile.is_open()) {
			for (const auto& coef : m_coefficients) outFile << coef << "\n";
			outFile.close();
			LOG_DEBUG("Coefficients saved in weights.dat.");
		}
		else
			LOG_CRITICAL("Impossible to open weights.dat to write coefficients.");
	}

	// Saving the coordinates
	m_coordinates.reserve(runtimeConfig.m_coordinatesRBF.size());
	for (const auto& coord : runtimeConfig.m_coordinatesRBF) {
		if      (typeid(*coord) == typeid(Coordinate2D<FloatingPrecision>))
			m_coordinates.push_back(std::make_unique<Coordinate2D<FloatingPrecision>>(
								  dynamic_cast<const Coordinate2D<FloatingPrecision>&>(*coord)));
		else if (typeid(*coord) == typeid(Coordinate3DSpherical<FloatingPrecision>))
			m_coordinates.push_back(std::make_unique<Coordinate3DSpherical<FloatingPrecision>>(
								  dynamic_cast<const Coordinate3DSpherical<FloatingPrecision>&>(*coord)));
		else if (typeid(*coord) == typeid(Coordinate3DRusinkiewicz<FloatingPrecision>))
			m_coordinates.push_back(std::make_unique<Coordinate3DRusinkiewicz<FloatingPrecision>>(
								  dynamic_cast<const Coordinate3DRusinkiewicz<FloatingPrecision>&>(*coord)));
		else
			LOG_CRITICAL("Unsupported Coordinate type.");
	}

	LOG_DEBUG("RBF interpolator initialisation completed.");

}

template <typename FloatingPrecision>
FloatingPrecision RBFInterpolator<FloatingPrecision>::interpolate(const std::unique_ptr<Coordinate>& coordinate) const 
{
	FloatingPrecision result = static_cast<FloatingPrecision>(0);
	for (size_t i = 0; i < m_coordinates.size(); i++) {
		FloatingPrecision distance = m_topology->getDistance(*m_coordinates[i], *coordinate);
		result += m_coefficients[i] * m_kernel(distance);
	}

	// Grazing angle correction made if required
	if (typeid(*coordinate) == typeid(Coordinate2D<FloatingPrecision>)) {
		Coordinate2D<FloatingPrecision>& coord = dynamic_cast<Coordinate2D<FloatingPrecision>&>(*coordinate);
		result /= m_grazingAngleFctHemisphereOne(coord.getTheta());
	}
	else if (typeid(*coordinate) == typeid(Coordinate3DSpherical<FloatingPrecision>)) {
		Coordinate3DSpherical<FloatingPrecision>& coord = dynamic_cast<Coordinate3DSpherical<FloatingPrecision>&>(*coordinate);
		result /= m_grazingAngleFctHemisphereOne(coord.getThetaI());
		result /= m_grazingAngleFctHemisphereTwo(coord.getThetaO());
	}
	else if (typeid(*coordinate) == typeid(Coordinate3DRusinkiewicz<FloatingPrecision>)) {
		Coordinate3DRusinkiewicz<FloatingPrecision>& coord = dynamic_cast<Coordinate3DRusinkiewicz<FloatingPrecision>&>(*coordinate);
		result /= m_grazingAngleFctHemisphereOne(coord.getThetaH());
		result /= m_grazingAngleFctHemisphereTwo(coord.getThetaD());
	}
	else if (typeid(*coordinate) == typeid(GrazingCoordinate)) {
		result = static_cast<FloatingPrecision>(0);
	}
	else
		LOG_CRITICAL("Unsupported Coordinate type.");

	// Non-negativity correction made if required
	return glm::pow(result, m_nonNegativityCorrectionParameter);
}

template <typename FloatingPrecision>
void RBFInterpolator<FloatingPrecision>::reduceInterpolator(const FloatingPrecision& threshold)
{
	for (auto iterator = m_coefficients.begin(); iterator != m_coefficients.end(); ) {
		if (glm::abs(*iterator) < threshold) {
			iterator = m_coefficients.erase(iterator);
			m_coordinates.erase(m_coordinates.begin() + (iterator - m_coefficients.begin()));
		}
		else
			++iterator;
	}

	LOG_DEBUG("RBF Interpolator reduced to ", m_coefficients.size(), 
		     " coefficients. Threshold has been set to ", threshold);
}

template <typename FloatingPrecision>
std::vector<FloatingPrecision> RBFInterpolator<FloatingPrecision>::getCoefficients() const
{
	return m_coefficients;
}

// Explicit instantiation
template class RBFInterpolator<float>;
template class RBFInterpolator<double>;