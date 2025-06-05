#pragma once

#include <fstream>

#include <glm/glm.hpp>

#include "coordinate.hpp"
#include "logger.hpp"
#include "topology.hpp"


// Forward declarations
template <typename ReturnType>
class RBFModel;

// Type aliases
using RBFModelf = RBFModel<float>;
using RBFModeld = RBFModel<double>;
using RBFModelvec3 = RBFModel<glm::vec3>;


// Floating precision of the data
template <typename ReturnType>
struct FloatingPrecision {
	using type = ReturnType;
};

// Specialisation for glm::vec3
template <>
struct FloatingPrecision<glm::vec3> {
	using type = float;
};

// Alias for the floating precision type
template <typename ReturnType>
using FloatingPrecision_t = typename FloatingPrecision<ReturnType>::type;

// Power function wrapper
template <typename ReturnType>
ReturnType power(const ReturnType& base, const FloatingPrecision_t<ReturnType>& exponent) {
	return glm::pow(base, exponent);
}

// Power function wrapper for glm::vec3
template <typename ReturnType>
glm::vec3 power(const glm::vec3& base, const ReturnType& exponent) {
	return glm::vec3(glm::pow(base.x, exponent), glm::pow(base.y, exponent), glm::pow(base.z, exponent));
}

/*
 * @brief
 *
 * tparam ReturnType: Type of the return value. Can be generally float,
 *        or double, or glm::vec3. This will be defined by the datafile
 *        .RBFCoeffs.
 *
 */
template <typename ReturnType>
class RBFModel {

public:
	/*
	 * @brief Reads a file containing the parameters of a RBF model, and
	 *        then instantiates and returns a RBFModel instance that can
	 *        be used directly to compute the BRDF.
	 *
	 * @note Can be used either to just read the data of the file only,
			 or to instantiate the model.
	 *
	 * @param filePath : [string] Absolute or relative path to the file.
	 *                  The file must have the type format '.RBFCoeff'
	 * @param verbose  : [bool] If true, prints the content of the file.
	 * @param clusterID: [int] ID of the cluster to read. Default is 0.
	 *                   If set to -1, will read all the clusters.
	 *
	 * @return [RBFModel] object containing the parameters of the model.
	 *         Can be then used to compute the BRDF.
	 */
	static RBFModel readFile(const std::string& filePath, const bool verbose = false, const int& clusterID = 0) {
		// Initialising model
		RBFModel model;

		// Setting the logger level
		if (verbose)
			logger.level = LogLevel::VERBOSE;
		else
			logger.level = LogLevel::WARN;

		// Opening the file
		std::ifstream file(filePath, std::ios::binary);
		if (!file.is_open())
			LOG_CRITICAL("File ", filePath, " could not be opened.");
		else
			LOG_INFO("File opened.");

		// Reading the header of the file
		int versionMajor, versionMinor, versionPatch, dataType, dimension;
		file.read(reinterpret_cast<char*>(&versionMajor), sizeof(int));
		file.read(reinterpret_cast<char*>(&versionMinor), sizeof(int));
		file.read(reinterpret_cast<char*>(&versionPatch), sizeof(int));
		LOG_INFO("File written in version ", versionMajor,
										".", versionMinor,
									    ".", versionPatch);

		file.read(reinterpret_cast<char*>(&dataType), sizeof(int));
		std::string floatPrecision, dataTypeString;
		if      (dataType == 0) {
			floatPrecision = "single";
			dataTypeString = "scalar";
		}
		else if (dataType == 1) {
			floatPrecision = "double";
			dataTypeString = "scalar";
		}
		else if (dataType == 2) {
			floatPrecision = "single";
			dataTypeString = "RGB";
		}
		else if (dataType == 3) {
			floatPrecision = "double";
			dataTypeString = "RGB";
		}
		else
			LOG_CRITICAL("Invalid data type : ", dataType);

		// Checking if the user has chosen the good instance of RBFModel
		if ((typeid(ReturnType) == typeid(float)      && dataType == 0) ||
			(typeid(ReturnType) == typeid(double)     && dataType == 1) ||
			(typeid(ReturnType) == typeid(glm::vec3)  && dataType == 2) ||
			(typeid(ReturnType) == typeid(glm::dvec3) && dataType == 3))
			LOG_INFO("Data saved as ", dataTypeString, " in ", floatPrecision, " precision");
		else {
			LOG_WARN("Wrong instanciation of RBFModel. Input data is ", 
				     dataTypeString, " ", floatPrecision, " precision.");
			if      (dataType == 0)
				LOG_CRITICAL("Please declare it as RBFModelf model = RBFModel<float>::readFile(...)");
			else if (dataType == 1)
				LOG_CRITICAL("Please declare it as RBFModeld model = RBFModel<double>::readFile(...)");
			else if (dataType == 2)
				LOG_CRITICAL("Please declare it as RBFModelvec3 model = RBFModel<glm::vec3>::readFile(...)");
			else if (dataType == 3)
				LOG_CRITICAL("Please declare it as RBFModeldvec3 model = RBFModel<glm::dvec3>::readFile(...)");
			else
				LOG_CRITICAL("The input type is currently not supported.");
		}

		file.read(reinterpret_cast<char*>(&dimension), sizeof(int));
		if (dimension >= 2 && dimension <= 4)
			LOG_INFO("BRDF dimensions: ", dimension);
		else
			LOG_CRITICAL("Invalid number of dimensions: ", dimension);

		char parameterisation[4] = { 0 }; // Fourth character is null terminator
		file.read(parameterisation, 3);
		std::string parameterisationStr(parameterisation);
		if (parameterisationStr == "SPH")
			model.m_parameterisation = "spherical";
		else
			LOG_CRITICAL("Invalid parameterisation: ", parameterisationStr);
		LOG_INFO("Parameterisation: ", model.m_parameterisation);

		char smoothingFunctionHemisphereOne[4] = { 0 }; // Fourth character is null terminator
		file.read(smoothingFunctionHemisphereOne, 3);
		std::string smoothingFunctionHemisphereOneStr(smoothingFunctionHemisphereOne);
		if      (smoothingFunctionHemisphereOneStr == "NUL") {
			model.m_smoothingFunctionHemisphereOne = [](FloatingPrecision_t<ReturnType> arg) { return static_cast<FloatingPrecision_t<ReturnType>>(1.); };
			LOG_INFO("Smoothing function hemisphere one: none");
		}
		else if (smoothingFunctionHemisphereOneStr == "COS") {
			model.m_smoothingFunctionHemisphereOne = [](FloatingPrecision_t<ReturnType> arg) { return glm::cos(arg); };
			LOG_INFO("Smoothing function hemisphere one: cosine");
		}
		else
			LOG_CRITICAL("Invalid smoothing function hemisphere one: ", smoothingFunctionHemisphereOneStr);

		char smoothingFunctionHemisphereTwo[4] = { 0 }; // Fourth character is null terminator
		file.read(smoothingFunctionHemisphereTwo, 3);
		std::string smoothingFunctionHemisphereTwoStr(smoothingFunctionHemisphereTwo);
		if (smoothingFunctionHemisphereTwoStr == "NUL") {
			model.m_smoothingFunctionHemisphereTwo = [](FloatingPrecision_t<ReturnType> arg) { return static_cast<FloatingPrecision_t<ReturnType>>(1.); };
			LOG_INFO("Smoothing function hemisphere two: none");
		}
		else if (smoothingFunctionHemisphereTwoStr == "COS") {
			model.m_smoothingFunctionHemisphereTwo = [](FloatingPrecision_t<ReturnType> arg) { return glm::cos(arg); };
			LOG_INFO("Smoothing function hemisphere two: cosine");
		}
		else
			LOG_CRITICAL("Invalid smoothing function hemisphere two: ", smoothingFunctionHemisphereTwoStr);

		float regularisationParameter;
		file.read(reinterpret_cast<char*>(&regularisationParameter), sizeof(float));
		if (regularisationParameter >= 0)
			LOG_INFO("Regularisation parameter: ", regularisationParameter);
		else
			LOG_CRITICAL("Invalid regularisation parameter (must be >= 0): ", regularisationParameter);

		int nonNegativityParameter;
		file.read(reinterpret_cast<char*>(&nonNegativityParameter), sizeof(int));
		if (nonNegativityParameter == 1 || nonNegativityParameter % 2 == 0) {
			model.m_nonNegativityCorrectionParameter = static_cast<FloatingPrecision_t<ReturnType>>(nonNegativityParameter);
			LOG_INFO("Non-negativity correction parameter: ", nonNegativityParameter);
		}
		else
			LOG_CRITICAL("Invalid non-negativity correction (should be a power of 2): ", nonNegativityParameter);

		// Defining the topology
		bool forceReciprocity, forceBilateralSymmetry;
		file.read(reinterpret_cast<char*>(&forceReciprocity)      , sizeof(bool));
		file.read(reinterpret_cast<char*>(&forceBilateralSymmetry), sizeof(bool));
		if (dimension == 2) {
			if (forceBilateralSymmetry) {
				model.m_topology = std::make_unique<Topology2DSym<FloatingPrecision_t<ReturnType>>>();
				LOG_INFO("BRDF topology initialised as 2D symmetrical.");
			}
			else {
				model.m_topology = std::make_unique<Topology2D<FloatingPrecision_t<ReturnType>>>();
				LOG_INFO("BRDF topology initialised as 2D.");
			}
		}
		else if (dimension == 3) {
			if (forceBilateralSymmetry) {
				if (forceReciprocity) {
					model.m_topology = std::make_unique<Topology3DSphRecSym<FloatingPrecision_t<ReturnType>>>();
					LOG_INFO("BRDF topology initialised as 3D reciprocal symmetrical.");
				}
				else {
					model.m_topology = std::make_unique<Topology3DSphSym<FloatingPrecision_t<ReturnType>>>();
					LOG_INFO("BRDF topology initialised as 3D symmetrical.");
				}
			}
			else {
				if (forceReciprocity) {
					model.m_topology = std::make_unique<Topology3DSphRec<FloatingPrecision_t<ReturnType>>>();
					LOG_INFO("BRDF topology initialised as 3D reciprocal.");
				}
				else {
					model.m_topology = std::make_unique<Topology3DSph<FloatingPrecision_t<ReturnType>>>();
					LOG_INFO("BRDF topology initialised as 3D.");
				}
			}
		}
		else
			LOG_CRITICAL("RIVOLI currently supports only 2D or 3D BRDF"
						 ". Given number of dimensions: ", dimension);

		// Defining the kernel
		char kernel[4] = { 0 }; // Fourth character is null terminator
		int  kernelNbParams;
		file.read(kernel, 3);
		file.read(reinterpret_cast<char*>(&kernelNbParams), sizeof(int));
		std::string kernelStr(kernel);
		if (kernelStr == "LIN" && kernelNbParams == 0) {
			model.m_kernel = [](FloatingPrecision_t<ReturnType> arg) { return arg; };
			LOG_INFO("Kernel: linear");
		}
		else
			LOG_CRITICAL("Invalid kernel: ", kernelStr, " with number of parameters: ", kernelNbParams);

		float thresholdCoefficient;
		file.read(reinterpret_cast<char*>(&thresholdCoefficient), sizeof(float));
		if (thresholdCoefficient >= 0) {
			LOG_INFO("Threshold coefficient: ", thresholdCoefficient);
			model.m_thresholdCoefficient = static_cast<FloatingPrecision_t<ReturnType>>(thresholdCoefficient);
		}
		else
			LOG_CRITICAL("Invalid threshold coefficient (must be >= 0): ", thresholdCoefficient);

		int nbClusters;
		file.read(reinterpret_cast<char*>(&nbClusters), sizeof(int));
		if (nbClusters == 0) nbClusters++;
		if      (clusterID > nbClusters + 1)
			LOG_CRITICAL("Invalid cluster required: ", clusterID, ". Only ", nbClusters, "clusters in file ", filePath);
		else if (nbClusters <= 0)
			LOG_CRITICAL("Invalid number of clusters (must be >= 0): ", nbClusters);
		else
			LOG_INFO("Number of clusters: ", nbClusters);

		bool uniqueLocationRBF;
		file.read(reinterpret_cast<char*>(&uniqueLocationRBF), sizeof(bool));
		LOG_INFO("Unique location RBF: ", uniqueLocationRBF ? "true" : "false");

		file.read(reinterpret_cast<char*>(&model.m_nbRBF), sizeof(int));
		if (model.m_nbRBF > 0)
			LOG_INFO("Number of RBF: ", model.m_nbRBF);
		else
			LOG_CRITICAL("Invalid number of RBF models (must be > 0): ", model.m_nbRBF);

		std::streampos positionEndOfHeader = file.tellg();

		for (size_t localClusterID = (clusterID >= 0 ? clusterID : 0);
			 localClusterID < (clusterID >= 0 ? clusterID + 1 : nbClusters);
			 localClusterID++) {

			// Repositioning the cursor
			file.clear(); file.seekg(positionEndOfHeader, std::ios::beg);

			// Reading of the header of the file over
			// Reading and saving the RBF coordinates
			if (!(uniqueLocationRBF)) {
				// Moving the cursor to pass RBF coordinates
				std::streampos deltaStreamPosition = dimension * localClusterID * model.m_nbRBF * sizeof(FloatingPrecision_t<ReturnType>);
				file.seekg(deltaStreamPosition, std::ios::cur);
				// Moving the cursor to pass RBF weights
				deltaStreamPosition = localClusterID * model.m_nbRBF * sizeof(ReturnType);
				file.seekg(deltaStreamPosition, std::ios::cur);
			}

			for (size_t i = 0; i < model.m_nbRBF; i++) {
				if (file.eof()) // Check if the file has ended prematurely
					LOG_CRITICAL("Reached EOF unexpectedly at coordinate number ", i);

				if (!file) // Checks if the file is still in a good state
					LOG_CRITICAL("Error reading the file at coordinate number ", i);

				if (dimension == 2) {
					FloatingPrecision_t<ReturnType> theta, phi;
					file.read(reinterpret_cast<char*>(&theta), sizeof(FloatingPrecision_t<ReturnType>));
					file.read(reinterpret_cast<char*>(&phi), sizeof(FloatingPrecision_t<ReturnType>));
					model.m_coordinates.push_back(std::make_unique<Coordinate2D<FloatingPrecision_t<ReturnType>>>(theta, phi));
				}
				else if (dimension == 3) {
					FloatingPrecision_t<ReturnType> thetaOne, thetaTwo, phiTwo;
					file.read(reinterpret_cast<char*>(&thetaOne), sizeof(FloatingPrecision_t<ReturnType>));
					file.read(reinterpret_cast<char*>(&thetaTwo), sizeof(FloatingPrecision_t<ReturnType>));
					file.read(reinterpret_cast<char*>(&phiTwo), sizeof(FloatingPrecision_t<ReturnType>));
					if (model.m_parameterisation == "spherical")
						model.m_coordinates.push_back(std::make_unique<Coordinate3DSpherical<FloatingPrecision_t<ReturnType>>>(thetaOne, thetaTwo, phiTwo));
					else if (model.m_parameterisation == "rusinkiewicz")
						model.m_coordinates.push_back(std::make_unique<Coordinate3DRusinkiewicz<FloatingPrecision_t<ReturnType>>>(thetaOne, thetaTwo, phiTwo));
					else
						LOG_CRITICAL("Invalid parameterisation: ", model.m_parameterisation);
				}
				else
					LOG_CRITICAL("Invalid number of dimensions: ", dimension);
			}

			if (uniqueLocationRBF) {
				// Moving the cursor to pass RBF Weights
				std::streampos deltaStreamPosition = localClusterID * model.m_nbRBF * sizeof(ReturnType);
				file.seekg(deltaStreamPosition, std::ios::cur);
			}

			// Reading and saving the RBF weights
			for (size_t rbfID = 0; rbfID < model.m_nbRBF; rbfID++) {
				ReturnType weight;
				file.read(reinterpret_cast<char*>(&weight), sizeof(ReturnType));
				model.m_weights.push_back(weight);
			}

		}

		return model;
	}

	/*
	 * @brief Computes the BRDF value for a given pair of incident and
	 *        observed directions.
	 *
	 * @note If the .RBFCoeff file has multiple RBF models stored, the
	 *       cluster parameter can be used to select the one needed.
	 *       theta and phi follow the convention of physicists, being
	 *       respectively the zenithal (or colatitude) and azimuthal
	 *       (or longitude) angle in spherical coordinates. They are
	 *       expressed in radians.
	 *
	 * @tparam ReturnType: Type of the return value. Can be generally
	 *         float or double, but can be any type that supports the
	 *         .RBFCoeff file, such as glm::vec3 is the data is RGB.
	 *
	 * @param wi: [glm::vec2] 2D vector containing for the incident
	 *            direction its spherical coordinates theta and phi
	 * @param wo: [glm::vec2] 2D vector containing for the observed
	 *            direction its spherical coordinates theta and phi
	 *
	 * @return [ReturnType] value of the BRDF for the given directions.
	 */
	ReturnType eval(glm::vec2 wi, glm::vec2 wo, size_t clusterID = 0) {
		// Converting the input coordinate to the right type
		FloatingPrecision_t<ReturnType> thetaI = static_cast<FloatingPrecision_t<ReturnType>>(wi.x);
		FloatingPrecision_t<ReturnType> phiI   = static_cast<FloatingPrecision_t<ReturnType>>(wi.y);
		FloatingPrecision_t<ReturnType> thetaO = static_cast<FloatingPrecision_t<ReturnType>>(wo.x);
		FloatingPrecision_t<ReturnType> phiO   = static_cast<FloatingPrecision_t<ReturnType>>(wo.y);
		std::unique_ptr<Coordinate> inputCoordinate;
		if      (m_topology->getDimension() == 2)
			inputCoordinate = std::make_unique<Coordinate2D<FloatingPrecision_t<ReturnType>>>(thetaO, phiO);
		else if (m_topology->getDimension() == 3) {
			if      (m_parameterisation == "spherical")
				inputCoordinate = std::make_unique<Coordinate3DSpherical<FloatingPrecision_t<ReturnType>>>(thetaI, thetaO, phiO);
			else if (m_parameterisation == "rusinkiewicz")
				inputCoordinate = std::make_unique<Coordinate3DRusinkiewicz<FloatingPrecision_t<ReturnType>>>(thetaI, thetaO, phiO);
			else
				LOG_CRITICAL("Invalid parameterisation: ", m_parameterisation);
		}
		else
			LOG_CRITICAL("Invalid number of dimensions: ", m_topology->getDimension());

		ReturnType result = ReturnType(0);
		for (size_t i = 0; i < m_nbRBF; i++) {
			// Compute the distance between the input bidirection and the stored one for the RBF weight
			FloatingPrecision_t<ReturnType> distance = static_cast<FloatingPrecision_t<ReturnType>>(0);

			// Compute the distance between the input bidirection and the stored one for the RBF weight
			distance = m_topology->getDistance(*inputCoordinate, *m_coordinates[i]);

			FloatingPrecision_t<ReturnType> localRBFResult = static_cast<FloatingPrecision_t<ReturnType>>(0);
			localRBFResult = m_kernel(distance);

			result += m_weights[clusterID * (m_nbRBF) + i] * localRBFResult;
		}

		// Grazing angle correction made if required
		if      (m_topology->getDimension() == 2) {
			Coordinate2D<FloatingPrecision_t<ReturnType>>& coord = dynamic_cast<Coordinate2D<FloatingPrecision_t<ReturnType>>&>(*inputCoordinate);
			result /= m_smoothingFunctionHemisphereOne(coord.getTheta());
		}
		else if (m_topology->getDimension() == 3) {
			if      (m_parameterisation == "spherical") {
				Coordinate3DSpherical<FloatingPrecision_t<ReturnType>>& coord = dynamic_cast<Coordinate3DSpherical<FloatingPrecision_t<ReturnType>>&>(*inputCoordinate);
				result /= m_smoothingFunctionHemisphereOne(coord.getThetaI());
				result /= m_smoothingFunctionHemisphereTwo(coord.getThetaO());
			}
			else if (m_parameterisation == "rusinkiewicz") {
				Coordinate3DRusinkiewicz<FloatingPrecision_t<ReturnType>>& coord = dynamic_cast<Coordinate3DRusinkiewicz<FloatingPrecision_t<ReturnType>>&>(*inputCoordinate);
				result /= m_smoothingFunctionHemisphereOne(coord.getThetaH());
				result /= m_smoothingFunctionHemisphereTwo(coord.getThetaD());
			}
			else
				LOG_CRITICAL("Invalid parameterisation: ", m_parameterisation);
		}
		else
			LOG_CRITICAL("Invalid number of dimensions: ", m_topology->getDimension());

		// Add non-negativity correction if needed
		result = power(result, m_nonNegativityCorrectionParameter);

		return result;
	}

	FloatingPrecision_t<ReturnType> eval(glm::vec2 wi, glm::vec2 wo, size_t clusterID = 0, size_t channel = 0) {
		if ((typeid(ReturnType) == typeid(float)) || (typeid(ReturnType) == typeid(double)))
			LOG_CRITICAL("eval() function called with four arguments only works for RGB models."
				         " Please call eval() with three arguments only for scalar BRDF models.");
		if (channel < 0 || channel > 2)
			LOG_CRITICAL("Invalid channel: ", channel, ". Must be among {0: Red, 1: Green, 2: Blue}.");

		// Converting the input coordinate to the right type
		FloatingPrecision_t<ReturnType> thetaI = static_cast<FloatingPrecision_t<ReturnType>>(wi.x);
		FloatingPrecision_t<ReturnType> phiI   = static_cast<FloatingPrecision_t<ReturnType>>(wi.y);
		FloatingPrecision_t<ReturnType> thetaO = static_cast<FloatingPrecision_t<ReturnType>>(wo.x);
		FloatingPrecision_t<ReturnType> phiO   = static_cast<FloatingPrecision_t<ReturnType>>(wo.y);
		std::unique_ptr<Coordinate> inputCoordinate;
		if (m_topology->getDimension() == 2)
			inputCoordinate = std::make_unique<Coordinate2D<FloatingPrecision_t<ReturnType>>>(thetaO, phiO);
		else if (m_topology->getDimension() == 3) {
			if (m_parameterisation == "spherical")
				inputCoordinate = std::make_unique<Coordinate3DSpherical<FloatingPrecision_t<ReturnType>>>(thetaI, thetaO, phiO);
			else if (m_parameterisation == "rusinkiewicz")
				inputCoordinate = std::make_unique<Coordinate3DRusinkiewicz<FloatingPrecision_t<ReturnType>>>(thetaI, thetaO, phiO);
			else
				LOG_CRITICAL("Invalid parameterisation: ", m_parameterisation);
		}
		else
			LOG_CRITICAL("Invalid number of dimensions: ", m_topology->getDimension());

		FloatingPrecision_t<ReturnType> result = static_cast<FloatingPrecision_t<ReturnType>>(0);
		for (size_t i = 0; i < m_nbRBF; i++) {

			// Ignore some of the RBF it their associated is negligible compared to the threshold
			FloatingPrecision_t<ReturnType> weight = m_weights[clusterID * (m_nbRBF)+i][channel];
			if (glm::abs(weight) <= m_thresholdCoefficient)
				continue;

			// Compute the distance between the input bidirection and the stored one for the RBF weight
			FloatingPrecision_t<ReturnType> distance = static_cast<FloatingPrecision_t<ReturnType>>(0);

			// Compute the distance between the input bidirection and the stored one for the RBF weight
			distance = m_topology->getDistance(*inputCoordinate, *m_coordinates[i]);

			FloatingPrecision_t<ReturnType> localRBFResult = static_cast<FloatingPrecision_t<ReturnType>>(0);
			localRBFResult = m_kernel(distance);

			if constexpr (std::is_same_v<ReturnType, glm::vec3> || std::is_same_v<ReturnType, glm::dvec3>)
				result += weight * localRBFResult;
		}

		// Grazing angle correction made if required
		if      (m_topology->getDimension() == 2) {
			Coordinate2D<FloatingPrecision_t<ReturnType>>& coord = dynamic_cast<Coordinate2D<FloatingPrecision_t<ReturnType>>&>(*inputCoordinate);
			result /= m_smoothingFunctionHemisphereOne(coord.getTheta());
		}
		else if (m_topology->getDimension() == 3) {
			if (m_parameterisation == "spherical") {
				Coordinate3DSpherical<FloatingPrecision_t<ReturnType>>& coord = dynamic_cast<Coordinate3DSpherical<FloatingPrecision_t<ReturnType>>&>(*inputCoordinate);
				result /= m_smoothingFunctionHemisphereOne(coord.getThetaI());
				result /= m_smoothingFunctionHemisphereTwo(coord.getThetaO());
			}
			else if (m_parameterisation == "rusinkiewicz") {
				Coordinate3DRusinkiewicz<FloatingPrecision_t<ReturnType>>& coord = dynamic_cast<Coordinate3DRusinkiewicz<FloatingPrecision_t<ReturnType>>&>(*inputCoordinate);
				result /= m_smoothingFunctionHemisphereOne(coord.getThetaH());
				result /= m_smoothingFunctionHemisphereTwo(coord.getThetaD());
			}
			else
				LOG_CRITICAL("Invalid parameterisation: ", m_parameterisation);
		}
		else
			LOG_CRITICAL("Invalid number of dimensions: ", m_topology->getDimension());

		// Add non-negativity correction if needed
		result = power(result, m_nonNegativityCorrectionParameter);

		return result;
	}

	FloatingPrecision_t<ReturnType> m_nonNegativityCorrectionParameter;
	FloatingPrecision_t<ReturnType> m_thresholdCoefficient;

	std::string m_parameterisation;
	std::unique_ptr<Topology<FloatingPrecision_t<ReturnType>>> m_topology;

	std::function<FloatingPrecision_t<ReturnType>(FloatingPrecision_t<ReturnType>)> m_smoothingFunctionHemisphereOne;
	std::function<FloatingPrecision_t<ReturnType>(FloatingPrecision_t<ReturnType>)> m_smoothingFunctionHemisphereTwo;
	std::function<FloatingPrecision_t<ReturnType>(FloatingPrecision_t<ReturnType>)> m_kernel;

	std::vector<std::unique_ptr<Coordinate>> m_coordinates;
	std::vector<ReturnType> m_weights;

	int m_nbRBF;

};