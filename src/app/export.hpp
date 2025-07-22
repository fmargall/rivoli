#pragma once

#include <fstream>
#include <mutex>
#include <omp.h>
#include <string>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>

#include "arrays.hpp"
#include "configurator.hpp"
#include "coordinate.hpp"
#include "interpolator.hpp"
#include "logger.hpp"
#include "../version.hpp"

// Forward declarations
template <typename FloatingPrecision>
class RuntimeConfig;

template <typename FloatingPrecision>
class RBFInterpolator;

// Mutex for writing to the output file
extern std::mutex writeFileMutex;

template <typename FloatingPrecision>
void initRBFCoeffsFile(
	const std::string& outputFilePath,
	const RuntimeConfig<FloatingPrecision>& runtimeConfig,

	const bool& isRGB
) {
	// Opening the output .RBFCoeffs file. If it exists, it will be overwritten.
	std::ofstream outputDataFile(outputFilePath, std::ios::binary | std::ios::trunc);
	if (outputDataFile.is_open())
		LOG_DEBUG("Output file ", outputFilePath, " opened successfully.");
	else
		LOG_CRITICAL("Output file ", outputFilePath, " could not be opened.");

	// Writing the version of the software
	int versionMajor = RIVOLI_VERSION_MAJOR, versionMinor = RIVOLI_VERSION_MINOR, versionPatch = RIVOLI_VERSION_PATCH;
	outputDataFile.write(reinterpret_cast<char*>(&versionMajor), sizeof(int));
	outputDataFile.write(reinterpret_cast<char*>(&versionMinor), sizeof(int));
	outputDataFile.write(reinterpret_cast<char*>(&versionPatch), sizeof(int));
	LOG_TRACE("RIVOLI version (", RIVOLI_VERSION, ") added to output file.");

	// Writing data type
	int dataType;
	if (isRGB) // 2 for float RGB, 3 for double RGB
		dataType = (runtimeConfig.m_floatingPointPrecision == "FP32") ? 2 : 3;
	else       // 0 for float, 1 for double
		dataType = (runtimeConfig.m_floatingPointPrecision == "FP32") ? 0 : 1;
	outputDataFile.write(reinterpret_cast<char*>(&dataType), sizeof(int));
	
	// Writing the number of dimensions
	int nbDimensions = static_cast<int>(runtimeConfig.m_nbDimensions);
	outputDataFile.write(reinterpret_cast<char*>(&nbDimensions), sizeof(int));

	// Writing the parameterisation: SPH for spherical, RUS for Rusinkiewicz
	std::string parameterisation = (runtimeConfig.m_inputParameterisation == "spherical") ? "SPH" : "RUS";
	outputDataFile.write(parameterisation.c_str(), parameterisation.size());

	// Writing the smoothing function for hemisphere one
	std::string smoothingFunctionHemisphereOne;
	if      (runtimeConfig.m_forceGrazingAnglesNullFunctionHemisphereOne == "linear")
		smoothingFunctionHemisphereOne = "LIN";
	else if (runtimeConfig.m_forceGrazingAnglesNullFunctionHemisphereOne == "cosine")
		smoothingFunctionHemisphereOne = "COS";
	else if (runtimeConfig.m_forceGrazingAnglesNullFunctionHemisphereOne == "none")
		smoothingFunctionHemisphereOne = "NUL";
	else
		LOG_CRITICAL("Invalid smoothing function for hemisphere one: " + runtimeConfig.m_forceGrazingAnglesNullFunctionHemisphereOne);
	outputDataFile.write(smoothingFunctionHemisphereOne.c_str(), smoothingFunctionHemisphereOne.size());

	// Writing the smoothing function for hemisphere two
	std::string smoothingFunctionHemisphereTwo;
	if      (runtimeConfig.m_forceGrazingAnglesNullFunctionHemisphereTwo == "linear")
		smoothingFunctionHemisphereTwo = "LIN";
	else if (runtimeConfig.m_forceGrazingAnglesNullFunctionHemisphereTwo == "cosine")
		smoothingFunctionHemisphereTwo = "COS";
	else if (runtimeConfig.m_forceGrazingAnglesNullFunctionHemisphereTwo == "none")
		smoothingFunctionHemisphereTwo = "NUL";
	else
		LOG_CRITICAL("Invalid smoothing function for hemisphere two: " + runtimeConfig.m_forceGrazingAnglesNullFunctionHemisphereTwo);
	outputDataFile.write(smoothingFunctionHemisphereTwo.c_str(), smoothingFunctionHemisphereTwo.size());

	float regularisationParameter = static_cast<float>(runtimeConfig.m_regularisationParameter);
	outputDataFile.write(reinterpret_cast<char*>(&regularisationParameter), sizeof(float));
	int nonNegativityCorrectionParameterInt = static_cast<int>(runtimeConfig.m_nonNegativityCorrectionParameter);
	outputDataFile.write(reinterpret_cast<char*>(&nonNegativityCorrectionParameterInt), sizeof(int));

	bool forceReciprocity = runtimeConfig.m_forceReciprocity;
	bool forceBilateralSymmetry = runtimeConfig.m_forceBilateralSymmetry;
	outputDataFile.write(reinterpret_cast<char*>(&forceReciprocity), sizeof(bool));
	outputDataFile.write(reinterpret_cast<char*>(&forceBilateralSymmetry), sizeof(bool));

	// Writing the RBF kernel
	std::string kernel;
	int kernelNbParams;
	if (runtimeConfig.m_kernel == "linear") {
		kernel = "LIN";
		kernelNbParams = 0;
	}
	outputDataFile.write(kernel.c_str(), kernel.size());
	outputDataFile.write(reinterpret_cast<char*>(&kernelNbParams), sizeof(int));

	// Writing the threshold coefficient
	float thresholdCoef = runtimeConfig.m_thresholdCoef;
	outputDataFile.write(reinterpret_cast<char*>(&thresholdCoef), sizeof(float));
	// Writing the number of clusters
	int nbClusters = runtimeConfig.m_nbClusters;
	outputDataFile.write(reinterpret_cast<char*>(&nbClusters), sizeof(int));
	// Writing if unique location RBF mode
	bool uniqueLocationRBF = runtimeConfig.m_uniqueLocationRBF;
	outputDataFile.write(reinterpret_cast<char*>(&uniqueLocationRBF), sizeof(bool));
	// Writing number of RBF per cluster
	int nbRBF = runtimeConfig.m_coordinatesRBF.size();
	outputDataFile.write(reinterpret_cast<char*>(&nbRBF), sizeof(int));

	if (uniqueLocationRBF) {
		// Writing the RBF coordinates
		for (size_t coordinateID = 0; coordinateID < runtimeConfig.m_coordinatesRBF.size(); coordinateID++) {
			const auto& coordinate = runtimeConfig.m_coordinatesRBF[coordinateID];
			if (typeid(*coordinate) == typeid(Coordinate2D<FloatingPrecision>)) {
				const auto& coord = static_cast<Coordinate2D<FloatingPrecision>&>(*coordinate);
				FloatingPrecision theta = coord.getTheta();
				FloatingPrecision phi   = coord.getPhi();
				outputDataFile.write(reinterpret_cast<char*>(&theta), sizeof(FloatingPrecision));
				outputDataFile.write(reinterpret_cast<char*>(&phi)  , sizeof(FloatingPrecision));
			}
			else if (typeid(*coordinate) == typeid(Coordinate3DSpherical<FloatingPrecision>)) {
				const auto& coord = static_cast<Coordinate3DSpherical<FloatingPrecision>&>(*coordinate);
				FloatingPrecision thetaI   = coord.getThetaI();
				FloatingPrecision thetaO   = coord.getThetaO();
				FloatingPrecision deltaPhi = coord.getDeltaPhi();
				outputDataFile.write(reinterpret_cast<char*>(&thetaI)  , sizeof(FloatingPrecision));
				outputDataFile.write(reinterpret_cast<char*>(&thetaO)  , sizeof(FloatingPrecision));
				outputDataFile.write(reinterpret_cast<char*>(&deltaPhi), sizeof(FloatingPrecision));
			}
			else if (typeid(*coordinate) == typeid(Coordinate3DRusinkiewicz<FloatingPrecision>)) {
				const auto& coord = static_cast<Coordinate3DRusinkiewicz<FloatingPrecision>&>(*coordinate);
				FloatingPrecision thetaH = coord.getThetaH();
				FloatingPrecision thetaD = coord.getThetaD();
				FloatingPrecision phiD   = coord.getPhiD();
				outputDataFile.write(reinterpret_cast<char*>(&thetaH), sizeof(FloatingPrecision));
				outputDataFile.write(reinterpret_cast<char*>(&thetaD), sizeof(FloatingPrecision));
				outputDataFile.write(reinterpret_cast<char*>(&phiD)  , sizeof(FloatingPrecision));
			}
			else
				LOG_CRITICAL("Invalid coordinate type.");
		}
	}

	outputDataFile.close();

	LOG_DEBUG(outputFilePath, " initialised successfully.");
}

template <typename FloatingPrecision, typename DataType>
void writeToRBFCoeffs(
	const std::string& outputFilePath,
	
	const RuntimeConfig<FloatingPrecision>& runtimeConfig,
	const std::vector<DataType>& inputData,
	const size_t& clusterID = 0)
{
	// Lock mutex to protect the file access for writing
	std::lock_guard<std::mutex> guard(writeFileMutex);

	// Size of the header can vary depending on
	// the kernel and its number of parameters.
	std::streampos kernelNbParamsStreamPosition = 
		5 *     sizeof(int)   + // 3: v.MAJOR.MINOR.PATCH + 1: dataType + 1: nbDimensions            => 5
		4 * 4 * sizeof(char)  + // 1: parameterisation + 2: smoothingFunctionHemispheres + 1: kernel => 4
		                        //    /!\ 3 CHAR are encoded using 4 BYTES                           => 4 * 4
		1 *     sizeof(float) + // 1: regularisationParameter                                        => 1
		2 *     sizeof(bool);   // 1: forceReciprocity + 1: forceBilateralSymmetry                   => 2
	int kernelNbParams = 0;

	std::fstream outputFile(outputFilePath, std::ios::in | std::ios::out | std::ios::binary);
	if (outputFile.is_open()) {
		outputFile.seekp(kernelNbParamsStreamPosition);

		// Reading the number of kernel parameters and moving after parameters
		outputFile.read(reinterpret_cast<char*>(&kernelNbParams), sizeof(int));
		std::streampos streamPosition = kernelNbParamsStreamPosition + static_cast<std::streamoff>(kernelNbParams * sizeof(float));

		streamPosition +=
			    sizeof(float) + // 1: threshold coefficient
			3 * sizeof(int)   + // 1: number of kernel parameters + 1: number of clusters + 1: number of RBF per cluster => 3
			    sizeof(bool);   // 1: unique location RBF

		if (runtimeConfig.m_uniqueLocationRBF)
			// In unique location RBF mode, RBF coordinates are directly stored right after the header
			streamPosition += runtimeConfig.m_nbDimensions * inputData.size() * sizeof(FloatingPrecision);
		else
			// In non-unique location RBF mode, we need to store the number of RBF for each cluster
			streamPosition += clusterID * runtimeConfig.m_nbDimensions * inputData.size() * sizeof(FloatingPrecision);

		// Adding to the stream position the location of the previous clusters
		if (typeid(DataType) == typeid(FloatingPrecision))
			// If DataType is FloatingPrecision it means that we are in scalar mode
			// meaning that we do not store RGB values, but only one value per RBF.
			streamPosition += clusterID * inputData.size() * sizeof(FloatingPrecision);
		else
			// If DataType is not FloatingPrecision, it means that we
			// are in RGB mode meaning that we store 3 values per RBF
			streamPosition += 3 * clusterID * inputData.size() * sizeof(FloatingPrecision);

		outputFile.seekp(streamPosition);

		// Writing the RBF location if needed
		if (!(runtimeConfig.m_uniqueLocationRBF)) {			
			for (size_t i = 0; i < runtimeConfig.m_coordinatesRBF.size(); i++) {
				const auto& coordinate = runtimeConfig.m_coordinatesRBF[i];
				if      (typeid(*coordinate) == typeid(Coordinate2D<FloatingPrecision>)) {
					const auto& coord = static_cast<Coordinate2D<FloatingPrecision>&>(*coordinate);
					FloatingPrecision theta = coord.getTheta();
					FloatingPrecision phi   = coord.getPhi();
					outputFile.write(reinterpret_cast<const char*>(&theta), sizeof(FloatingPrecision));
					outputFile.write(reinterpret_cast<const char*>(&phi)  , sizeof(FloatingPrecision));
				}
				else if (typeid(*coordinate) == typeid(Coordinate3DSpherical<FloatingPrecision>)) {
					const auto& coord = static_cast<Coordinate3DSpherical<FloatingPrecision>&>(*coordinate);
					FloatingPrecision thetaI   = coord.getThetaI();
					FloatingPrecision thetaO   = coord.getThetaO();
					FloatingPrecision deltaPhi = coord.getDeltaPhi();
					outputFile.write(reinterpret_cast<const char*>(&thetaI)  , sizeof(FloatingPrecision));
					outputFile.write(reinterpret_cast<const char*>(&thetaO)  , sizeof(FloatingPrecision));
					outputFile.write(reinterpret_cast<const char*>(&deltaPhi), sizeof(FloatingPrecision));
				}
				else if (typeid(*coordinate) == typeid(Coordinate3DRusinkiewicz<FloatingPrecision>)) {
					const auto& coord = static_cast<Coordinate3DRusinkiewicz<FloatingPrecision>&>(*coordinate);
					FloatingPrecision thetaH = coord.getThetaH();
					FloatingPrecision thetaD = coord.getThetaD();
					FloatingPrecision phiD   = coord.getPhiD();
					outputFile.write(reinterpret_cast<const char*>(&thetaH), sizeof(FloatingPrecision));
					outputFile.write(reinterpret_cast<const char*>(&thetaD), sizeof(FloatingPrecision));
					outputFile.write(reinterpret_cast<const char*>(&phiD)  , sizeof(FloatingPrecision));
				}
				else
					LOG_CRITICAL("Invalid coordinate type.");
			}
		}

		// Writing the RBF weights
		for (size_t i = 0; i < inputData.size(); i++) {
			if      (typeid(DataType) == typeid(FloatingPrecision)) {
				// We are in scalar mode, only one value is stored
				const FloatingPrecision& value = reinterpret_cast<const FloatingPrecision&>(inputData[i]);
				outputFile.write(reinterpret_cast<const char*>(&value), sizeof(FloatingPrecision));
			}
			else if (typeid(DataType) == typeid(glm::vec3)) {
				// We are in RGB mode, three values are stored
				const glm::vec3& value = reinterpret_cast<const glm::vec3&>(inputData[i]);
				outputFile.write(reinterpret_cast<const char*>(&value.r), sizeof(FloatingPrecision));
				outputFile.write(reinterpret_cast<const char*>(&value.g), sizeof(FloatingPrecision));
				outputFile.write(reinterpret_cast<const char*>(&value.b), sizeof(FloatingPrecision));
			}
			else if (typeid(DataType) == typeid(glm::dvec3)) {
				// We are in RGB mode, three values are stored
				const glm::dvec3& value = reinterpret_cast<const glm::dvec3&>(inputData[i]);
				outputFile.write(reinterpret_cast<const char*>(&value.r), sizeof(FloatingPrecision));
				outputFile.write(reinterpret_cast<const char*>(&value.g), sizeof(FloatingPrecision));
				outputFile.write(reinterpret_cast<const char*>(&value.b), sizeof(FloatingPrecision));
			}
			else
				LOG_CRITICAL("Invalid data type.");
		}

		outputFile.close();
	}
	else
		LOG_CRITICAL("Output file ", outputFilePath, " could not be opened.");
}

template <typename FloatingPrecision, typename DataType>
void writeAllToRBFCoeffs(
	const std::string& outputFilePath,
	
	const RuntimeConfig<FloatingPrecision>& runtimeConfig,
	const std::vector<DataType>& inputData,
	const std::vector<std::unique_ptr<Coordinate>>& inputCoordinates)
{
	// Security check if input sizes are equal
	if (inputData.size() != inputCoordinates.size())
		LOG_CRITICAL("Input data size and input coordinates size do not match."
		             "inputData: ", inputData.size(), " | vs inputCoordinates:"
			         " ", inputCoordinates.size());

	// Size of the header can vary depending on
	// the kernel and its number of parameters.
	std::streampos kernelNbParamsStreamPosition =
		5 * sizeof(int)      + // 3: v.MAJOR.MINOR.PATCH + 1: dataType + 1: nbDimensions            => 5
		4 * 4 * sizeof(char) + // 1: parameterisation + 2: smoothingFunctionHemispheres + 1: kernel => 4
		                       //    /!\ 3 CHAR are encoded using 4 BYTES                           => 4 * 4
		1 * sizeof(float)    + // 1: regularisationParameter                                        => 1
		2 * sizeof(bool);      // 1: forceReciprocity + 1: forceBilateralSymmetry                   => 2
	int kernelNbParams = 0;

	std::fstream outputFile(outputFilePath, std::ios::in | std::ios::out | std::ios::binary);
	if (outputFile.is_open()) {
		outputFile.seekp(kernelNbParamsStreamPosition);

		// Reading the number of kernel parameters and moving after parameters
		outputFile.read(reinterpret_cast<char*>(&kernelNbParams), sizeof(int));
		std::streampos streamPosition = kernelNbParamsStreamPosition + static_cast<std::streamoff>(kernelNbParams * sizeof(float));

		streamPosition +=
			    sizeof(float) + // 1: threshold coefficient
			3 * sizeof(int)   + // 1: number of kernel parameters + 1: number of clusters + 1: number of RBF per cluster => 3
			    sizeof(bool);   // 1: unique location RBF

		if (runtimeConfig.m_uniqueLocationRBF)
			// In unique location RBF mode, RBF coordinates are directly stored right after the header
			streamPosition += runtimeConfig.m_nbDimensions * inputData.size() * sizeof(FloatingPrecision);

		outputFile.seekp(streamPosition);

		// If we are in unique location RBF mode, the coordinates are already written in the header and we
		// only have to write the RBF weights. If we are in a non-unique location RBF mode, first, we will
		// need to write the coordinates, then the RBF weights, for each cluster.
		size_t numberOfLoops = runtimeConfig.m_uniqueLocationRBF ? inputData.size() : 2 * inputData.size();

		for (size_t i = 0; i < numberOfLoops; i++) {
			// The two following identifiers may look weird but are quite simple. The first one returns
			// either 0 or 1, depending on whether we need to read and write the RBF coordinates or its
			// coefficients.
			size_t isCoefLoop    = ((i - (i % runtimeConfig.m_numberRBF)) / runtimeConfig.m_numberRBF) % 2;
			// The coefficientID is the index of the coefficient in the RBF bidirection array or either
			// in the RBF weights array.
			size_t coefficientID = ((i - (i % (2 * runtimeConfig.m_numberRBF))) / 2) + (i % runtimeConfig.m_numberRBF);

			if (isCoefLoop > 1)                    LOG_CRITICAL("Invalid coefficient loop index: ", isCoefLoop);
			if (coefficientID >= inputData.size()) LOG_CRITICAL("Coefficient ID out of bounds: ", coefficientID, " >= ", inputData.size());

			// We are not in unique location RBF mode and we are iterating
			// over the coordinates, so we need to write the bidirections.
			if (!(runtimeConfig.m_uniqueLocationRBF) && (isCoefLoop == 0)) {
				const auto& coordinate = inputCoordinates[coefficientID];
				if      (typeid(*coordinate) == typeid(Coordinate2D<FloatingPrecision>)) {
					const auto& coord = static_cast<Coordinate2D<FloatingPrecision>&>(*coordinate);
					FloatingPrecision theta = coord.getTheta();
					FloatingPrecision phi   = coord.getPhi();
					outputFile.write(reinterpret_cast<const char*>(&theta), sizeof(FloatingPrecision));
					outputFile.write(reinterpret_cast<const char*>(&phi), sizeof(FloatingPrecision));
				}
				else if (typeid(*coordinate) == typeid(Coordinate3DSpherical<FloatingPrecision>)) {
					const auto& coord = static_cast<Coordinate3DSpherical<FloatingPrecision>&>(*coordinate);
					FloatingPrecision thetaI   = coord.getThetaI();
					FloatingPrecision thetaO   = coord.getThetaO();
					FloatingPrecision deltaPhi = coord.getDeltaPhi();
					outputFile.write(reinterpret_cast<const char*>(&thetaI), sizeof(FloatingPrecision));
					outputFile.write(reinterpret_cast<const char*>(&thetaO), sizeof(FloatingPrecision));
					outputFile.write(reinterpret_cast<const char*>(&deltaPhi), sizeof(FloatingPrecision));
				}
				else if (typeid(*coordinate) == typeid(Coordinate3DRusinkiewicz<FloatingPrecision>)) {
					const auto& coord = static_cast<Coordinate3DRusinkiewicz<FloatingPrecision>&>(*coordinate);
					FloatingPrecision thetaH = coord.getThetaH();
					FloatingPrecision thetaD = coord.getThetaD();
					FloatingPrecision phiD   = coord.getPhiD();
					outputFile.write(reinterpret_cast<const char*>(&thetaH), sizeof(FloatingPrecision));
					outputFile.write(reinterpret_cast<const char*>(&thetaD), sizeof(FloatingPrecision));
					outputFile.write(reinterpret_cast<const char*>(&phiD), sizeof(FloatingPrecision));
				}
				else
					LOG_CRITICAL("Invalid coordinate type for coefficientID: ", coefficientID);
			}

			// We are not in unique location RBF mode and we are iterating over
			// the coefficients, so we need to write the associated RBF weights
			else if (!(runtimeConfig.m_uniqueLocationRBF) && (isCoefLoop == 1)) {
				if      (typeid(DataType) == typeid(FloatingPrecision)) {
					// We are in scalar mode, only one value is stored
					const FloatingPrecision& value = reinterpret_cast<const FloatingPrecision&>(inputData[coefficientID]);
					outputFile.write(reinterpret_cast<const char*>(&value), sizeof(FloatingPrecision));
				}
				else if (typeid(DataType) == typeid(glm::vec3)) {
					// We are in RGB mode, three values are stored
					const glm::vec3& value = reinterpret_cast<const glm::vec3&>(inputData[coefficientID]);
					outputFile.write(reinterpret_cast<const char*>(&value.r), sizeof(FloatingPrecision));
					outputFile.write(reinterpret_cast<const char*>(&value.g), sizeof(FloatingPrecision));
					outputFile.write(reinterpret_cast<const char*>(&value.b), sizeof(FloatingPrecision));
				}
				else if (typeid(DataType) == typeid(glm::dvec3)) {
					// We are in RGB mode, three values are stored
					const glm::dvec3& value = reinterpret_cast<const glm::dvec3&>(inputData[coefficientID]);
					outputFile.write(reinterpret_cast<const char*>(&value.r), sizeof(FloatingPrecision));
					outputFile.write(reinterpret_cast<const char*>(&value.g), sizeof(FloatingPrecision));
					outputFile.write(reinterpret_cast<const char*>(&value.b), sizeof(FloatingPrecision));
				}
				else
					LOG_CRITICAL("Invalid data type.");
			}

			// We are in unique location RBF mode and 
			// we are iterating over the RBG weights.
			else {
				if      (typeid(DataType) == typeid(FloatingPrecision)) {
					// We are in scalar mode, only one value is stored
					const FloatingPrecision& value = reinterpret_cast<const FloatingPrecision&>(inputData[i]);
					outputFile.write(reinterpret_cast<const char*>(&value), sizeof(FloatingPrecision));
				}
				else if (typeid(DataType) == typeid(glm::vec3)) {
					// We are in RGB mode, three values are stored
					const glm::vec3& value = reinterpret_cast<const glm::vec3&>(inputData[i]);
					outputFile.write(reinterpret_cast<const char*>(&value.r), sizeof(FloatingPrecision));
					outputFile.write(reinterpret_cast<const char*>(&value.g), sizeof(FloatingPrecision));
					outputFile.write(reinterpret_cast<const char*>(&value.b), sizeof(FloatingPrecision));
				}
				else if (typeid(DataType) == typeid(glm::dvec3)) {
					// We are in RGB mode, three values are stored
					const glm::dvec3& value = reinterpret_cast<const glm::dvec3&>(inputData[i]);
					outputFile.write(reinterpret_cast<const char*>(&value.r), sizeof(FloatingPrecision));
					outputFile.write(reinterpret_cast<const char*>(&value.g), sizeof(FloatingPrecision));
					outputFile.write(reinterpret_cast<const char*>(&value.b), sizeof(FloatingPrecision));
				}
				else
					LOG_CRITICAL("Invalid data type.");
			}
		}

		outputFile.close();
	}
	else
		LOG_CRITICAL("Output file ", outputFilePath, " could not be opened.");
}

template <typename FloatingPrecision>
void exportToMERL(
	const std::string& outputFilePath, 
	              
	RBFInterpolator<FloatingPrecision>& interpolatorR, 
	RBFInterpolator<FloatingPrecision>& interpolatorG, 
	RBFInterpolator<FloatingPrecision>& interpolatorB,
	
	const bool& parallelComputing) 
{

	LOG_INFO("Output file path name is: ", outputFilePath);
	// Open the output file
	std::ofstream outputFile(outputFilePath, std::ios::out | std::ios::binary);
	if (outputFile.is_open())
		LOG_DEBUG("Output file ", outputFilePath, " opened successfully.");
	else
		LOG_CRITICAL("Output file ", outputFilePath, " could not be opened.");

	// Listing all the values needed for each coordinate
	// and taking into account that thetaH is not linear
	std::vector<FloatingPrecision> phiDArray   = linspace(static_cast<FloatingPrecision>(0), glm::pi<FloatingPrecision>()                , 180);
	std::vector<FloatingPrecision> thetaDArray = linspace(static_cast<FloatingPrecision>(0), glm::half_pi<FloatingPrecision>()           , 90);
	std::vector<FloatingPrecision> thetaHArray = linspace(static_cast<FloatingPrecision>(0), glm::sqrt(glm::half_pi<FloatingPrecision>()), 90);
	for (size_t i = 0; i < thetaHArray.size(); i++)
		thetaHArray[i] = thetaHArray[i] * thetaHArray[i];

	// Header mandatory to create a MERL-style binary file
	int header[] = { 90, 90, 180 };
	outputFile.write(reinterpret_cast<const char*>(header), sizeof(header));

	double thetaH, thetaD, phiD, interpolatedValue;

	// Compute the BRDF values for each RGB channel
	std::vector<double> datasetR(180 * 90 * 90);
	std::vector<double> datasetG(180 * 90 * 90);
	std::vector<double> datasetB(180 * 90 * 90);

	// Reducing interpolator for faster computation. Will
	// not affect the result if the threshold is set to 0
	if (&interpolatorR == &interpolatorB && &interpolatorG == &interpolatorB)
		interpolatorR.reduceInterpolator();
	else {
		interpolatorR.reduceInterpolator();
		interpolatorG.reduceInterpolator();
		interpolatorB.reduceInterpolator();
	}

	std::atomic<size_t> completedIterations{ 0 };
	#pragma omp parallel for private(thetaH, thetaD, phiD) if(parallelComputing)
	for (int i = 0; i < 180 * 90 * 90; i++) {

		size_t phiDID   =   i % 180;
		size_t thetaDID = ((i - phiDID) / 180) % 90;
		size_t thetaHID =  (i - phiDID - 180 * thetaDID) / (90 * 180);

		thetaH = thetaHArray[thetaHID];
		thetaD = thetaDArray[thetaDID];
		phiD   = phiDArray[phiDID];

		// Compute the result here
		Coordinate3DRusinkiewicz<FloatingPrecision> interpolatedBidirectionRusinkiewicz(thetaH, thetaD, phiD);
		Coordinate3DSpherical<FloatingPrecision> interpolatedBidirectionSpherical = static_cast<Coordinate3DSpherical<FloatingPrecision>>(interpolatedBidirectionRusinkiewicz);

		// In the case where we are doing panchromatic interpolation,
		// We can use the same interpolator for all the RGB channels.
		if (&interpolatorR == &interpolatorB && &interpolatorG == &interpolatorB) {
			datasetR[i] = interpolatorR.interpolate(std::make_unique<Coordinate3DSpherical<FloatingPrecision>>(interpolatedBidirectionSpherical));
			datasetG[i] = datasetB[i] = datasetR[i];
		}
		// If not, we need to compute the BRDF values for each channel
		else {
			datasetR[i] = interpolatorR.interpolate(std::make_unique<Coordinate3DSpherical<FloatingPrecision>>(interpolatedBidirectionSpherical));
			datasetG[i] = interpolatorG.interpolate(std::make_unique<Coordinate3DSpherical<FloatingPrecision>>(interpolatedBidirectionSpherical));
			datasetB[i] = interpolatorB.interpolate(std::make_unique<Coordinate3DSpherical<FloatingPrecision>>(interpolatedBidirectionSpherical));
		}

		// Tracking progress with parallel computing
		logger.displayProgressBar(completedIterations.load(std::memory_order_relaxed), 180 * 90 * 90);
		completedIterations.fetch_add(1, std::memory_order_relaxed);
	}

	// Pre-multiply datasets and convert to double for writing
	std::vector<double> adjustedR(datasetR.size()), adjustedG(datasetG.size()), adjustedB(datasetB.size());
	for (size_t i = 0; i < datasetR.size(); ++i) adjustedR[i] = static_cast<double>(datasetR[i]) * 1500.;
	for (size_t i = 0; i < datasetG.size(); ++i) adjustedG[i] = static_cast<double>(datasetG[i]) * (1500. / 1.15);
	for (size_t i = 0; i < datasetB.size(); ++i) adjustedB[i] = static_cast<double>(datasetB[i]) * (1500. / 1.66);

	// Write the adjusted BRDF values for each RGB channel
	outputFile.write(reinterpret_cast<const char*>(adjustedR.data()), sizeof(double) * adjustedR.size());
	outputFile.write(reinterpret_cast<const char*>(adjustedG.data()), sizeof(double) * adjustedG.size());
	outputFile.write(reinterpret_cast<const char*>(adjustedB.data()), sizeof(double) * adjustedB.size());

	outputFile.close();

	LOG_INFO("MERL binary file ", outputFilePath, " created.");
}