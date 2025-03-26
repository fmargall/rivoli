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
	// Opening the output .RBFCoeffs file
	std::ofstream outputDataFile(outputFilePath, std::ios::binary | std::ios::trunc);
	if (outputDataFile.is_open())
		LOG_DEBUG("Output file ", outputFilePath, " opened successfully.");
	else
		LOG_CRITICAL("Output file output.RBFCoeffs could not be opened.");

	// Writing the version of the software
	int versionMajor = 0, versionMinor = 1, versionPatch = 0;
	outputDataFile.write(reinterpret_cast<char*>(&versionMajor), sizeof(int));
	outputDataFile.write(reinterpret_cast<char*>(&versionMinor), sizeof(int));
	outputDataFile.write(reinterpret_cast<char*>(&versionPatch), sizeof(int));

	// Writing data type
	int dataType;
	if (isRGB) // 2 for float RGB, 3 for double RGB
		dataType = (runtimeConfig.m_floatingPointPrecision == "FP32") ? 2 : 3;
	else       // 0 for float, 1 for double
		dataType = (runtimeConfig.m_floatingPointPrecision == "FP32") ? 0 : 1;
	outputDataFile.write(reinterpret_cast<char*>(&dataType), sizeof(int));
	
	// Writing the number of dimensions
	int nbDimensionsInt = static_cast<int>(runtimeConfig.m_nbDimensions);
	outputDataFile.write(reinterpret_cast<char*>(&nbDimensionsInt), sizeof(int));

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
	const size_t& clusterID = -1)
{
	// Lock mutex to protect the file access for writing
	std::lock_guard<std::mutex> guard(writeFileMutex);

	// Computing stream position
	std::streampos streamPosition;

	std::ofstream outputFile(outputFilePath, std::ios::in | std::ios::out | std::ios::binary);
	if (outputFile.is_open()) {
		outputFile.seekp(streamPosition);
		// Writing the number of RBF only if needed
		if (!(runtimeConfig.m_uniqueLocationRBF)) {

		}
		// Writing RBF locations if needed
		// 
		// Writing RBF coefficients
		
		// 
		//outputFile.write(inputData.c_str(), inputData.size());
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