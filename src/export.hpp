#pragma once

#include <fstream>
#include <omp.h>
#include <string>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>

#include "arrays.hpp"
#include "configurator.hpp"
#include "interpolator.hpp"
#include "logger.hpp"

template <typename FloatingPrecision>
void exportToMERL(
	const std::string& outputFilePath, 
	              
	RBFInterpolator<FloatingPrecision>& interpolatorR, 
	RBFInterpolator<FloatingPrecision>& interpolatorG, 
	RBFInterpolator<FloatingPrecision>& interpolatorB,
	
	const bool& parallelComputing) 
{
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