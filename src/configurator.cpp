#include <filesystem>
#include <fstream>

#include "BrdfSamplesCoupole.hpp"
#include "configurator.hpp"
#include "coordinate.hpp"
#include "export.hpp"
#include "interpolator.hpp"
#include "logger.hpp"

MainConfig::MainConfig(const std::string& configFilePath) {
	// Open the configuration file
	std::ifstream configFile(configFilePath);
	if (configFile.is_open())
		LOG_DEBUG("Configuration file ", configFilePath, " opened.");
	else
		LOG_CRITICAL("Configuration file ", configFilePath, " could not be opened.");

	// Browsing file to read configuration
	std::string line; size_t lineID = 0;
	while (std::getline(configFile, line)) {
		lineID++; // Incrementing line number

		if (line.empty() || line[0] == '#' || line[0] == '[')
			continue; // This line contains no data to read.

		size_t pos = line.find('=');
		if (pos != std::string::npos) {
			// Reading key and value
			std::string key = line.substr(0, pos);
			std::string value = line.substr(pos + 1);
			// Removing potential leading and trailing spaces
			key.erase(key.find_last_not_of(" \t") + 1);
			value.erase(0, value.find_first_not_of(" \t"));
			value.erase(value.find_last_not_of(" \t") + 1);

			// Used to stop the loop if an invalid key is found
			bool warning = false, breaker = false;

			if (key == "logLevel") {
				if (value == "OFF" || value == "0")
					logger.level = LogLevel::OFF;
				else if (value == "CRITICAL" || value == "1")
					logger.level = LogLevel::CRITICAL;
				else if (value == "ERR" || value == "2")
					logger.level = LogLevel::ERR;
				else if (value == "WARN" || value == "3")
					logger.level = LogLevel::WARN;
				else if (value == "INFO" || value == "4")
					logger.level = LogLevel::INFO;
				else if (value == "DEBUG" || value == "5")
					logger.level = LogLevel::DEBUG;
				else if (value == "TRACE" || value == "6")
					logger.level = LogLevel::TRACE;
				else
					breaker = true;
			}
			else if (key == "floatingPointFormat") {
				if (value == "FP32" || value == "FP64")
					m_floatingPointPrecision = value;
				else
					breaker = true;
			}
			else if (key == "kernel") {
				if (value == "linear")
					m_kernel = value;
				else
					breaker = true;
			}
			else if (key == "forceBilateralSymmetry") {
				if (value == "true")
					m_forceBilateralSymmetry = true;
				else if (value == "false")
					m_forceBilateralSymmetry = false;
				else
					breaker = true;
			}
			else if (key == "forceReciprocity") {
				if (value == "true")
					m_forceReciprocity = true;
				else if (value == "false")
					m_forceReciprocity = false;
				else
					breaker = true;
			}
			else if (key == "parallelComputing") {
				if (value == "true")
					m_parallelComputing = true;
				else if (value == "false")
					m_parallelComputing = false;
				else
					breaker = true;
			}
			else if (key == "inputFilePath")
				m_inputFilePath = value;
			else if (key == "outputFilePath")
				m_outputFilePath = value;
			else if (key == "outputFormat") {
				if (value == "MERL")
					m_outputFormat = value;
				else
					breaker = true;
			}
			else if (key == "regularisationParameter")
				m_regularisationParameter = std::stof(value);
			else if (key == "nonNegativityCorrectionParameter")
				m_nonNegativityCorrectionParameter = std::stof(value);
			else if (key == "locationRBFFilePath")
				m_locationRBFFilePath = value;
			else {
				LOG_WARN("Invalid key value: ", key, " in config file ",
					      configFilePath, ". This line will be ignored");
				warning = true;
			}

			if   (breaker)  // Logging the errors and stopping the configuration
				LOG_CRITICAL("Invalid value for key ", key, " in configuration file: ", value);
			if (!(warning)) // Logging the modifications
				LOG_DEBUG(key, " set to: ", value);

		}

		else
			// No '=' found means the line is unreadable.
			LOG_WARN("Unable to read line number ", 
				     lineID, " with value: ", line);
	}

	LOG_TRACE("Configuration file ", configFilePath, " read. MainConfig initialisiation over.");
}

std::string MainConfig::getFloatingPointPrecision() const {
	return m_floatingPointPrecision;
}

template <typename FloatingPrecision>
RuntimeConfig<FloatingPrecision>::RuntimeConfig(const MainConfig& mainConfig) : MainConfig(mainConfig)
{
	if      (typeid(FloatingPrecision) == typeid(float))
		LOG_TRACE("Using single precision floating point");
	else if (typeid(FloatingPrecision) == typeid(double))
		LOG_TRACE("Using double precision floating point");
	else
		LOG_CRITICAL("Unknown floating point precision");

	std::filesystem::path inputFilePath(m_inputFilePath);
	
	// The file that is about to be read comes from the Coupole
	if (inputFilePath.extension() == ".brdfSamples") {
		BrdfSamplesCoupole brdfSamples;
		int nbClusters = brdfSamples.prepareSparseRead(inputFilePath.string());

		LOG_TRACE("File ", inputFilePath.string(), " loaded.");

		// Initialising the topology
		m_nbDimensions = 3;
		m_topology = initTopology(*this);

		// Initialising RBF coordinates
		std::ifstream fileRBFLocation(m_locationRBFFilePath);
		if (fileRBFLocation.is_open())
			LOG_DEBUG("RBF location file " + m_locationRBFFilePath + " opened successfully.");
		else
			LOG_CRITICAL("RBF location file " + m_locationRBFFilePath + " could not be opened.");

		std::string line;
		while (std::getline(fileRBFLocation, line)) {
			if (line.empty())
				continue; // Skip empty lines

			std::istringstream lineStream(line);

			FloatingPrecision thetaOne, thetaTwo, phiTwo;
			lineStream >> thetaOne >> thetaTwo >> phiTwo;
			m_coordinatesRBF.push_back(std::make_unique<Coordinate3DSpherical<FloatingPrecision>>(thetaOne, thetaTwo, phiTwo));
		}

		LOG_DEBUG("RBF file ", m_locationRBFFilePath, " is loaded. ",
				   m_coordinatesRBF.size(), " configurations saved.");

		// Using the chosen topology, we can clean the RBF
		// coordinates and suppress the useless duplicates
		for (auto iteratorOne = m_coordinatesRBF.begin(); iteratorOne != m_coordinatesRBF.end(); ++iteratorOne) {
			for (auto iteratorTwo = iteratorOne + 1; iteratorTwo != m_coordinatesRBF.end(); ) {
				if (m_topology->getDistance(**iteratorOne, **iteratorTwo) == static_cast<FloatingPrecision>(0))
					iteratorTwo = m_coordinatesRBF.erase(iteratorTwo);
				else
					++iteratorTwo;
			}
		}
		LOG_DEBUG("RBF configurations cleaned. ", m_coordinatesRBF.size(), " configurations kept.");

		// Opening the output .RBFCoeffs file
		std::ofstream outputDataFile("output.RBFCoeffs", std::ios::binary | std::ios::trunc);
		if (outputDataFile.is_open())
			LOG_DEBUG("Output file output.RBFCoeffs opened successfully.");
		else
			LOG_CRITICAL("Output file output.RBFCoeffs could not be opened.");

		// Writing the header of the file
		int versionMajor = 0, versionMinor = 1, versionPatch = 0;
		outputDataFile.write(reinterpret_cast<char*>(&versionMajor), sizeof(int));
		outputDataFile.write(reinterpret_cast<char*>(&versionMinor), sizeof(int));
		outputDataFile.write(reinterpret_cast<char*>(&versionPatch), sizeof(int));

		int dataType = (m_floatingPointPrecision == "FP32") ? 2 : 3;
		outputDataFile.write(reinterpret_cast<char*>(&dataType), sizeof(int));
		int nbDimensionsInt = static_cast<int>(3);
		outputDataFile.write(reinterpret_cast<char*>(&nbDimensionsInt), sizeof(int));

		std::string parameterisation = (m_inputParameterisation == "spherical") ? "SPH" : "RUS";
		outputDataFile.write(parameterisation.c_str(), parameterisation.size());

		std::string smoothingFunctionHemisphereOne;
		if (m_forceGrazingAnglesNullFunctionHemisphereOne == "linear")
			smoothingFunctionHemisphereOne = "LIN";
		else if (m_forceGrazingAnglesNullFunctionHemisphereOne == "cosine")
			smoothingFunctionHemisphereOne = "COS";
		else if (m_forceGrazingAnglesNullFunctionHemisphereOne == "none")
			smoothingFunctionHemisphereOne = "NUL";
		else
			LOG_CRITICAL("Invalid smoothing function for hemisphere one: " + m_forceGrazingAnglesNullFunctionHemisphereOne);
		outputDataFile.write(smoothingFunctionHemisphereOne.c_str(), smoothingFunctionHemisphereOne.size());

		std::string smoothingFunctionHemisphereTwo;
		if (m_forceGrazingAnglesNullFunctionHemisphereTwo == "linear")
			smoothingFunctionHemisphereTwo = "LIN";
		else if (m_forceGrazingAnglesNullFunctionHemisphereTwo == "cosine")
			smoothingFunctionHemisphereTwo = "COS";
		else if (m_forceGrazingAnglesNullFunctionHemisphereTwo == "none")
			smoothingFunctionHemisphereTwo = "NUL";
		else
			LOG_CRITICAL("Invalid smoothing function for hemisphere two: " + m_forceGrazingAnglesNullFunctionHemisphereTwo);
		outputDataFile.write(smoothingFunctionHemisphereTwo.c_str(), smoothingFunctionHemisphereTwo.size());

		outputDataFile.write(reinterpret_cast<char*>(&m_regularisationParameter), sizeof(float));
		int nonNegativityCorrectionParameterInt = static_cast<int>(m_nonNegativityCorrectionParameter);
		outputDataFile.write(reinterpret_cast<char*>(&nonNegativityCorrectionParameterInt), sizeof(int));

		outputDataFile.write(reinterpret_cast<char*>(&m_forceReciprocity), sizeof(bool));
		outputDataFile.write(reinterpret_cast<char*>(&m_forceBilateralSymmetry), sizeof(bool));

		// Writing the RBF kernel
		std::string kernel;
		int kernelNbParams;
		if (m_kernel == "linear") {
			kernel = "LIN";
			kernelNbParams = 0;
		}
		outputDataFile.write(kernel.c_str(), kernel.size());
		outputDataFile.write(reinterpret_cast<char*>(&kernelNbParams), sizeof(int));

		// Writing the threshold coefficient
		outputDataFile.write(reinterpret_cast<char*>(&m_thresholdCoef), sizeof(float));
		// Writing the number of clusters
		outputDataFile.write(reinterpret_cast<char*>(&nbClusters), sizeof(int));
		// Writing the number of RBF
		int nbRBF = m_coordinatesRBF.size();
		outputDataFile.write(reinterpret_cast<char*>(&nbRBF), sizeof(int));

		LOG_INFO("Computation and writing of all interpolators started.");
		for (size_t clusterID = 0; clusterID < nbClusters; clusterID++) {
			// Prepare the cluster for reading only
			brdfSamples.readOneCluster(clusterID, 0);

			// Computing interpolator for red channel
			m_coordinates.clear(); m_values.clear();
			std::vector<glm::vec2> woVector;
			std::vector<glm::vec2> wiVector;
			std::vector<double> brdfVector;
			brdfSamples.getData(0, 0, woVector, wiVector, brdfVector);
			for (size_t sampleID = 0; sampleID < woVector.size(); sampleID++) {
				m_coordinates.push_back(std::make_unique<Coordinate3DSpherical<FloatingPrecision>>(static_cast<FloatingPrecision>(wiVector[sampleID].x),
																								   static_cast<FloatingPrecision>(woVector[sampleID].x),
																								   static_cast<FloatingPrecision>(wiVector[sampleID].y)));
				m_values.push_back(static_cast<FloatingPrecision>(brdfVector[sampleID]));
			}
			RBFInterpolator<FloatingPrecision> interpolatorR(*this);


			// Computing interpolator for green channel
			m_coordinates.clear(); m_values.clear();
			woVector.clear(); wiVector.clear(); brdfVector.clear();
			brdfSamples.getData(0, 1, woVector, wiVector, brdfVector);
			for (size_t sampleID = 0; sampleID < woVector.size(); sampleID++) {
				m_coordinates.push_back(std::make_unique<Coordinate3DSpherical<FloatingPrecision>>(static_cast<FloatingPrecision>(wiVector[sampleID].x),
																								   static_cast<FloatingPrecision>(woVector[sampleID].x),
																								   static_cast<FloatingPrecision>(wiVector[sampleID].y)));
				m_values.push_back(static_cast<FloatingPrecision>(brdfVector[sampleID]));
			}
			RBFInterpolator<FloatingPrecision> interpolatorG(*this);

			// Computing interpolator for blue channel
			m_coordinates.clear(); m_values.clear();
			woVector.clear(); wiVector.clear(); brdfVector.clear();
			brdfSamples.getData(0, 2, woVector, wiVector, brdfVector);
			for (size_t sampleID = 0; sampleID < woVector.size(); sampleID++) {
				m_coordinates.push_back(std::make_unique<Coordinate3DSpherical<FloatingPrecision>>(static_cast<FloatingPrecision>(wiVector[sampleID].x),
																								   static_cast<FloatingPrecision>(woVector[sampleID].x),
																								   static_cast<FloatingPrecision>(wiVector[sampleID].y)));
				m_values.push_back(static_cast<FloatingPrecision>(brdfVector[sampleID]));
			}
			RBFInterpolator<FloatingPrecision> interpolatorB(*this);

			// Writing the cluster in the file
			for (size_t coefID = 0; coefID < interpolatorR.getCoefficients().size(); coefID++) {
				FloatingPrecision coefR = interpolatorR.getCoefficients()[coefID];
				FloatingPrecision coefG = interpolatorG.getCoefficients()[coefID];
				FloatingPrecision coefB = interpolatorB.getCoefficients()[coefID];

				// Writing the coefficients
				glm::vec3 coefficients(coefR, coefG, coefB);
				outputDataFile.write(reinterpret_cast<char*>(&coefficients), sizeof(glm::vec3));

				if (outputDataFile.fail())
					LOG_CRITICAL("Error while writing the coefficients number ", coefID, " in the output file.");
			}

			//LOG_INFO("Cluster ", clusterID + 1, "/", nbClusters, " interpolated and written.");
			//logger.displayProgressBar(clusterID, nbClusters);

			if (clusterID == 0) {
				exportToMERL("lastCluster.binary", interpolatorR, interpolatorG, interpolatorB, true);
			}
		}
	}
	
	else {
		// Open the input file
		std::ifstream inputFile(m_inputFilePath);
		if (inputFile.is_open())
			LOG_DEBUG("Input file " + m_inputFilePath + " opened successfully.");
		else
			LOG_CRITICAL("Input file " + m_inputFilePath + " could not be opened.");

		// Reading the number of dimensions in the input datafile
		std::string line; std::getline(inputFile, line);
		std::istringstream s(line); FloatingPrecision columnValue;
		size_t nbDimensions = 0;  while (s >> columnValue) nbDimensions++;
		// One of the columns (thet last one) is for the value of the BRDF
		LOG_DEBUG(--nbDimensions, " dimensions detected in the input file.");
		m_nbDimensions = nbDimensions;

		// Reset cursor to the beginning
		inputFile.clear(); inputFile.seekg(0);

		// Parsing the input file
		while (std::getline(inputFile, line)) {
			if (line.empty())
				continue; // Skip empty lines

			std::istringstream lineStream(line);
			FloatingPrecision value;

			if (nbDimensions == 2) {
				FloatingPrecision theta, phi;
				lineStream >> theta >> phi >> value;
				m_coordinates.push_back(std::make_unique<Coordinate2D<FloatingPrecision>>(theta, phi));

			}
			else if (nbDimensions == 3) {
				FloatingPrecision thetaOne, thetaTwo, phiTwo;
				lineStream >> thetaOne >> thetaTwo >> phiTwo >> value;
				m_coordinates.push_back(std::make_unique<Coordinate3DSpherical<FloatingPrecision>>(thetaOne, thetaTwo, phiTwo));

			}
			else if (nbDimensions == 4) {
				FloatingPrecision thetaOne, phiOne, thetaTwo, phiTwo;
				lineStream >> thetaOne >> phiOne >> thetaTwo >> phiTwo >> value;

			}
			else
				LOG_CRITICAL("RIVOLI supports only 2D, 3D, or 4D BRDF.");

			// For testing purposes
			m_values.push_back(value);
		}

		LOG_INFO("Input file ", m_inputFilePath, " loaded. ", m_values.size(), " configurations saved.");


		// Initialising RBF coordinates. If no RBF location file is
		// provided, the RBF coordinates are the same as the input.
		if (m_locationRBFFilePath == "none") {
			// Making deep copy of the data
			m_coordinatesRBF.reserve(m_coordinates.size());
			for (const auto& coord : m_coordinates)
				m_coordinatesRBF.push_back(std::make_unique<Coordinate>(*coord));

			// Checking the number of coordinates
			if ((m_coordinates.size() != m_coordinatesRBF.size()) &&
				(typeid(*m_coordinates[0]) != typeid(*m_coordinatesRBF[0])))
				LOG_CRITICAL("RBF coordinates initialisation from input coordinates failed.");

		}
		else {
			std::ifstream fileRBFLocation(m_locationRBFFilePath);
			if (fileRBFLocation.is_open())
				LOG_DEBUG("RBF location file " + m_locationRBFFilePath + " opened successfully.");
			else
				LOG_CRITICAL("RBF location file " + m_locationRBFFilePath + " could not be opened.");

			std::string line;
			while (std::getline(fileRBFLocation, line)) {
				if (line.empty())
					continue; // Skip empty lines

				std::istringstream lineStream(line);
				FloatingPrecision value;

				if (nbDimensions == 2) {
					FloatingPrecision theta, phi;
					lineStream >> theta >> phi >> value;
					m_coordinatesRBF.push_back(std::make_unique<Coordinate2D<FloatingPrecision>>(theta, phi));

				}
				else if (nbDimensions == 3) {
					FloatingPrecision thetaOne, thetaTwo, phiTwo;
					lineStream >> thetaOne >> thetaTwo >> phiTwo >> value;
					m_coordinatesRBF.push_back(std::make_unique<Coordinate3DSpherical<FloatingPrecision>>(thetaOne, thetaTwo, phiTwo));

				}
				else if (nbDimensions == 4) {
					FloatingPrecision thetaOne, phiOne, thetaTwo, phiTwo;
					lineStream >> thetaOne >> phiOne >> thetaTwo >> phiTwo >> value;

				}
				else
					LOG_CRITICAL("RIVOLI supports only 2D, 3D, or 4D BRDF.");
			}

			LOG_DEBUG("RBF file ", m_locationRBFFilePath, " is loaded. ",
				m_coordinatesRBF.size(), " configurations saved.");
		}

		// Initialising the topology
		m_topology = initTopology(*this);

		// Using the chosen topology, we can clean the RBF
		// coordinates and suppress the useless duplicates
		for (auto iteratorOne = m_coordinatesRBF.begin(); iteratorOne != m_coordinatesRBF.end(); ++iteratorOne) {
			for (auto iteratorTwo = iteratorOne + 1; iteratorTwo != m_coordinatesRBF.end(); ) {
				if (m_topology->getDistance(**iteratorOne, **iteratorTwo) == static_cast<FloatingPrecision>(0))
					iteratorTwo = m_coordinatesRBF.erase(iteratorTwo);
				else
					++iteratorTwo;
			}
		}
		LOG_DEBUG("RBF configurations cleaned. ", m_coordinatesRBF.size(), " configurations kept.");

		// Initialising the RBF interpolator
		RBFInterpolator<FloatingPrecision> interpolator(*this);

		// Export BRDF
		if (m_outputFormat == "MERL")
			exportToMERL(m_outputFilePath, interpolator, interpolator, interpolator, m_parallelComputing);
	}

	
}

template <typename FloatingPrecision>
std::unique_ptr<Topology<FloatingPrecision>> RuntimeConfig<FloatingPrecision>::initTopology(RuntimeConfig<FloatingPrecision>& runtimeConfig) {
	// Initialising the topology pointer
	std::unique_ptr<Topology<FloatingPrecision>> topology;

	if      (runtimeConfig.m_nbDimensions == 2) {
		if (runtimeConfig.m_forceBilateralSymmetry) {
			topology = std::make_unique<Topology2DSym<FloatingPrecision>>();
			LOG_DEBUG("BRDF topology initialised as 2D symmetrical.");
		}
		else {
			topology = std::make_unique<Topology2D<FloatingPrecision>>();
			LOG_DEBUG("BRDF topology initialised as 2D.");
		}
	}
	else if (runtimeConfig.m_nbDimensions == 3) {
		if (runtimeConfig.m_forceBilateralSymmetry) {
			if (runtimeConfig.m_forceReciprocity) {
				topology = std::make_unique<Topology3DSphRecSym<FloatingPrecision>>();
				LOG_DEBUG("BRDF topology initialised as 3D reciprocal symmetrical.");
			}
			else {
				topology = std::make_unique<Topology3DSphSym<FloatingPrecision>>();
				LOG_DEBUG("BRDF topology initialised as 3D symmetrical.");
			}
		}
		else {
			if (runtimeConfig.m_forceReciprocity) {
				topology = std::make_unique<Topology3DSphRec<FloatingPrecision>>();
				LOG_DEBUG("BRDF topology initialised as 3D reciprocal.");
			}
			else {
				topology = std::make_unique<Topology3DSph<FloatingPrecision>>();
				LOG_DEBUG("BRDF topology initialised as 3D.");
			}
		}
	}
	else
		LOG_CRITICAL("RIVOLI currently supports only 2D or 3D BRDF. Given "
			         "number of dimensions: ", runtimeConfig.m_nbDimensions);

	return topology;
}

// Explicit instantiation
template class RuntimeConfig<float>;
template class RuntimeConfig<double>;