#include <fstream>

#include "configurator.hpp"
#include "coordinate.hpp"
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
				if (value == "OFF"           || value == "0")
					logger.level = LogLevel::OFF;
				else if (value == "CRITICAL" || value == "1")
					logger.level = LogLevel::CRITICAL;
				else if (value == "ERR"      || value == "2")
					logger.level = LogLevel::ERR;
				else if (value == "WARN"     || value == "3")
					logger.level = LogLevel::WARN;
				else if (value == "INFO"     || value == "4")
					logger.level = LogLevel::INFO;
				else if (value == "DEBUG"    || value == "5")
					logger.level = LogLevel::DEBUG;
				else if (value == "TRACE"    || value == "6")
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
				if      (value == "true")
					m_forceBilateralSymmetry = true;
				else if (value == "false")
					m_forceBilateralSymmetry = false;
				else
					breaker = true;
			}
			else if (key == "forceReciprocity") {
				if      (value == "true")
					m_forceReciprocity = true;
				else if (value == "false")
					m_forceReciprocity = false;
				else
					breaker = true;
			}
			else if (key == "inputFilePath")
				m_inputFilePath = value;
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
RuntimeConfig<FloatingPrecision>::RuntimeConfig(const MainConfig& mainConfig) : MainConfig(mainConfig) {
	if      (typeid(FloatingPrecision) == typeid(float))
		LOG_TRACE("Using single precision floating point");
	else if (typeid(FloatingPrecision) == typeid(double))
		LOG_TRACE("Using double precision floating point");
	else
		LOG_CRITICAL("Unknown floating point precision");

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

		if      (nbDimensions == 2) {
			FloatingPrecision theta, phi;
			lineStream >> theta >> phi >> value;
			m_coordinates.push_back(std::make_unique<Coordinate2D<FloatingPrecision>>(theta, phi));
			LOG_TRACE("New 2D coordinate point added, with theta: ", theta, " and phi: ", phi);

		}
		else if (nbDimensions == 3) {
			FloatingPrecision thetaOne, thetaTwo, phiTwo;
			lineStream >> thetaOne >> thetaTwo >> phiTwo >> value;
			m_coordinates.push_back(std::make_unique<Coordinate3DSpherical<FloatingPrecision>>(thetaOne, thetaTwo, phiTwo));
			LOG_TRACE("New 3D coordinate point added, with thetaOne: ", thetaOne, ", thetaTwo: ", thetaTwo, " and phiTwo: ", phiTwo);

		}
		else if (nbDimensions == 4) {
			FloatingPrecision thetaOne, phiOne, thetaTwo, phiTwo;
			lineStream >> thetaOne >> phiOne >> thetaTwo >> phiTwo >> value;

		}
		else
			LOG_CRITICAL("RIVOLI supports only 2D, 3D, or 4D BRDF.");

		m_values.push_back(value);
	}

	LOG_INFO("Input file ", m_inputFilePath, " loaded. ",m_values.size(), " configurations saved.");

	
	// Initialising RBF coordinates. If no RBF location file is
	// provided, the RBF coordinates are the same as the input.
	if (m_locationRBFFilePath == "none") {
		// Making deep copy of the data
		m_coordinatesRBF.reserve(m_coordinates.size());
		for (const auto& coord : m_coordinates)
			m_coordinatesRBF.push_back(std::make_unique<Coordinate>(*coord));

		// Checking the number of coordinates
		if (m_coordinates.size() != m_coordinatesRBF.size())
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
				LOG_TRACE("New 2D coordinate point added, with theta: ", theta, " and phi: ", phi);

			}
			else if (nbDimensions == 3) {
				FloatingPrecision thetaOne, thetaTwo, phiTwo;
				lineStream >> thetaOne >> thetaTwo >> phiTwo >> value;
				m_coordinatesRBF.push_back(std::make_unique<Coordinate3DSpherical<FloatingPrecision>>(thetaOne, thetaTwo, phiTwo));
				LOG_TRACE("New 3D coordinate point added, with thetaOne: ", thetaOne, ", thetaTwo: ", thetaTwo, " and phiTwo: ", phiTwo);

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

	// Initialising the RBF interpolator
	RBFInterpolator<FloatingPrecision> interpolator(*this);

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
				LOG_DEBUG("BRDF topology initialised as 2D reciprocal symmetrical.");
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
		LOG_CRITICAL("RIVOLI currently supports only 2D or 3D BRDF.");

	return topology;
}

// Explicit instantiation
template class RuntimeConfig<float>;
template class RuntimeConfig<double>;