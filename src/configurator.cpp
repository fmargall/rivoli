#include <fstream>

#include "configurator.hpp"
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
			bool breaker = false;

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
			else
				LOG_WARN("Invalid key value: ", key, " in config file ",
					     configFilePath, ". This line will be ignored");

			if (breaker) // Logging the errors and stopping the configuration
				LOG_CRITICAL("Invalid value for key ", key, " in configuration file: ", value);
			else // Logging the modifications
				LOG_DEBUG(key, " set to: ", value);

		}
	}
}

std::string MainConfig::getFloatingPointPrecision() const {
	return m_floatingPointPrecision;
}

template <typename FloatingPrecision>
RuntimeConfig<FloatingPrecision>::RuntimeConfig(const MainConfig& mainConfig) {
	LOG_DEBUG("Runtime configuration initialised.");
}