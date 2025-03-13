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
}