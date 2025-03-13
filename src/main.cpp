#include <string>

#include "configurator.hpp"
#include "logger.hpp"

int main(int argc, char* argv[]) {
	LOG_TRACE("RIVOLI program started.");

	// Read configuration file
	if (argc != 2)
		LOG_CRITICAL("Invalid number of arguments. Please provide"
			         " only the path to the configuration file.");

	// Initialize the main configuration
	std::string configFilePath = argv[1];
	MainConfig mainConfig(configFilePath);



	return 0;
}