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

	std::string floatingPointPrecision = mainConfig.getFloatingPointPrecision();

	// Initialisation of the runtime configuration
	std::unique_ptr<MainConfig> runtimeConfig;

	// Selecting numerical precision (FP32 for single and FP64 for double precision)
	if      (floatingPointPrecision == "FP32")
		runtimeConfig = std::make_unique<RuntimeConfig<float>>(mainConfig);
	else if (floatingPointPrecision == "FP64")
		runtimeConfig = std::make_unique<RuntimeConfig<double>>(mainConfig);
	else
		LOG_CRITICAL("Unknown floating point precision: ", floatingPointPrecision, ". Should be 'FP32' or 'FP64' only.");

	LOG_TRACE("RIVOLI program will end.");
	return 0;
}