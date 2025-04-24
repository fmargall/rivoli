#include <mutex>

#include <glm/glm.hpp>

#include "export.hpp"

std::mutex writeFileMutex;

// Explicit instantiations
template void initRBFCoeffsFile<float> (const std::string& outputFilePath, const RuntimeConfig<float>&  runtimeConfig, const bool& isRGB);
template void initRBFCoeffsFile<double>(const std::string& outputFilePath, const RuntimeConfig<double>& runtimeConfig, const bool& isRGB);

template void writeToRBFCoeffs<float> (const std::string& outputFilePath, const RuntimeConfig<float>&  runtimeConfig, const std::vector<float>&     inputData, const size_t& clusterID);
template void writeToRBFCoeffs<float> (const std::string& outputFilePath, const RuntimeConfig<float>&  runtimeConfig, const std::vector<glm::vec3>& inputData, const size_t& clusterID);
template void writeToRBFCoeffs<double>(const std::string& outputFilePath, const RuntimeConfig<double>& runtimeConfig, const std::vector<double>&    inputData, const size_t& clusterID);
template void writeToRBFCoeffs<double>(const std::string& outputFilePath, const RuntimeConfig<double>& runtimeConfig, const std::vector<glm::vec3>& inputData, const size_t& clusterID);
