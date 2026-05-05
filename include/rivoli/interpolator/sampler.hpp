#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <random>
#include <string_view>
#include <vector>

#include <Eigen/Dense>

namespace rivoli {


enum class SamplerType {
    LipschitzRandomized
};

constexpr std::string_view toString(SamplerType samplerType) {
    switch (samplerType) {
        case SamplerType::LipschitzRandomized: return "LipschitzRandomized";
    }
    return "Unknown";
}


template <typename FP, size_t Dimension>
struct SampledData {
    std::array<std::vector<FP>, Dimension> coordinates;
    std::vector<FP> values;
};


template <typename FP, typename TopologyType>
std::vector<FP> _computeLocalLipschitzConstants(
    const std::array<std::vector<FP>, TopologyType::dimension>& inputCoordinates,
    const            std::vector<FP>&                           inputValues,
    const TopologyType& topology)
{
    size_t N = inputValues.size();
    std::vector<FP> localLipschitzConstants(N, static_cast<FP>(1.));

    if (N < 2)
        LOG_CRITICAL("At least two points are required to compute local Lipschitz constants.");

    const std::size_t k = std::min(size_t{5}, N - 1);
    if (k < 5) LOG_WARNING("Very small number of points to compute local Lipschitz constants"
                           ". Using k = ", k, ". Usually a minimum of k = 5 is recommended.");

    const FP epsilon = std::numeric_limits<FP>::epsilon() * FP(100);

    // Computing the distance matrix between all pairs of input coordinates
    Eigen::Matrix<FP, Eigen::Dynamic, Eigen::Dynamic> distanceMatrix(N, N);
    distanceMatrix.diagonal().setZero();

    for (size_t i = 0; i < N; i++) {
        // Since distance matrix is symmetrical
        // only upper triangular part is needed
        for (std::size_t j = i + 1; j < N; j++) {

            // Compute the distance between two points using the topology's distance function
            FP distance = [&]<std::size_t... I>(std::index_sequence<I...>) {
                return topology.getDistanceScalar(inputCoordinates[I][i]..., inputCoordinates[I][j]...);
            }(std::make_index_sequence<TopologyType::dimension>{});

            distanceMatrix(i, j) = distance;
            distanceMatrix(j, i) = distance; // Equal by symmetry
        }

        // Each distance associated to point i is computed
        std::vector<std::pair<FP, std::size_t>> distances;
        distances.reserve(N - 1);
        for (std::size_t j = 0; j < N; ++j) {
            if (j == i) continue; // Skip distance to itself
            distances.emplace_back(distanceMatrix(i, j), j);
        }

        // Keep k nearest neighbors
        if (k < distances.size()) {
            std::nth_element(
                distances.begin(), distances.begin() + k, distances.end(),
                [](const auto& a, const auto& b) { return a.first < b.first; }
            );
        }

        FP maxLipschitzConstant = static_cast<FP>(0.);
        for (std::size_t neighborID = 0; neighborID < k; ++neighborID) {
            const auto& [distance, j] = distances[neighborID];

            FP lipschitzConstant = std::abs(inputValues[i] - inputValues[j]) / (distance + epsilon);
            if (lipschitzConstant > maxLipschitzConstant)
                maxLipschitzConstant = lipschitzConstant;
        }

        localLipschitzConstants[i] = maxLipschitzConstant;
    }

    LOG_TRACE("Local Lipschitz constants computed for all points.");

    return localLipschitzConstants;
}

template <typename FP>
void _clipConstantsAtQuantile(std::vector<FP>& unsortedConstants, FP quantile = static_cast<FP>(0.99))
{
    if (quantile < static_cast<FP>(0.) || quantile > static_cast<FP>(1.))
        LOG_CRITICAL("Quantile must be between 0 and 1. Given: ", quantile);

    std::vector<FP> sortedConstants = unsortedConstants;
    std::sort(sortedConstants.begin(), sortedConstants.end());

    size_t quantileIndex = static_cast<size_t>(quantile * (sortedConstants.size() - 1));
    FP quantileValue = sortedConstants[quantileIndex];
    for (FP& constant : unsortedConstants)
        constant = std::min(constant, quantileValue);
}

/*
 * This method implements a weighted sampling without replacement algorithm described
 * in "Weighted random sampling with a reservoir" by Pavlos S. Efraimidis and Paul G.
 * Spirakis (DOI: 10.1109/ICDE.2006.1).
 */
template <typename FP>
std::vector<size_t> _weightedSampleWithoutReplacement(
    const std::vector<FP>& weights, size_t sampleSize,
    std::uint32_t seed = 0)
{
    const size_t N = weights.size();

    if (sampleSize > N)
        LOG_CRITICAL("Requested sample size (", sampleSize, ") exceeds population size (", N, ").");

    // Check special case
    if (sampleSize == N) {
        std::vector<size_t> allIndices(N);
        std::iota(allIndices.begin(), allIndices.end(), size_t(0));
        return allIndices;
    }

    std::mt19937 rng(seed);
    // We exclude here the possibility of zero weights, since we will work with log keys rightafter
    std::uniform_real_distribution<FP> uniform(std::numeric_limits<FP>::min(), static_cast<FP>(1.));

    using ReservoirEntry  = std::pair<FP, size_t>;
    auto reservoirCompare = [](const ReservoirEntry& a, const ReservoirEntry& b) { return a.first > b.first; };
    std::vector<ReservoirEntry> reservoir;
    reservoir.reserve(sampleSize);

    for (size_t i = 0; i < N; ++i) {
        FP weight = weights[i];

        if (weight < static_cast<FP>(0.))
            LOG_CRITICAL("Negative weight encountered at index ", i, ": ", weight);

        // Zero weights are simply ignored
        if (weight == static_cast<FP>(0.))
            continue;

        const FP logKey = std::log(uniform(rng)) / weight;
        if (reservoir.size() < sampleSize) {
            reservoir.emplace_back(logKey, i);
            std::push_heap(reservoir.begin(), reservoir.end(), reservoirCompare);
        }
        else if (logKey > reservoir.front().first) {
            // New candidate beats the current worst: evict the top, then insert
            std::pop_heap(reservoir.begin(), reservoir.end(), reservoirCompare);
            reservoir.back() = { logKey, i };
            std::push_heap(reservoir.begin(), reservoir.end(), reservoirCompare);
        }
    }

    if (reservoir.size() < sampleSize)
        LOG_CRITICAL("Could not draw ", sampleSize, " samples: only ",
             reservoir.size(), " items had strictly positive weight.");

    std::vector<size_t> indicesKept;
    indicesKept.reserve(sampleSize);
    for (const auto& [logKey, index] : reservoir)
        indicesKept.push_back(index);

    return indicesKept;
}

/*
 * This method is an implementation of a method proposed in: "Revisiting Chebyshev
 * Polynomial and Anisotropic RBF Models for Tabular Regression" by Luciano Gerber
 * and Huw Lloyd (DOI: 10.48550/arXiv.2602.22422). See Eq. 1 for more.
 */
template <typename FP, typename TopologyType>
SampledData<FP, TopologyType::dimension> _sampleDataLipschitzRandomized(
    const std::array<std::vector<FP>, TopologyType::dimension>& inputCoordinates,
    const            std::vector<FP>&                           inputValues,
    const TopologyType& topology,
    size_t numberOfSamples,

    std::uint32_t seed = 0)
{
    // Computing local Lipschitz constants for each input value and associated coordinate
    std::vector<FP> localLipschitzConstants = _computeLocalLipschitzConstants(inputCoordinates, inputValues, topology);

    // As in the ref. paper, extreme values are clipped for better robustness
    _clipConstantsAtQuantile(localLipschitzConstants, static_cast<FP>(0.99));

    // Data kept are sampled without replacement with selection probability proportional to the local Lipschitz constant
    std::vector<size_t> indicesKept = _weightedSampleWithoutReplacement(localLipschitzConstants, numberOfSamples, seed);

    SampledData<FP, TopologyType::dimension> sampledData;

    // Initialize the sampled data with the proper dimensions
    for (std::size_t d = 0; d < TopologyType::dimension; d++)
        sampledData.coordinates[d].reserve(indicesKept.size());
    sampledData.values.reserve(indicesKept.size());

    // Save sampled data to returned struct
    for (std::size_t index : indicesKept) {
        for (std::size_t d = 0; d < TopologyType::dimension; d++)
            sampledData.coordinates[d].push_back(inputCoordinates[d][index]);
        sampledData.values.push_back(inputValues[index]);
    }

    return sampledData;
}

template <typename FP, typename TopologyType>
SampledData<FP, TopologyType::dimension> sampleData(
    const std::array<std::vector<FP>, TopologyType::dimension>& inputCoordinates,
    const            std::vector<FP>&                           inputValues,
    const TopologyType& topology,
    size_t numberOfSamples,

    const SamplerType& samplerType = SamplerType::LipschitzRandomized,
    std::uint32_t seed = 0)
{
    switch (samplerType) {
        case SamplerType::LipschitzRandomized: return _sampleDataLipschitzRandomized(inputCoordinates, inputValues, topology, numberOfSamples, seed);
    }

    LOG_CRITICAL("Unsupported sampler type: ", toString(samplerType));
}

}
