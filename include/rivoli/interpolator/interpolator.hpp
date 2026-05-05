#pragma once

#include <array>
#include <vector>

#include <Eigen/Dense>

#include <vectra/vectra.hpp>
#include <tinylogger/tinylogger.hpp>

#include <rivoli/kernels/kernels.hpp>
#include <rivoli/topologies/topologies.hpp>

#include "sampler.hpp"

namespace rivoli {


template <typename KernelType, typename TopologyType>
class Interpolator {

    static constexpr vectra::SIMDLevel level = TopologyType::level;

    using FP  = typename TopologyType::FP;
    using vct = vectra::Vectratype<FP, level>;

    //static constexpr std::size_t alignment = vct::alignment();

    using vctVector = std::vector<vct, vectra::aligned_allocator<vct, alignof(vct)>>;

private:
    // Forward declaration of the variable member _dimension used
    // directly right after in the public section and constructor

    // Natural integer that defines the dimension of the topology
    // Is directly defined from the topology itself. Should be 2,
    // 3 or 4.
    static constexpr size_t _dimension = TopologyType::dimension;

public:

    Interpolator(
        const std::array<std::vector<FP>, _dimension>& coordinates,
        const		     std::vector<FP>               coefficients,

        const KernelType&   kernel,
        const TopologyType& topology
    )
        : _kernel(kernel), _topology(topology)
    {
        _setCoordinates (coordinates);
        _setCoefficients(coefficients);
    }

    Interpolator(
        const std::array<std::vector<FP>, _dimension>& inputCoordinates,
        const            std::vector<FP>&              inputValues,

        const KernelType& kernel,
        const TopologyType& topology,

        // Optional additional parameters
        const FP     tikhonovRegularizationFactor,
        const bool   nonNegativity,
        const size_t sampledDataSize = 0 // Other value than 0 activates sampling
    )
        : _kernel(kernel), _topology(topology), _forceNonNegativity(nonNegativity)
    {
        for (size_t dimOne = 0; dimOne < _dimension; dimOne++) {
            if (inputCoordinates[dimOne].size() != inputValues.size())
                LOG_CRITICAL("Array sizes do not match. Input coordinates: ", inputCoordinates[dimOne].size(),
                             " | input values: ", inputValues.size());

            for (size_t dimTwo = dimOne + 1; dimTwo < _dimension; dimTwo++) {
                if (inputCoordinates[dimOne].size() != inputCoordinates[dimTwo].size())
                    LOG_CRITICAL("Input coordinates sizes do not match. Dimension ", dimOne, ": ",
                                 inputCoordinates[dimOne].size(), "  |  Dimension ", dimTwo, ": ",
                                 inputCoordinates[dimTwo].size());
            }
        }

        // Clean input data for better stability of the system, by enforcing non-negativity
        // and by removing any duplicates coordinates, by keeping for them their mean value
        auto [preprocessedCoordinates, preprocessedValues] = _preprocessInputData(inputCoordinates, inputValues);

        // Once the input data have been preprocessed, it may be required to sample them,
        // in order to reduce the number of points and thus the size of the kernel matrix
        if (sampledDataSize != 0) {
            SampledData<FP, _dimension> sampledData = sampleData<FP, TopologyType>(
                preprocessedCoordinates, preprocessedValues, topology, sampledDataSize);

            preprocessedCoordinates = sampledData.coordinates;
            preprocessedValues = sampledData.values;
        }

        // Eigen check for current SIMD instruction set version used, for debugging
        LOG_DEBUG("Current Eigen SIMD support: ", Eigen::SimdInstructionSetsInUse());

        // RBF interpolator is made by computing its coefficients, starting by
        // the computation of its kernel-distance matrix, using Eigen library.
        LOG_TRACE("Computing kernel distance matrix...");
        Eigen::Matrix<FP, Eigen::Dynamic, Eigen::Dynamic> kernelDistanceMatrix
            = _computeKernelDistanceMatrix(preprocessedCoordinates);
        LOG_TRACE("Kernel distance matrix computed.");

        // Tikhonov regularization is added to the diagonal of the kernel distance matrix
        if (tikhonovRegularizationFactor > static_cast<FP>(0.)) {
            kernelDistanceMatrix.diagonal().array() += tikhonovRegularizationFactor;
        }

        // Initialisation of result vector containing BRDF values
        Eigen::Vector<FP, Eigen::Dynamic> resultVector =
            Eigen::Map<const Eigen::Vector<FP, Eigen::Dynamic>>(
                preprocessedValues.data(), preprocessedValues.size());

        // Enforcing non-negativity on the result vector if required
        if (nonNegativity) resultVector = resultVector.array().sqrt();

        // Coefficients are computed using Eigen library
        Eigen::Vector<FP, Eigen::Dynamic> coefficients;

        // The system can be solved directly if we have a square matrix
        LOG_TRACE("Computing the coefficients...");
        if (kernelDistanceMatrix.rows() == kernelDistanceMatrix.cols()) {
            LOG_TRACE("Performing LDLT decomposition...");
            // LDLT decomposition is used for better performance, but less stable than LU decomposition
            Eigen::LDLT<Eigen::Matrix<FP, Eigen::Dynamic, Eigen::Dynamic>> ldlt(kernelDistanceMatrix);
            LOG_TRACE("LDLT decomposition achieved.");

            if (ldlt.info() != Eigen::Success)
                LOG_CRITICAL("LDLT decomposition failed. The kernel distance matrix might not be positive definite. "
                             "Consider using a more stable decomposition method, such as LU decomposition, or adding"
                             " a stronger Tikhonov regularization.");

            // Estimating LDLT condition number
            const auto& vectorD = ldlt.vectorD();
            FP conditionNumber = vectorD.cwiseAbs().maxCoeff() / vectorD.cwiseAbs().minCoeff();
            if (conditionNumber > static_cast<FP>(1e8))
                LOG_WARNING("Matrix likely ill-conditioned. Estimated condition number: ", conditionNumber);
            else
                LOG_DEBUG("Condition number estimated from LDLT diagonal decomposition: ", conditionNumber);

            coefficients = ldlt.solve(resultVector);

            // Checking residuals for debugging
            Eigen::Vector<FP, Eigen::Dynamic> residuals = (kernelDistanceMatrix * coefficients - resultVector);
            FP relativeError = residuals.norm() / resultVector.norm();
            LOG_DEBUG("Relative error of residuals: ", relativeError);

        }
        else
            LOG_CRITICAL("For now, only square kernel distance matrix is supported. Kernel distance matrix"
                         " rows: ", kernelDistanceMatrix.rows(), " | cols: ", kernelDistanceMatrix.cols());
        LOG_TRACE(coefficients.size(), " coefficients computed.");

        FP epsilon = std::numeric_limits<FP>::epsilon();
        Eigen::Index numZeroCoefs = (coefficients.array().abs() < epsilon).count();
        if (numZeroCoefs > 0)
            LOG_WARNING(numZeroCoefs, " coefficients with value equal to 0 detected.");

        _setCoordinates(preprocessedCoordinates);
        _setCoefficients(coefficients);

        LOG_INFO(topology.name, " interpolator initialised successfully.");
    }

    template <typename... CoordinatesType>
    FP interpolate (const CoordinatesType&... coordinates) const noexcept {

        // Convert scalar inputs into chosen SIMD backend type through vct
        std::array<vct, _dimension> coordinatesSIMD{ vct(coordinates)... };
        vct results = vct::zero();

        for (size_t i = 0; i < _coefficients.size(); i++) {
            if constexpr (KernelType::isAnisotropic) {
                auto distances = [&]<std::size_t... I>(std::index_sequence<I...>) {
                    return _topology.getDistances(_coordinates[I][i]..., coordinatesSIMD[I]...);
                }(std::make_index_sequence<_dimension>{});

                results = results + _coefficients[i] * _kernel(distances);
            }
            else {
                vct distance = [&]<std::size_t... I>(std::index_sequence<I...>) {
                    return _topology.getDistance(_coordinates[I][i]..., coordinatesSIMD[I]...);
                }(std::make_index_sequence<_dimension>{});

                results = results + _coefficients[i] * _kernel(distance);
            }
        }

        FP result = results.hsum();
        if (_forceNonNegativity)
            return result * result;
        else
            return result;
    }

private:

    void _setCoordinates(const std::array<std::vector<FP>, _dimension>& coordinates) noexcept {

        // Coordinates will be stored directly in the SIMD backend type.
        // Since the backend works by packets on a certain size, one may
        // have a rest, that will be added as an additional packet, then
        // filled with zeros.
        size_t remainingCoord =   coordinates[0].size() % vct::width();
        size_t sizeOfSIMDData = ((coordinates[0].size() - remainingCoord) / vct::width())
            + (remainingCoord != 0 ? 1 : 0);

        // Resizing _coordinates length for SIMD
        for (auto& coordinatesDim : _coordinates)
            coordinatesDim.resize(sizeOfSIMDData);

        for (size_t scalarID = 0, simdID = 0; scalarID + vct::width() < coordinates[0].size(); simdID++, scalarID += vct::width()) {
            // Conversion to SIMD backend for each dimension
            [&] <std::size_t... I>(std::index_sequence<I...>) {
                ((_coordinates[I][simdID] = vct::loadu(coordinates[I].data() + scalarID)), ...);
            }(std::make_index_sequence<_dimension>{});
        }

        // Handling potential tail
        if (remainingCoord != 0) {
            // Initialisation of last tail block to zero for
            // every dimension of the remaining coordinates:
            std::array<vct, _dimension> remainingCoords;
            for (auto& remainingCoordsDim : remainingCoords) remainingCoordsDim = vct::zero();

            // Filling the remaining useful part of the tail
            // block, for every dimension of the coordinates
            [&] <std::size_t... I>(std::index_sequence<I...>) {
                (([&] {
                    alignas(vct::alignment()) FP tmp[vct::width()] = {};
                    for (size_t j = 0; j < remainingCoord; ++j)
                        tmp[j] = coordinates[I][(sizeOfSIMDData - 1) * vct::width() + j];

                    remainingCoords[I] = vct::loadu(tmp);
                    }()), ...);
            }(std::make_index_sequence<_dimension>{});

            // Storing the final backend block for coordinates
            [&] <std::size_t... I>(std::index_sequence<I...>) {
                ((_coordinates[I][sizeOfSIMDData - 1] = remainingCoords[I]), ...);
            }(std::make_index_sequence<_dimension>{});
        }
    }

    template <typename CoefsVectorType>
    void _setCoefficients(const CoefsVectorType& coefficients) noexcept {

        // Coefficients will be stored directly in the SIMD backend type
        // Since the backend works by packets on a certain size, one may
        // have a rest, that will be added as an additional packet, then
        // filled with zeros.
        size_t remainingCoefs =   coefficients.size() % vct::width();
        size_t sizeOfSIMDData = ((coefficients.size() - remainingCoefs) / vct::width())
            + (remainingCoefs != 0 ? 1 : 0);

        _coefficients.resize(sizeOfSIMDData);

        for (size_t scalarID = 0, simdID = 0; scalarID + vct::width() < coefficients.size(); simdID++, scalarID += vct::width())
            _coefficients[simdID] = vct::loadu(coefficients.data() + scalarID);

        // Handling potential tail
        if (remainingCoefs != 0) {
            alignas(vct::alignment()) FP tmp[vct::width()] = {};

            // Filling the useful part of the tail block for the RBF coefficients
            for (size_t j = 0; j < remainingCoefs; ++j)
                tmp[j] = coefficients[(sizeOfSIMDData - 1) * vct::width() + j];

            // Storing final backend block for the coefficients
            _coefficients[sizeOfSIMDData - 1] = vct::loadu(tmp);
        }
    }

    std::pair<std::array<std::vector<FP>, _dimension>, std::vector<FP>> _preprocessInputData(
        const std::array<std::vector<FP>, _dimension>& inputCoordinates, const std::vector<FP>& inputValues) const
    {
        LOG_TRACE("Preprocessing started...");

        const size_t N = inputValues.size();

        // Removing negative BRDF values
        size_t negativeValuesCounter = 0;
        std::array<std::vector<FP>, _dimension> nonNegativeCoordinates;
        std::vector<FP> nonNegativeValues;

        for (size_t d = 0; d < _dimension; d++)
            nonNegativeCoordinates[d].reserve(N);
        nonNegativeValues.reserve(N);

        for (size_t i = 0; i < N; i++) {
            if (inputValues[i] < static_cast<FP>(0.)) {
                negativeValuesCounter++;
                continue;
            }

            for (size_t d = 0; d < _dimension; d++)
                nonNegativeCoordinates[d].push_back(inputCoordinates[d][i]);

            nonNegativeValues.push_back(inputValues[i]);
        }

        if (negativeValuesCounter > 0)
            LOG_WARNING("Negative values found during preprocess. Number"
                        " of values removed: ", negativeValuesCounter);

        const size_t nonNegativeN = nonNegativeValues.size();

        // Duplicates can be found using union-find data structure,
        // by uniting the indices of the coordinates that are equal
        // then by computing the mean value for each group.
        struct UnionFind final {
            std::vector<size_t> parent, rank;

            UnionFind(size_t n) : parent(n), rank(n, 0) {
                for (size_t i = 0; i < n; ++i) parent[i] = i;
            }

            size_t find(size_t x) {
                if (parent[x] != x)
                    parent[x] = find(parent[x]);
                return parent[x];
            }

            void unite(size_t x, size_t y) {
                size_t rootX = find(x), rootY = find(y);
                if (rootX != rootY) {
                    if (rank[rootX] < rank[rootY]) std::swap(rootX, rootY);
                    parent[rootY] = rootX;

                    if (rank[rootX] == rank[rootY]) rank[rootX]++;
                }
            }
        };

        UnionFind unionFind(nonNegativeN);

        // Research of duplicates : two coordinates will be considered as
        // duplicates if their distance is smaller than a certain epsilon
        const FP epsilon = std::numeric_limits<FP>::epsilon() * FP(10);

        for (size_t i = 0; i < nonNegativeN; i++) {
            for (size_t j = i + 1; j < nonNegativeN; j++) {

                FP distance = [&]<std::size_t... I>(std::index_sequence<I...>) {
                    return _topology.getDistanceScalar(nonNegativeCoordinates[I][i]..., nonNegativeCoordinates[I][j]...);
                }(std::make_index_sequence<_dimension>{});

                if (distance < epsilon) unionFind.unite(i, j);
            }
        }

        // Packing duplicates by groups of indices, and
        // then computing the mean value for each group
        std::unordered_map<size_t, std::vector<size_t>> groups;
        groups.reserve(nonNegativeN);

        for (size_t i = 0; i < nonNegativeN; i++)
            groups[unionFind.find(i)].push_back(i);

        std::array<std::vector<FP>, _dimension> preprocessedCoordinates;
        std::vector<FP> preprocessedValues;

        const size_t newN = groups.size();
        for (size_t d = 0; d < _dimension; d++)
            preprocessedCoordinates[d].reserve(newN);

        preprocessedValues.reserve(newN);

        // Computing mean value for each group of duplicates
        for (const auto& [root, indices] : groups) {
            // For each duplicate only a single coordinate is
            // kept, since they are all considered equivalent
            for (size_t d = 0; d < _dimension; d++)
                // indices[0] will be used as the reference cooordinate for the group
                preprocessedCoordinates[d].push_back(nonNegativeCoordinates[d][indices[0]]);

            // Computing mean value
            FP meanValue = FP(0.);
            for (size_t idx : indices)
                meanValue += nonNegativeValues[idx];

            meanValue /= static_cast<FP>(indices.size());

            preprocessedValues.push_back(meanValue);
        }

        LOG_DEBUG("Preprocessing complete. Initial number of coordinates: "
                  , N, " | vs number of duplicates removed: ", N - newN);

        return { std::move(preprocessedCoordinates), std::move(preprocessedValues) };
    }

    Eigen::Matrix<FP, Eigen::Dynamic, Eigen::Dynamic> _computeKernelDistanceMatrix(
        const std::array<std::vector<FP>, _dimension>& coordinates) const
    {
        const size_t N = coordinates[0].size();

        Eigen::Matrix<FP, Eigen::Dynamic, Eigen::Dynamic> kernelDistanceMatrix(N, N);

        // Since the kernel distance is symmetric, we either need to
        // computer the upper or lower triangular par of the matrix.
        for (size_t i = 0; i < N; i++) {
            if constexpr (KernelType::isAnisotropic) {
                // Anisotropic kernels use different distance structure
                rivoli::HemisphericDistances<FP, level> zeroDistances{};
                zeroDistances.hemisphericDistanceOne = vct::zero();
                zeroDistances.hemisphericDistanceTwo = vct::zero();

                // Values on the diagonal always take the value of the kernel, with a distance of zero
                kernelDistanceMatrix(i, i) = _kernel(zeroDistances).hsum() / static_cast<FP>(vct::width());
            }
            else {
                // Values on the diagonal always take the value of the kernel, with a distance of zero
                kernelDistanceMatrix(i, i) = _kernel(vct::zero()).hsum() / static_cast<FP>(vct::width());
            }

            for (size_t j = i + 1; j < N; j++) {
                FP kernelDistanceValue;

                if constexpr (KernelType::isAnisotropic) {
                    // Anisotropic kernels use different distance structure
                    auto distancesScalar = [&]<std::size_t... I>(std::index_sequence<I...>) {
                        return _topology.getDistancesScalar(coordinates[I][i]..., coordinates[I][j]...);
                    }(std::make_index_sequence<_dimension>{});

                    rivoli::HemisphericDistances<FP, level> distances{
                        vct(distancesScalar.hemisphericDistanceOne),
                        vct(distancesScalar.hemisphericDistanceTwo)
                    };

                    kernelDistanceValue = _kernel(distances).hsum() / static_cast<FP>(vct::width());
                }
                else {
                    FP distance = [&]<std::size_t... I>(std::index_sequence<I...>) {
                        return _topology.getDistanceScalar(coordinates[I][i]..., coordinates[I][j]...);
                    }(std::make_index_sequence<_dimension>{});

                    kernelDistanceValue = _kernel(vct(distance)).hsum() / static_cast<FP>(vct::width());
                }

                kernelDistanceMatrix(i, j) = kernelDistanceValue;
                kernelDistanceMatrix(j, i) = kernelDistanceValue; // Equal by symmetry
            }
        }

        return kernelDistanceMatrix;
    }

    const KernelType   _kernel;
    const TopologyType _topology;

    // Additional parameters
    bool _forceNonNegativity;

    std::array<vctVector, _dimension> _coordinates;
    vctVector						  _coefficients;
};


}
