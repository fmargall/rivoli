#pragma once

#include <array>
#include <vector>

#include <Eigen/Dense>

#include <vectra/vectra.hpp>
#include <tinylogger/tinylogger.hpp>

#include <rivoli/kernels/kernels.hpp>
#include <rivoli/topologies/topologies.hpp>

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
		: _kernel(std::move(kernel)), _topology(std::move(topology))
	{
		_setCoordinates (coordinates);
		_setCoefficients(coefficients);
	}

	Interpolator(
		const std::array<std::vector<FP>, _dimension>& inputCoordinates,
		const std::array<std::vector<FP>, _dimension>& outputCoordinates,
		const            std::vector<FP>&              inputValues,

		const KernelType& kernel,
		const TopologyType& topology
	)
		: _kernel(std::move(kernel)), _topology(std::move(topology))
	{
		for (size_t dimOne = 0; dimOne < _dimension; dimOne++) {
			if (inputCoordinates[dimOne].size() != inputValues.size())
				LOG_CRITICAL("Array sizes do not match. Input coordinates: ", inputCoordinates[dimOne].size(),
							 " | output coordinates: ", outputCoordinates[dimOne].size(), " | input values: ",
							 inputValues.size());

			if (inputCoordinates[dimOne].size() < outputCoordinates[dimOne].size())
				LOG_WARNING("Underdetermined problem: ", inputCoordinates[dimOne].size(), " data ",
							"points for ", outputCoordinates[dimOne].size(), " output RBF kernels.");

			for (size_t dimTwo = dimOne + 1; dimTwo < _dimension; dimTwo++) {
				if (inputCoordinates[dimOne].size() != inputCoordinates[dimTwo].size())
					LOG_CRITICAL("Input coordinates sizes do not match. Dimension ", dimOne, ": ",
								 inputCoordinates[dimOne].size(), "  |  Dimension ", dimTwo, ": ",
								 inputCoordinates[dimTwo].size());
				if (outputCoordinates[dimOne].size() != outputCoordinates[dimTwo].size())
					LOG_CRITICAL("Output coordinates sizes do not match. Dimension ", dimOne, ": ",
								 outputCoordinates[dimOne].size(), "  |  Dimension ", dimTwo, ": ",
								 outputCoordinates[dimTwo].size());
			}
		}

		// Eigen check for current SIMD instruction set version used, for debugging
		LOG_DEBUG("Current Eigen SIMD support: ", Eigen::SimdInstructionSetsInUse());

		// RBF interpolator is made by computing its coefficients, starting by
		// the computation of its kernel-distance matrix, using Eigen library.
		Eigen::Matrix<FP, Eigen::Dynamic, Eigen::Dynamic> kernelDistanceMatrix
			= _computeKernelDistanceMatrix(inputCoordinates, outputCoordinates);

		// Initialisation of result vector containing BRDF values
		Eigen::Vector<FP, Eigen::Dynamic> resultVector =
			Eigen::Map<const Eigen::Vector<FP, Eigen::Dynamic>>(
				inputValues.data(), inputValues.size());

		// Coefficients are computed using Eigen library
		Eigen::Vector<FP, Eigen::Dynamic> coefficients;

		// The system can be solved directly if we have a square matrix
		if (kernelDistanceMatrix.rows() == kernelDistanceMatrix.cols()) {
			// FullPivLU decomposition is used for a better stability, even if it is one of the worst for performance
			Eigen::FullPivLU<Eigen::Matrix<FP, Eigen::Dynamic, Eigen::Dynamic>> luDecomposition(kernelDistanceMatrix);
			coefficients = luDecomposition.solve(resultVector);
		}
		else
			LOG_CRITICAL("For now, only square kernel distance matrix is supported. Kernel distance matrix"
						 " rows: ", kernelDistanceMatrix.rows(), " | cols: ", kernelDistanceMatrix.cols());

		_setCoordinates(inputCoordinates);
		_setCoefficients(coefficients);

		LOG_INFO(topology.name, " interpolator initialised successfully.");
	}

	template <typename... CoordinatesType>
	FP interpolate (const CoordinatesType&... coordinates) const noexcept {
		
		// Convert scalar inputs into chosen SIMD backend type through vct
		std::array<vct, _dimension> coordinatesSIMD{ vct(coordinates)... };
		vct results = vct::zero();

		for (size_t i = 0; i < _coefficients.size(); i++) {
			vct distances = [&]<std::size_t... I>(std::index_sequence<I...>) {
				return _topology.getDistance(_coordinates[I][i]..., coordinatesSIMD[I]...);
			}(std::make_index_sequence<_dimension>{});

			results = results + _coefficients[i] * _kernel(distances);
		}

		return results.hsum();
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
				// Deprecated version. May be unsafe with SIMD
				/*
				(([&] {
					FP* arr = reinterpret_cast<FP*>(&remainingCoords[I]);
					for (size_t j = 0; j < remainingCoord; ++j)
						arr[j] = coordinates[I][(sizeOfSIMDData - 1) * vct::width() + j];
					}()), ...);
				*/
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

	void _setCoefficients(const std::vector<FP>& coefficients) noexcept {

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
			// Deprecated version. May be unsafe for SIMD
			/*
			// Initialisation of the last tail block to
			// zero for the remaining RBF coefficients:
			vct remainingCoefsBlock = vct::zero();

			// Filling the useful part of the tail block for the RBF coefficients
			FP* remainingCoefsArray = reinterpret_cast<FP*>(&remainingCoefsBlock);
			for (size_t j = 0; j < remainingCoefs; ++j)
				remainingCoefsArray[j] = coefficients[(sizeOfSIMDData - 1) * vct::width() + j];

			// Storing the final backend block for the coefficients
			_coefficients[sizeOfSIMDData - 1] = remainingCoefsBlock;
			*/
			// Initialisation of the last tail block to
			// zero for the remaining RBF coefficients:
			alignas(vct::alignment()) FP tmp[vct::width()] = {};

			// Filling the useful part of the tail block for the RBF coefficients
			for (size_t j = 0; j < remainingCoefs; ++j)
				tmp[j] = coefficients[(sizeOfSIMDData - 1) * vct::width() + j];

			// Storing final backend block for the coefficients
			_coefficients[sizeOfSIMDData - 1] = vct::loadu(tmp);
		}
	}

	Eigen::Matrix<FP, Eigen::Dynamic, Eigen::Dynamic> _computeKernelDistanceMatrix(
		const std::array<std::vector<FP>, _dimension>& coordinates) const 
	{
		const size_t N = coordinates[0].size();

		Eigen::Matrix<FP, Eigen::Dynamic, Eigen::Dynamic> kernelDistanceMatrix(N, N);

		// Since the kernel distance is symmetric, we either need to
		// computer the upper or lower triangular par of the matrix.
		for (size_t i = 0; i < N; i++) {
			// Values on the diagonal always take the value of the kernel, with a distance of zero
			kernelDistanceMatrix(i, i) = _kernel(vct::zero()).hsum() / static_cast<FP>(vct::width());

			for (size_t j = i + 1; j < N; j++) {

				FP distance = [&]<std::size_t... I>(std::index_sequence<I...>) {
					return _topology.getDistanceScalar(coordinates[I][i]..., coordinates[I][j]...);
				}(std::make_index_sequence<_dimension>{});

				FP kernelDistanceValue = _kernel(vct(distance)).hsum() / static_cast<FP>(vct::width());

				kernelDistanceMatrix(i, j) = kernelDistanceValue;
				kernelDistanceMatrix(j, i) = kernelDistanceValue; // Equal by symmetry
			}
		}

		return kernelDistanceMatrix;
	}

	const KernelType   _kernel;
	const TopologyType _topology;

	std::array<vctVector, _dimension> _coordinates;
	vctVector						  _coefficients;
};


}
