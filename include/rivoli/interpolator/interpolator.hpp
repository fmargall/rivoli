#pragma once

#include <array>
#include <vector>

#include <Eigen/Dense>

#include <rivoli/kernels/kernels.hpp>
#include <rivoli/topologies/topologies.hpp>

namespace rivoli {


template <typename KernelType, typename TopologyType>
class Interpolator {

	static constexpr vectra::SIMDLevel level = TopologyType::level;

	using FP  = typename TopologyType::FP;
	using vct = vectra::Vectratype<FP, level>;

	using vctVector = std::vector<vct, vectra::aligned_allocator<vct, vct::alignment()>>;

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

	const KernelType   _kernel;
	const TopologyType _topology;

	std::array<vctVector, _dimension> _coordinates;
	vctVector						  _coefficients;
};


}