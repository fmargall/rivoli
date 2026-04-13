#include <rivoli/rivoli.hpp>

#include "bind_interpolator.hpp"


namespace nb = nanobind;

template <typename... Ts>
struct type_list {};

template<typename Enum, Enum... Values>
struct value_list {};

using SIMDLevels = value_list<
	vectra::SIMDLevel        , // Mandatory to expose type of enum before the values
	vectra::SIMDLevel::None  , // All of the following backends are not implemented.
	//vectra::SIMDLevel::SSE   ,
	//vectra::SIMDLevel::SSE2  ,
	//vectra::SIMDLevel::SSE3  ,
	vectra::SIMDLevel::SSE41 ,
	//vectra::SIMDLevel::SSE42 ,
	vectra::SIMDLevel::AVX
	//vectra::SIMDLevel::AVX2  ,
	//vectra::SIMDLevel::AVX512
>;

using KernelTypes = value_list<
	rivoli::KernelType              , // Mandatory to expose type of enum before the values
	rivoli::KernelType::Linear      ,
	rivoli::KernelType::Cubic       ,
	rivoli::KernelType::Epanechnikov,
	rivoli::KernelType::Gaussian    ,
	rivoli::KernelType::Laplacian
>;

template <size_t Dimension, bool Bilateral, bool Reciprocal, bool Isotropy, rivoli::CoordinateSystem coordSystem>
struct TopologyTags {
	static constexpr size_t dimension = Dimension;
	static constexpr bool bilateral   = Bilateral;
	static constexpr bool reciprocal  = Reciprocal;
	static constexpr bool isotropy    = Isotropy;
	static constexpr rivoli::CoordinateSystem coordSystem = coordSystem;
};

using TopologyTypes = type_list<
	TopologyTags<2, false, false, false, rivoli::CoordinateSystem::Spherical>   , // 2S
	TopologyTags<2, true , false, false, rivoli::CoordinateSystem::Spherical>   , // 2S      bilateral

	TopologyTags<3, false, false, false, rivoli::CoordinateSystem::Spherical>   , // 1R x 2S euclidean
	TopologyTags<3, true , false, false, rivoli::CoordinateSystem::Spherical>   , // 1R x 2S bilateral  euclidean
	TopologyTags<3, false, true , false, rivoli::CoordinateSystem::Spherical>   , // 1R x 2S reciprocal euclidean
	TopologyTags<3, true , true , false, rivoli::CoordinateSystem::Spherical>   , // 1R x 2S bilateral  reciprocal euclidean
	TopologyTags<3, false, true , false, rivoli::CoordinateSystem::Rusinkiewicz>, // 1R x 2S reciprocal Rusinkiewicz euclidean
	TopologyTags<3, true , true , false, rivoli::CoordinateSystem::Rusinkiewicz>, // 1R x 2S bilateral  reciprocal Rusinkiewicz euclidean

	TopologyTags<3, false, false, true , rivoli::CoordinateSystem::Spherical>   , // 1R x 2S isotropic  euclidean
	TopologyTags<3, true , false, true , rivoli::CoordinateSystem::Spherical>   , // 1R x 2S bilateral  isotropic  euclidean
	TopologyTags<3, false, true , true , rivoli::CoordinateSystem::Spherical>   , // 1R x 2S reciprocal isotropic  euclidean
	TopologyTags<3, true , true , true , rivoli::CoordinateSystem::Spherical>   , // 1R x 2S bilateral  reciprocal isotropic    euclidean
	TopologyTags<3, false, true , true , rivoli::CoordinateSystem::Rusinkiewicz>, // 1R x 2S reciprocal isotropic  Rusinkiewicz euclidean
	TopologyTags<3, true , true , true , rivoli::CoordinateSystem::Rusinkiewicz>, // 1R x 2S bilateral  reciprocal isotropic    Rusinkiewicz euclidean

	TopologyTags<4, false, false, false, rivoli::CoordinateSystem::Spherical>   , // 2S x 2S euclidean
	TopologyTags<4, true , false, false, rivoli::CoordinateSystem::Spherical>   , // 2S x 2S bilateral  euclidean
	TopologyTags<4, false, true , false, rivoli::CoordinateSystem::Spherical>   , // 2S x 2S reciprocal euclidean
	TopologyTags<4, true , true , false, rivoli::CoordinateSystem::Spherical>   , // 2S x 2S bilateral  reciprocal euclidean
	TopologyTags<4, false, true , false, rivoli::CoordinateSystem::Rusinkiewicz>, // 2S x 2S reciprocal Rusinkiewicz euclidean
	TopologyTags<4, true , true , false, rivoli::CoordinateSystem::Rusinkiewicz>  // 2S x 2S bilateral  reciprocal Rusinkiewicz euclidean
>;

NB_MODULE(_binding, m) {

	// Compile-time, for-loop over FP types
	[&] <typename... FPs>(type_list<FPs...>) {
		([&] {
			using FP = FPs;
			
			// Compile-time for-loop over SIMD levels
			[&] <vectra::SIMDLevel... Levels>(value_list<vectra::SIMDLevel, Levels...>) {
				([&] {
					constexpr auto level = Levels;

					// Compile-time for-loop over kernels
					[&] <rivoli::KernelType... Ks>(value_list<rivoli::KernelType, Ks...>) {
						([&] {
							constexpr auto kernelType = Ks;

							// Compile-time for-loop over topologies
							[&] <typename... Topologies>(type_list<Topologies...>) {
								([&] {
									using Topology = Topologies;

									constexpr size_t dim      = Topology::dimension;
									constexpr bool bilateral  = Topology::bilateral;
									constexpr bool reciprocal = Topology::reciprocal;
									constexpr bool isotropy   = Topology::isotropy;
									constexpr rivoli::CoordinateSystem coordSystem = Topology::coordSystem;

									rivoli::bindInterpolator<FP, level, dim, bilateral, reciprocal, isotropy, coordSystem, kernelType>(m);

								}(), ...);
							}(TopologyTypes{});

						}(), ...);
					}(KernelTypes{});

				}(), ...);
			}(SIMDLevels{});

		}(), ...);
	}(type_list<float, double>{});

}
