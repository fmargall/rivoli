#pragma once

#include <rivoli/kernels/kernels.hpp>

#include <vectra/vectra.hpp>


namespace rivoli {


enum class KernelType {
	Linear,
	Cubic,
	Epanechnikov,
	Gaussian,
	Laplacian,

	// Anisotropic kernels
	AnisotropicGaussian,
	AnisotropicLaplacian
};

template <KernelType kernelType, typename FP, vectra::SIMDLevel level>
struct KernelTraits;

template <KernelType kernelType, typename FP, vectra::SIMDLevel level>
using KernelSelector = typename KernelTraits<kernelType, FP, level>::type;


template <typename FP, vectra::SIMDLevel level>
struct KernelTraits<KernelType::Linear, FP, level> {
	using type = KernelLinear<FP, level>;
};

template <typename FP, vectra::SIMDLevel level>
struct KernelTraits<KernelType::Cubic, FP, level> {
	using type = KernelCubic<FP, level>;
};

template <typename FP, vectra::SIMDLevel level>
struct KernelTraits<KernelType::Epanechnikov, FP, level> {
	using type = KernelEpanechnikov<FP, level>;
};

template <typename FP, vectra::SIMDLevel level>
struct KernelTraits<KernelType::Gaussian, FP, level> {
	using type = KernelGaussian<FP, level>;
};

template <typename FP, vectra::SIMDLevel level>
struct KernelTraits<KernelType::Laplacian, FP, level> {
	using type = KernelLaplacian<FP, level>;
};


// Anisotropic kernels
template <typename FP, vectra::SIMDLevel level>
struct KernelTraits<KernelType::AnisotropicGaussian, FP, level> {
	using type = KernelAnisotropicGaussian<FP, level>;
};

template <typename FP, vectra::SIMDLevel level>
struct KernelTraits<KernelType::AnisotropicLaplacian, FP, level> {
	using type = KernelAnisotropicLaplacian<FP, level>;
};

}
