#pragma once

#include <rivoli/kernels/kernels.hpp>

#include <vectra/vectra.hpp>


namespace rivoli {


enum class KernelType {
	Linear
};

template <KernelType kernelType, typename FP, vectra::SIMDLevel level>
struct KernelTraits;

template <KernelType kernelType, typename FP, vectra::SIMDLevel level>
using KernelSelector = typename KernelTraits<kernelType, FP, level>::type;

template <typename FP, vectra::SIMDLevel level>
struct KernelTraits<KernelType::Linear, FP, level> {
	using type = KernelLinear<FP, level>;
};


}