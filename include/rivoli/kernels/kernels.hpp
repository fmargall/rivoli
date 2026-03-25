#pragma once

#include <type_traits>

#include <vectra/vectra.hpp>


namespace rivoli {

template <typename _FP, vectra::SIMDLevel _level, typename DerivedKernel>
class Kernel {
public:
	using FP = _FP;
	static constexpr vectra::SIMDLevel level = _level;

	using vct = vectra::Vectratype<FP, level>;

	vct operator()(const vct& r) const {
		return static_cast<const DerivedKernel*>(this)->_runKernel(r);
	}

	// Scalar overload: only enabled when backend::type != FP (i.e. when we have a true SIMD type)
	//                  This will prevent ambiguity when backend::type == FP  (for scalar backend)
	FP operator()(FP r) const {
		vct result = (*this)(vct(r));
		if constexpr (vct::width() == 1)
			return result.value;
		else
			return result.hsum() / static_cast<FP>(vct::width());
	}
};

template <typename _FP, vectra::SIMDLevel _level>
class KernelLinear : public Kernel<_FP, _level, KernelLinear<_FP, _level>> {

	template <typename _FPAlias, vectra::SIMDLevel _levelAlias, typename DerivedKernel>
	friend class Kernel;

public:
	static constexpr std::string_view name = "Linear";

protected:
	using FP = _FP;
	static constexpr vectra::SIMDLevel level = _level;

	using vct = vectra::Vectratype<FP, level>;

	static vct _runKernel(const vct& r) {
		return r;
	}
};

template <typename _FP, vectra::SIMDLevel _level>
class KernelCubic : public Kernel<_FP, _level, KernelCubic<_FP, _level>> {

	template <typename _FPAlias, vectra::SIMDLevel _levelAlias, typename DerivedKernel>
	friend class Kernel;

public:
	static constexpr std::string_view name = "Cubic";

protected:
	using FP = _FP;
	static constexpr vectra::SIMDLevel level = _level;

	using vct = vectra::Vectratype<FP, level>;

	static vct _runKernel(const vct& r) {
		return r * r * r;
	}
};

}