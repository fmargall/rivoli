#pragma once

#include <type_traits>

#include <vectra/vectra.hpp>

#include <rivoli/distances/hemispheric_distances.hpp>


namespace rivoli {

template <typename _FP, vectra::SIMDLevel _level, typename DerivedKernel>
class Kernel {
public:
	using FP = _FP;
	static constexpr vectra::SIMDLevel level = _level;

	using vct = vectra::Vectratype<FP, level>;

	// Default, can be overridden by derived kernel
	static constexpr bool isAnisotropic = false;

	vct operator()(const vct& r) const
		// Requirements for the activation of the generic overload
		requires requires (const DerivedKernel& k, const vct& x) {
			{ k._runKernel(x) } -> std::same_as<vct>;
		}
	{
		return static_cast<const DerivedKernel*>(this)->_runKernel(r);
	}

	// HemisphericDistance overload: enabled when backend::type != FP
	template <typename T>
		// Requirements for the activation of this overload
		requires (!std::same_as<std::decay_t<T>, FP>&&
			requires (const DerivedKernel& k, const T& x) {
		k._runKernel(x);})
	auto operator()(const T& x) const {
		return static_cast<const DerivedKernel*>(this)->_runKernel(x);
	}

	// Scalar overload: only enabled when backend::type != FP (i.e. when we have a true SIMD type)
	//                  This will prevent ambiguity when backend::type == FP  (for scalar backend)
	FP operator()(FP r) const
		// Requirements for the activation of the scalar overload
		requires requires (const DerivedKernel& k, const vct& x) {
			{ k._runKernel(x) } -> std::same_as<vct>;
		}
	{
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

template <typename _FP, vectra::SIMDLevel _level>
class KernelEpanechnikov : public Kernel<_FP, _level, KernelEpanechnikov<_FP, _level>> {

	template <typename _FPAlias, vectra::SIMDLevel _levelAlias, typename DerivedKernel>
	friend class Kernel;

protected:
	using FP = _FP;
	static constexpr vectra::SIMDLevel level = _level;

	using vct = vectra::Vectratype<FP, level>;

public:
	static constexpr std::string_view name = "Epanechnikov";

	// Since this kernel works with a parameter, an explicit constructor is required
	explicit KernelEpanechnikov(FP sigma) : _invSigmaSquared(vct(1. / (sigma * sigma))) {}

protected:
	// This is a non-static method, since this kernel needs a parameter to be computed
	// In this case, it is also required to add an explicit constructor to initialize.
	vct _runKernel(const vct& r) const {
		return vct::max(vct::zero(), vct(1) - (r * r) * _invSigmaSquared);
	}

private:
	vct _invSigmaSquared;
};

template <typename _FP, vectra::SIMDLevel _level>
class KernelGaussian : public Kernel<_FP, _level, KernelGaussian<_FP, _level>> {

	template <typename _FPAlias, vectra::SIMDLevel _levelAlias, typename DerivedKernel>
	friend class Kernel;

protected:
	using FP = _FP;
	static constexpr vectra::SIMDLevel level = _level;

	using vct = vectra::Vectratype<FP, level>;

public:
	static constexpr std::string_view name = "Gaussian";

	// Since this kernel works with a parameter, an explicit constructor is required
	explicit KernelGaussian(FP sigma) : _invTwoSigmaSq(vct(1. / (2. * sigma * sigma))) {}

protected:
	// This is a non-static method, since this kernel needs a parameter to be computed
	// In this case, it is also required to add an explicit constructor to initialize.
	vct _runKernel(const vct& r) const {
		return vct::exp(-(r * r) * _invTwoSigmaSq);
	}

private:
	vct _invTwoSigmaSq;
};

template <typename _FP, vectra::SIMDLevel _level>
class KernelLaplacian : public Kernel<_FP, _level, KernelLaplacian<_FP, _level>> {

	template <typename _FPAlias, vectra::SIMDLevel _levelAlias, typename DerivedKernel>
	friend class Kernel;

protected:
	using FP = _FP;
	static constexpr vectra::SIMDLevel level = _level;

	using vct = vectra::Vectratype<FP, level>;

public:
	static constexpr std::string_view name = "Laplacian";

	// Since this kernel works with a parameter, an explicit constructor is required
	explicit KernelLaplacian(FP sigma) : _invSigma(vct(1. / sigma)) {}

protected:
	// This is a non-static method, since this kernel needs a parameter to be computed
	// In this case, it is also required to add an explicit constructor to initialize.
	vct _runKernel(const vct& r) const {
		// The Laplacian kernel is normally defined as exp(-|r| / sigma), but since we
		// know that r is always positive (since it is a distance), we can simplify it
		return vct::exp(- r * _invSigma);
	}

private:
	vct _invSigma;

};


// Some kernels may be defined as anisotropic. In this case these ones work
// with a particular decomposed distance function, that will be expected to
// return the distance components separately.
template <typename _FP, vectra::SIMDLevel _level>
class KernelAnisotropicGaussian : public Kernel<_FP, _level, KernelAnisotropicGaussian<_FP, _level>> {

	template <typename _FPAlias, vectra::SIMDLevel _levelAlias, typename DerivedKernel>
	friend class Kernel;

protected:
	using FP = _FP;
	static constexpr vectra::SIMDLevel level = _level;

	using vct = vectra::Vectratype<FP, level>;

public:
	static constexpr std::string_view name = "AnisotropicGaussian";

	static constexpr bool isAnisotropic = true;

	// Since this kernel works with parameters, an explicit constructor is required
	explicit KernelAnisotropicGaussian(FP sigmaOne, FP sigmaTwo) : _invTwoSigmaOneSq(vct(1. / (2. * sigmaOne * sigmaOne))),
													               _invTwoSigmaTwoSq(vct(1. / (2. * sigmaTwo * sigmaTwo))) {}

protected:
	// This is a non-static method, since this kernel needs parameters to be computed
	// In this case, it is also required to add an explicit constructor to initialize
	vct _runKernel(const rivoli::HemisphericDistances<FP, level>& distances) const {
		return vct::exp(-(distances.hemisphericDistanceOne * distances.hemisphericDistanceOne) * _invTwoSigmaOneSq -
		                 (distances.hemisphericDistanceTwo * distances.hemisphericDistanceTwo) * _invTwoSigmaTwoSq);
	}

private:
	vct _invTwoSigmaOneSq;
	vct _invTwoSigmaTwoSq;

};

template <typename _FP, vectra::SIMDLevel _level>
class KernelAnisotropicLaplacian : public Kernel<_FP, _level, KernelAnisotropicLaplacian<_FP, _level>> {

	template <typename _FPAlias, vectra::SIMDLevel _levelAlias, typename DerivedKernel>
	friend class Kernel;

protected:
	using FP = _FP;
	static constexpr vectra::SIMDLevel level = _level;

	using vct = vectra::Vectratype<FP, level>;

public:
	static constexpr std::string_view name = "AnisotropicLaplacian";

	static constexpr bool isAnisotropic = true;

	// Since this kernel works with parameters, an explicit constructor is required
	explicit KernelAnisotropicLaplacian(FP sigmaOne, FP sigmaTwo) : _invSigmaOne(vct(1. / sigmaOne)),
													                _invSigmaTwo(vct(1. / sigmaTwo)) {}

protected:
	// This is a non-static method, since this kernel needs parameters to be computed
	// In this case, it is also required to add an explicit constructor to initialize
	vct _runKernel(const rivoli::HemisphericDistances<FP, level>& distances) const {
		return vct::exp(-vct::abs(distances.hemisphericDistanceOne) * _invSigmaOne -
		                 vct::abs(distances.hemisphericDistanceTwo) * _invSigmaTwo);
	}

private:
	vct _invSigmaOne;
	vct _invSigmaTwo;

};

}
