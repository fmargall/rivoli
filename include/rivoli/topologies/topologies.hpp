#pragma once

#include <vectra/vectra.hpp>

#include <rivoli/distances/distances.hpp>

namespace rivoli
{


template <typename _FP, vectra::SIMDLevel _level, typename DerivedTopology>
class Topology {
public:
	using FP = _FP;
	static constexpr vectra::SIMDLevel level = _level;

	using vct = vectra::Vectratype<FP, level>;

	template <typename... Args>
	FORCE_INLINE vct getDistance(Args&&... args) const {
		return static_cast<const DerivedTopology*>(this)->_getDistance(std::forward<Args>(args)...);
	}

};

template <typename _FP, vectra::SIMDLevel _level>
class Topology2S : public Topology<_FP, _level, Topology2S<_FP, _level>> {

	template <typename _FPAlias, vectra::SIMDLevel _levelAlias, typename DerivedTopology>
	friend class Topology;

public:
	using FP = _FP;
	static constexpr vectra::SIMDLevel level = _level;
	static constexpr std::size_t dimension = 2;
	static constexpr std::string_view name = "2S";

protected:
	using vct = vectra::Vectratype<FP, level>;

	FORCE_INLINE static vct _getDistance(vct thetaOne, vct phiOne,
										 vct thetaTwo, vct phiTwo)
	{
		return rivoli::distanceGreatCircle(thetaOne, phiOne, thetaTwo, phiTwo);
	}
};

template <typename _FP, vectra::SIMDLevel _level>
class Topology2SBilateral : public Topology<_FP, _level, Topology2SBilateral<_FP, _level>> {

	template <typename _FPAlias, vectra::SIMDLevel _levelAlias, typename DerivedTopology>
	friend class Topology;

public:
	using FP = _FP;
	static constexpr vectra::SIMDLevel level = _level;
	static constexpr std::size_t dimension = 2;
	static constexpr std::string_view name = "2SBilateral";

protected:
	using vct = vectra::Vectratype<FP, level>;

	FORCE_INLINE static vct _getDistance(vct thetaOne, vct phiOne,
										 vct thetaTwo, vct phiTwo)
	{
		return rivoli::distanceGreatCircleBilateral(thetaOne, phiOne, thetaTwo, phiTwo);
	}
};

template <typename _FP, vectra::SIMDLevel _level>
class Topology2Sx2SEuclidean : public Topology<_FP, _level, Topology2Sx2SEuclidean<_FP, _level>> {

	template <typename _FPAlias, vectra::SIMDLevel _levelAlias, typename DerivedTopology>
	friend class Topology;

public:
	using FP = _FP;
	static constexpr vectra::SIMDLevel level = _level;
	static constexpr std::size_t dimension = 4;
	static constexpr std::string_view name = "2Sx2SEuclidean";

protected:
	using vct = vectra::Vectratype<FP, level>;

	FORCE_INLINE static vct _getDistance(vct thetaAOne, vct phiAOne, vct thetaBOne, vct phiBOne,
										 vct thetaATwo, vct phiATwo, vct thetaBTwo, vct phiBTwo)
	{
		return rivoli::distance2Sx2SEuclidean(thetaAOne, phiAOne, thetaBOne, phiBOne,
											  thetaATwo, phiATwo, thetaBTwo, phiBTwo);
	}
};

template <typename _FP, vectra::SIMDLevel _level>
class Topology2Sx2SBilateralEuclidean : public Topology<_FP, _level, Topology2Sx2SBilateralEuclidean<_FP, _level>> {

	template <typename _FPAlias, vectra::SIMDLevel _levelAlias, typename DerivedTopology>
	friend class Topology;

public:
	using FP = _FP;
	static constexpr vectra::SIMDLevel level = _level;
	static constexpr std::size_t dimension = 4;
	static constexpr std::string_view name = "2Sx2SBilateralEuclidean";

protected:
	using vct = vectra::Vectratype<FP, level>;

	FORCE_INLINE static vct _getDistance(vct thetaAOne, vct phiAOne, vct thetaBOne, vct phiBOne,
										 vct thetaATwo, vct phiATwo, vct thetaBTwo, vct phiBTwo)
	{
		return rivoli::distance2Sx2SBilateralEuclidean(thetaAOne, phiAOne, thetaBOne, phiBOne,
													   thetaATwo, phiATwo, thetaBTwo, phiBTwo);
	}
};

template <typename _FP, vectra::SIMDLevel _level>
class Topology2Sx2SReciprocalEuclidean : public Topology<_FP, _level, Topology2Sx2SReciprocalEuclidean<_FP, _level>> {

	template <typename _FPAlias, vectra::SIMDLevel _levelAlias, typename DerivedTopology>
	friend class Topology;

public:
	using FP = _FP;
	static constexpr vectra::SIMDLevel level = _level;
	static constexpr std::size_t dimension = 4;
	static constexpr std::string_view name = "2Sx2SReciprocalEuclidean";

protected:
	using vct = vectra::Vectratype<FP, level>;

	FORCE_INLINE static vct _getDistance(vct thetaAOne, vct phiAOne, vct thetaBOne, vct phiBOne,
										 vct thetaATwo, vct phiATwo, vct thetaBTwo, vct phiBTwo)
	{
		return rivoli::distance2Sx2SReciprocalEuclidean(thetaAOne, phiAOne, thetaBOne, phiBOne,
													    thetaATwo, phiATwo, thetaBTwo, phiBTwo);
	}
};

template <typename _FP, vectra::SIMDLevel _level>
class Topology2Sx2SBilateralReciprocalEuclidean : public Topology<_FP, _level, Topology2Sx2SBilateralReciprocalEuclidean<_FP, _level>> {

	template <typename _FPAlias, vectra::SIMDLevel _levelAlias, typename DerivedTopology>
	friend class Topology;

public:
	using FP = _FP;
	static constexpr vectra::SIMDLevel level = _level;
	static constexpr std::size_t dimension = 4;
	static constexpr std::string_view name = "2Sx2SBilateralReciprocalEuclidean";

protected:
	using vct = vectra::Vectratype<FP, level>;

	FORCE_INLINE static vct _getDistance(vct thetaAOne, vct phiAOne, vct thetaBOne, vct phiBOne,
										 vct thetaATwo, vct phiATwo, vct thetaBTwo, vct phiBTwo)
	{
		return rivoli::distance2Sx2SBilateralReciprocalEuclidean(thetaAOne, phiAOne, thetaBOne, phiBOne,
																 thetaATwo, phiATwo, thetaBTwo, phiBTwo);
	}
};

}
