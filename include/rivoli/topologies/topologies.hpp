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
	static constexpr std::size_t dimension   = 2;

protected:
	using vct = vectra::Vectratype<FP, level>;

	FORCE_INLINE static vct _getDistance(vct thetaOne, vct phiOne,
										 vct thetaTwo, vct phiTwo)
	{
		return rivoli::distanceGreatCircle(thetaOne, phiOne, thetaTwo, phiTwo);
	}
};


}