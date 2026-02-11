#pragma once

#include <vectra/vectra.hpp>

template <typename _FP, vectra::SIMDLevel _level, typename DerivedTopology>
class Topology {
public:
	using FP = _FP;
	static constexpr vectra::SIMDLevel level = _level;

	using vct = vectra::Vectratype<FP, level>;

	template <typename... Args>
	vct getDistance(Args&&... args) const {
		return static_cast<const DerivedTopology*>(this)->_getDistance(std::forward<Args>(args)...);
	}

};