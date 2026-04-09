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

	template <typename... ScalarArgs>
	FORCE_INLINE FP getDistanceScalar(ScalarArgs&&... args) const {
		return getDistance(vct(std::forward<ScalarArgs>(args))...).hsum() / static_cast<FP>(vct::width());
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
class Topology1Rx2SEuclidean : public Topology<_FP, _level, Topology1Rx2SEuclidean<_FP, _level>> {

	template <typename _FPAlias, vectra::SIMDLevel _levelAlias, typename DerivedTopology>
	friend class Topology;

public:
	using FP = _FP;
	static constexpr vectra::SIMDLevel level = _level;
	static constexpr std::size_t dimension = 3;
	static constexpr std::string_view name = "1Rx2SEuclidean";

protected:
	using vct = vectra::Vectratype<FP, level>;

	FORCE_INLINE static vct _getDistance(vct thetaAOne, vct thetaBOne, vct phiBOne,
										 vct thetaATwo, vct thetaBTwo, vct phiBTwo)
	{
		return rivoli::distance1Rx2SEuclidean(thetaAOne, thetaBOne, phiBOne,
											  thetaATwo, thetaBTwo, phiBTwo);
	}
};

template <typename _FP, vectra::SIMDLevel _level>
class Topology1Rx2SIsotropicEuclidean : public Topology<_FP, _level, Topology1Rx2SIsotropicEuclidean<_FP, _level>> {

	template <typename _FPAlias, vectra::SIMDLevel _levelAlias, typename DerivedTopology>
	friend class Topology;

public:
	using FP = _FP;
	static constexpr vectra::SIMDLevel level = _level;
	static constexpr std::size_t dimension = 3;
	static constexpr std::string_view name = "1Rx2SIsotropicEuclidean";

protected:
	using vct = vectra::Vectratype<FP, level>;

	FORCE_INLINE static vct _getDistance(vct thetaAOne, vct thetaBOne, vct phiBOne,
										 vct thetaATwo, vct thetaBTwo, vct phiBTwo)
	{
		return rivoli::distance1Rx2SIsotropicEuclidean(thetaAOne, thetaBOne, phiBOne,
											           thetaATwo, thetaBTwo, phiBTwo);
	}
};

template <typename _FP, vectra::SIMDLevel _level>
class Topology1Rx2SBilateralEuclidean : public Topology<_FP, _level, Topology1Rx2SBilateralEuclidean<_FP, _level>> {

	template <typename _FPAlias, vectra::SIMDLevel _levelAlias, typename DerivedTopology>
	friend class Topology;

public:
	using FP = _FP;
	static constexpr vectra::SIMDLevel level = _level;
	static constexpr std::size_t dimension = 3;
	static constexpr std::string_view name = "1Rx2SBilateralEuclidean";

protected:
	using vct = vectra::Vectratype<FP, level>;

	FORCE_INLINE static vct _getDistance(vct thetaAOne, vct thetaBOne, vct phiBOne,
										 vct thetaATwo, vct thetaBTwo, vct phiBTwo)
	{
		return rivoli::distance1Rx2SBilateralEuclidean(thetaAOne, thetaBOne, phiBOne,
											           thetaATwo, thetaBTwo, phiBTwo);
	}
};

template <typename _FP, vectra::SIMDLevel _level>
class Topology1Rx2SBilateralIsotropicEuclidean : public Topology<_FP, _level, Topology1Rx2SBilateralIsotropicEuclidean<_FP, _level>> {

	template <typename _FPAlias, vectra::SIMDLevel _levelAlias, typename DerivedTopology>
	friend class Topology;

public:
	using FP = _FP;
	static constexpr vectra::SIMDLevel level = _level;
	static constexpr std::size_t dimension = 3;
	static constexpr std::string_view name = "1Rx2SBilateralIsotropicEuclidean";

protected:
	using vct = vectra::Vectratype<FP, level>;

	FORCE_INLINE static vct _getDistance(vct thetaAOne, vct thetaBOne, vct phiBOne,
										 vct thetaATwo, vct thetaBTwo, vct phiBTwo)
	{
		return rivoli::distance1Rx2SBilateralIsotropicEuclidean(thetaAOne, thetaBOne, phiBOne,
											                    thetaATwo, thetaBTwo, phiBTwo);
	}
};

template <typename _FP, vectra::SIMDLevel _level>
class Topology1Rx2SReciprocalEuclidean : public Topology<_FP, _level, Topology1Rx2SReciprocalEuclidean<_FP, _level>> {

	template <typename _FPAlias, vectra::SIMDLevel _levelAlias, typename DerivedTopology>
	friend class Topology;

public:
	using FP = _FP;
	static constexpr vectra::SIMDLevel level = _level;
	static constexpr std::size_t dimension = 3;
	static constexpr std::string_view name = "1Rx2SReciprocalEuclidean";

protected:
	using vct = vectra::Vectratype<FP, level>;

	FORCE_INLINE static vct _getDistance(vct thetaAOne, vct thetaBOne, vct phiBOne,
										 vct thetaATwo, vct thetaBTwo, vct phiBTwo)
	{
		return rivoli::distance1Rx2SReciprocalEuclidean(thetaAOne, thetaBOne, phiBOne,
											            thetaATwo, thetaBTwo, phiBTwo);
	}
};

template <typename _FP, vectra::SIMDLevel _level>
class Topology1Rx2SReciprocalIsotropicEuclidean : public Topology<_FP, _level, Topology1Rx2SReciprocalIsotropicEuclidean<_FP, _level>> {

	template <typename _FPAlias, vectra::SIMDLevel _levelAlias, typename DerivedTopology>
	friend class Topology;

public:
	using FP = _FP;
	static constexpr vectra::SIMDLevel level = _level;
	static constexpr std::size_t dimension = 3;
	static constexpr std::string_view name = "1Rx2SReciprocalIsotropicEuclidean";

protected:
	using vct = vectra::Vectratype<FP, level>;

	FORCE_INLINE static vct _getDistance(vct thetaAOne, vct thetaBOne, vct phiBOne,
										 vct thetaATwo, vct thetaBTwo, vct phiBTwo)
	{
		return rivoli::distance1Rx2SReciprocalIsotropicEuclidean(thetaAOne, thetaBOne, phiBOne,
											                     thetaATwo, thetaBTwo, phiBTwo);
	}
};

template <typename _FP, vectra::SIMDLevel _level>
class Topology1Rx2SBilateralReciprocalEuclidean : public Topology<_FP, _level, Topology1Rx2SBilateralReciprocalEuclidean<_FP, _level>> {

	template <typename _FPAlias, vectra::SIMDLevel _levelAlias, typename DerivedTopology>
	friend class Topology;

public:
	using FP = _FP;
	static constexpr vectra::SIMDLevel level = _level;
	static constexpr std::size_t dimension = 3;
	static constexpr std::string_view name = "1Rx2SBilateralReciprocalEuclidean";

protected:
	using vct = vectra::Vectratype<FP, level>;

	FORCE_INLINE static vct _getDistance(vct thetaAOne, vct thetaBOne, vct phiBOne,
										 vct thetaATwo, vct thetaBTwo, vct phiBTwo)
	{
		return rivoli::distance1Rx2SBilateralReciprocalEuclidean(thetaAOne, thetaBOne, phiBOne,
											                     thetaATwo, thetaBTwo, phiBTwo);
	}
};

template <typename _FP, vectra::SIMDLevel _level>
class Topology1Rx2SBilateralReciprocalIsotropicEuclidean : public Topology<_FP, _level, Topology1Rx2SBilateralReciprocalIsotropicEuclidean<_FP, _level>> {

	template <typename _FPAlias, vectra::SIMDLevel _levelAlias, typename DerivedTopology>
	friend class Topology;

public:
	using FP = _FP;
	static constexpr vectra::SIMDLevel level = _level;
	static constexpr std::size_t dimension = 3;
	static constexpr std::string_view name = "1Rx2SBilateralReciprocalIsotropicEuclidean";

protected:
	using vct = vectra::Vectratype<FP, level>;

	FORCE_INLINE static vct _getDistance(vct thetaAOne, vct thetaBOne, vct phiBOne,
										 vct thetaATwo, vct thetaBTwo, vct phiBTwo)
	{
		return rivoli::distance1Rx2SBilateralReciprocalIsotropicEuclidean(thetaAOne, thetaBOne, phiBOne,
											                              thetaATwo, thetaBTwo, phiBTwo);
	}
};

template <typename _FP, vectra::SIMDLevel _level>
class Topology1Rx2SReciprocalRusinkiewiczEuclidean : public Topology<_FP, _level, Topology1Rx2SReciprocalRusinkiewiczEuclidean<_FP, _level>> {

	template <typename _FPAlias, vectra::SIMDLevel _levelAlias, typename DerivedTopology>
	friend class Topology;

public:
	using FP = _FP;
	static constexpr vectra::SIMDLevel level = _level;
	static constexpr std::size_t dimension = 3;
	static constexpr std::string_view name = "1Rx2SReciprocalRusinkiewiczEuclidean";

protected:
	using vct = vectra::Vectratype<FP, level>;

	FORCE_INLINE static vct _getDistance(vct thetaAOne, vct thetaBOne, vct phiBOne,
										 vct thetaATwo, vct thetaBTwo, vct phiBTwo)
	{
		return rivoli::distance1Rx2SReciprocalRusinkiewiczEuclidean(thetaAOne, thetaBOne, phiBOne,
											                        thetaATwo, thetaBTwo, phiBTwo);
	}
};

template <typename _FP, vectra::SIMDLevel _level>
class Topology1Rx2SReciprocalRusinkiewiczIsotropicEuclidean : public Topology<_FP, _level, Topology1Rx2SReciprocalRusinkiewiczIsotropicEuclidean<_FP, _level>> {

	template <typename _FPAlias, vectra::SIMDLevel _levelAlias, typename DerivedTopology>
	friend class Topology;

public:
	using FP = _FP;
	static constexpr vectra::SIMDLevel level = _level;
	static constexpr std::size_t dimension = 3;
	static constexpr std::string_view name = "1Rx2SReciprocalRusinkiewiczIsotropicEuclidean";

protected:
	using vct = vectra::Vectratype<FP, level>;

	FORCE_INLINE static vct _getDistance(vct thetaAOne, vct thetaBOne, vct phiBOne,
										 vct thetaATwo, vct thetaBTwo, vct phiBTwo)
	{
		return rivoli::distance1Rx2SReciprocalRusinkiewiczIsotropicEuclidean(thetaAOne, thetaBOne, phiBOne,
											                                 thetaATwo, thetaBTwo, phiBTwo);
	}
};

template <typename _FP, vectra::SIMDLevel _level>
class Topology1Rx2SBilateralReciprocalRusinkiewiczEuclidean : public Topology<_FP, _level, Topology1Rx2SBilateralReciprocalRusinkiewiczEuclidean<_FP, _level>> {

	template <typename _FPAlias, vectra::SIMDLevel _levelAlias, typename DerivedTopology>
	friend class Topology;

public:
	using FP = _FP;
	static constexpr vectra::SIMDLevel level = _level;
	static constexpr std::size_t dimension = 3;
	static constexpr std::string_view name = "1Rx2SBilateralReciprocalRusinkiewiczEuclidean";

protected:
	using vct = vectra::Vectratype<FP, level>;

	FORCE_INLINE static vct _getDistance(vct thetaAOne, vct thetaBOne, vct phiBOne,
										 vct thetaATwo, vct thetaBTwo, vct phiBTwo)
	{
		return rivoli::distance1Rx2SBilateralReciprocalRusinkiewiczEuclidean(thetaAOne, thetaBOne, phiBOne,
											                                 thetaATwo, thetaBTwo, phiBTwo);
	}
};

template <typename _FP, vectra::SIMDLevel _level>
class Topology1Rx2SBilateralReciprocalRusinkiewiczIsotropicEuclidean : public Topology<_FP, _level, Topology1Rx2SBilateralReciprocalRusinkiewiczIsotropicEuclidean<_FP, _level>> {

	template <typename _FPAlias, vectra::SIMDLevel _levelAlias, typename DerivedTopology>
	friend class Topology;

public:
	using FP = _FP;
	static constexpr vectra::SIMDLevel level = _level;
	static constexpr std::size_t dimension = 3;
	static constexpr std::string_view name = "1Rx2SBilateralReciprocalRusinkiewiczIsotropicEuclidean";

protected:
	using vct = vectra::Vectratype<FP, level>;

	FORCE_INLINE static vct _getDistance(vct thetaAOne, vct thetaBOne, vct phiBOne,
										 vct thetaATwo, vct thetaBTwo, vct phiBTwo)
	{
		return rivoli::distance1Rx2SBilateralReciprocalRusinkiewiczIsotropicEuclidean(thetaAOne, thetaBOne, phiBOne,
											                                          thetaATwo, thetaBTwo, phiBTwo);
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

template <typename _FP, vectra::SIMDLevel _level>
class Topology2Sx2SReciprocalRusinkiewiczEuclidean : public Topology<_FP, _level, Topology2Sx2SReciprocalRusinkiewiczEuclidean<_FP, _level>> {

	template <typename _FPAlias, vectra::SIMDLevel _levelAlias, typename DerivedTopology>
	friend class Topology;

public:
	using FP = _FP;
	static constexpr vectra::SIMDLevel level = _level;
	static constexpr std::size_t dimension = 4;
	static constexpr std::string_view name = "2Sx2SReciprocalRusinkiewiczEuclidean";

protected:
	using vct = vectra::Vectratype<FP, level>;

	FORCE_INLINE static vct _getDistance(vct thetaHOne, vct phiHOne, vct thetaDOne, vct phiDOne,
										 vct thetaHTwo, vct phiHTwo, vct thetaDTwo, vct phiDTwo)
	{
		return rivoli::distance2Sx2SReciprocalRusinkiewiczEuclidean(thetaHOne, phiHOne, thetaDOne, phiDOne,
																    thetaHTwo, phiHTwo, thetaDTwo, phiDTwo);
	}
};

template <typename _FP, vectra::SIMDLevel _level>
class Topology2Sx2SBilateralReciprocalRusinkiewiczEuclidean : public Topology<_FP, _level, Topology2Sx2SBilateralReciprocalRusinkiewiczEuclidean<_FP, _level>> {

	template <typename _FPAlias, vectra::SIMDLevel _levelAlias, typename DerivedTopology>
	friend class Topology;

public:
	using FP = _FP;
	static constexpr vectra::SIMDLevel level = _level;
	static constexpr std::size_t dimension = 4;
	static constexpr std::string_view name = "2Sx2SBilateralReciprocalRusinkiewiczEuclidean";

protected:
	using vct = vectra::Vectratype<FP, level>;

	FORCE_INLINE static vct _getDistance(vct thetaHOne, vct phiHOne, vct thetaDOne, vct phiDOne,
										 vct thetaHTwo, vct phiHTwo, vct thetaDTwo, vct phiDTwo)
	{
		return rivoli::distance2Sx2SBilateralReciprocalRusinkiewiczEuclidean(thetaHOne, phiHOne, thetaDOne, phiDOne,
																			 thetaHTwo, phiHTwo, thetaDTwo, phiDTwo);
	}
};

}
