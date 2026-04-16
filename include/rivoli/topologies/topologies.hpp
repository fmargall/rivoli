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

	// By default. Can be overridden for each topology
	static constexpr bool hasDecomposedDistance = true;

	template <typename... Args>
	FORCE_INLINE vct getDistance(Args&&... args) const {
		return static_cast<const DerivedTopology*>(this)->_getDistance(std::forward<Args>(args)...);
	}

	template <typename... ScalarArgs>
	FORCE_INLINE FP getDistanceScalar(ScalarArgs&&... args) const {
		return getDistance(vct(std::forward<ScalarArgs>(args))...).hsum() / static_cast<FP>(vct::width());
	}


	// Some topologies can be computed with their components solved separately
	// This may be particularly useful for the use of some anisotropic kernels
	template <typename... Args>
	FORCE_INLINE auto getDistances(Args&&... args) const
		// Requirements for the activation of this getDistances overload
		requires requires (const DerivedTopology& topology, Args&&... xs) {
			topology._getDistances(std::forward<Args>(xs)...);
		}
	{
		return static_cast<const DerivedTopology*>(this)->_getDistances(std::forward<Args>(args)...);
	}

	template <typename... ScalarArgs>
	FORCE_INLINE auto getDistancesScalar(ScalarArgs&&... args) const {
		auto distances = getDistances(vct(std::forward<ScalarArgs>(args))...);

		return HemisphericDistancesScalar<FP>{
			distances.hemisphericDistanceOne.hsum() / static_cast<FP>(vct::width()),
			distances.hemisphericDistanceTwo.hsum() / static_cast<FP>(vct::width())
		};
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

	// This topology can be computed with its components solved separately
	// This is particularly useful for the use of some anisotropic kernels
	FORCE_INLINE static HemisphericDistances<FP, level> _getDistances(vct thetaOne, vct phiOne,
																	  vct thetaTwo, vct phiTwo)
	{
		return rivoli::hemisphericDistanceGreatCircle(thetaOne, phiOne, thetaTwo, phiTwo);
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

	// This topology can be computed with its components solved separately
	// This is particularly useful for the use of some anisotropic kernels
	FORCE_INLINE static HemisphericDistances<FP, level> _getDistances(vct thetaOne, vct phiOne,
																	  vct thetaTwo, vct phiTwo)
	{
		return rivoli::hemisphericDistanceGreatCircleBilateral(thetaOne, phiOne, thetaTwo, phiTwo);
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

	// This topology can be computed with its components solved separately
	// This is particularly useful for the use of some anisotropic kernels
	FORCE_INLINE static HemisphericDistances<FP, level> _getDistances(vct thetaAOne, vct thetaBOne, vct phiBOne,
										                              vct thetaATwo, vct thetaBTwo, vct phiBTwo)
	{
		return rivoli::hemisphericDistance1Rx2S(thetaAOne, thetaBOne, phiBOne,
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

	// NB: because of its isotropic nature, this topology cannot be computed with
	// its components solved separately, thus we may not implement _getDistances.
	static constexpr bool hasDecomposedDistance = false;

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

	// This topology can be computed with its components solved separately
	// This is particularly useful for the use of some anisotropic kernels
	FORCE_INLINE static HemisphericDistances<FP, level> _getDistances(vct thetaAOne, vct thetaBOne, vct phiBOne,
										                              vct thetaATwo, vct thetaBTwo, vct phiBTwo)
	{
		return rivoli::hemisphericDistance1Rx2SBilateral(thetaAOne, thetaBOne, phiBOne,
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

	// NB: because of its isotropic nature, this topology cannot be computed with
	// its components solved separately, thus we may not implement _getDistances.
	static constexpr bool hasDecomposedDistance = false;

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

	// NB: because of its reciprocal nature combined to a spherical parameterization,
	// the two hemispheres of this topology are strongly coupled, thus we should not
	// implement _getDistances.
	static constexpr bool hasDecomposedDistance = false;

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

	// NB: because of its isotropic nature, this topology cannot be computed with
	// its components solved separately, thus we may not implement _getDistances.
	static constexpr bool hasDecomposedDistance = false;

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

	// NB: because of its reciprocal nature combined to a spherical parameterization,
	// the two hemispheres of this topology are strongly coupled, thus we should not
	// implement _getDistances.
	static constexpr bool hasDecomposedDistance = false;

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

	// NB: because of its isotropic nature, this topology cannot be computed with
	// its components solved separately, thus we may not implement _getDistances.
	static constexpr bool hasDecomposedDistance = false;

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

	// This topology can be computed with its components solved separately
	// This is particularly useful for the use of some anisotropic kernels
	FORCE_INLINE static HemisphericDistances<FP, level> _getDistances(vct thetaAOne, vct thetaBOne, vct phiBOne,
																	  vct thetaATwo, vct thetaBTwo, vct phiBTwo)
	{
		return rivoli::hemisphericDistance1Rx2SReciprocalRusinkiewicz(thetaAOne, thetaBOne, phiBOne,
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

	// NB: because of its isotropic nature, this topology cannot be computed with
	// its components solved separately, thus we may not implement _getDistances.
	static constexpr bool hasDecomposedDistance = false;

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

	// This topology can be computed with its components solved separately
	// This is particularly useful for the use of some anisotropic kernels
	FORCE_INLINE static HemisphericDistances<FP, level> _getDistances(vct thetaAOne, vct thetaBOne, vct phiBOne,
																	  vct thetaATwo, vct thetaBTwo, vct phiBTwo)
	{
		return rivoli::hemisphericDistance1Rx2SBilateralReciprocalRusinkiewicz(thetaAOne, thetaBOne, phiBOne,
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

	// NB: because of its isotropic nature, this topology cannot be computed with
	// its components solved separately, thus we may not implement _getDistances.
	static constexpr bool hasDecomposedDistance = false;

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

	// This topology can be computed with its components solved separately
	// This is particularly useful for the use of some anisotropic kernels
	FORCE_INLINE static HemisphericDistances<FP, level> _getDistances(vct thetaAOne, vct phiAOne, vct thetaBOne, vct phiBOne,
																	  vct thetaATwo, vct phiATwo, vct thetaBTwo, vct phiBTwo)
	{
		return rivoli::hemisphericDistance2Sx2S(thetaAOne, phiAOne, thetaBOne, phiBOne,
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

	// This topology can be computed with its components solved separately
	// This is particularly useful for the use of some anisotropic kernels
	FORCE_INLINE static HemisphericDistances<FP, level> _getDistances(vct thetaAOne, vct phiAOne, vct thetaBOne, vct phiBOne,
																	  vct thetaATwo, vct phiATwo, vct thetaBTwo, vct phiBTwo)
	{
		return rivoli::hemisphericDistance2Sx2SBilateral(thetaAOne, phiAOne, thetaBOne, phiBOne,
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

	// NB: because of its reciprocal nature combined to a spherical parameterization,
	// the two hemispheres of this topology are strongly coupled, thus we should not
	// implement _getDistances.
	static constexpr bool hasDecomposedDistance = false;

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

	// NB: because of its reciprocal nature combined to a spherical parameterization,
	// the two hemispheres of this topology are strongly coupled, thus we should not
	// implement _getDistances.
	static constexpr bool hasDecomposedDistance = false;

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

	// This topology can be computed with its components solved separately
	// This is particularly useful for the use of some anisotropic kernels
	FORCE_INLINE static HemisphericDistances<FP, level> _getDistances(vct thetaAOne, vct phiAOne, vct thetaBOne, vct phiBOne,
																	  vct thetaATwo, vct phiATwo, vct thetaBTwo, vct phiBTwo)
	{
		return rivoli::hemisphericDistance2Sx2SReciprocalRusinkiewicz(thetaAOne, phiAOne, thetaBOne, phiBOne,
										                              thetaATwo, phiATwo, thetaBTwo, phiBTwo);
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

	// This topology can be computed with its components solved separately
	// This is particularly useful for the use of some anisotropic kernels
	FORCE_INLINE static HemisphericDistances<FP, level> _getDistances(vct thetaAOne, vct phiAOne, vct thetaBOne, vct phiBOne,
																	  vct thetaATwo, vct phiATwo, vct thetaBTwo, vct phiBTwo)
	{
		return rivoli::hemisphericDistance2Sx2SBilateralReciprocalRusinkiewicz(thetaAOne, phiAOne, thetaBOne, phiBOne,
										                                       thetaATwo, phiATwo, thetaBTwo, phiBTwo);
	}
};

}
