#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>

#include "coordinate.hpp"
#include "logger.hpp"


// Defines macro for forcing inlining based on the compiler
// This will be used for some of the distance functions. It
// will allow better performance (see the details in header
// docstring of the Topology3DSph::getDistance functions)
#if defined(_MSC_VER)
  #define FORCE_INLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
  #define FORCE_INLINE inline __attribute__((always_inline))
#else
  #define FORCE_INLINE inline
#endif


/*
 * @brief Computes the haversine of an angle
 *        Haversine is the square of the sine of half the angle
 *
 * @tparam FloatingPrecision FP precision. Usually float or double
 *
 * @param arg Angle, given in radians
 * 
 * @details The haversine function is called a lot in the code and
 *          should be very optimized. After profiling, we saw that
 *			the current implementation is as fast others including
 *			different types of optimizations.
 *			If one wants to try optimizing more we strongly invite
 *			you to profile the code before and after.
 * @note    If the function doesn't appear during profiling it may
 *          be because it is inlined. In this case refuse inlining
 *          with the keyword __declspec(noinline) for MSVC or with
 *          [[gnu::noinline]] or __attribute__((noinline)) for GCC
 *
 * @return haversine of the angle, without units
 */
template <typename FloatingPrecision>
FloatingPrecision haversine(const FloatingPrecision& arg) {
	return glm::sin(arg / 2) * glm::sin(arg / 2);
}

/*
 * @brief Computes the great circle distance between two points on a sphere
 *
 * @note Coordinates should be given in spherical coordinates following the
		 convention of physicists ISO 80000-2:2019, i.e.:
 *
 * @tparam FloatingPrecision FP precision. Usually float or double
 *
 * @param thetaOne polar angle, or colatitude angle, of first  point in rad
 * @param phiOne   azimuthal angle (longitude angle) of first  point in rad
 * @param thetaTwo polar angle, or colatitude angle  of second point in rad
 * @param phiTwo   azimuthal angle (longitude angle) of second point in rad
 *
 * @return great circle distance on a unit sphere with radius 1
 */
template <typename FloatingPrecision>
FloatingPrecision greatCircleDistance(const FloatingPrecision& thetaOne, const FloatingPrecision& phiOne,
								      const FloatingPrecision& thetaTwo, const FloatingPrecision& phiTwo) {
	return static_cast<FloatingPrecision>(2) * glm::asin(glm::sqrt(haversine(thetaTwo - thetaOne) + glm::sin(thetaTwo) * glm::sin(thetaOne) * haversine(phiTwo - phiOne)));
}

template <typename FloatingPrecision>
class Topology {
public:
	virtual ~Topology() = default;
	virtual std::unique_ptr<Topology> clone() const = 0; // Ajout de la méthode clone

	virtual FloatingPrecision getDistance(const Coordinate& coordinateOne, 
		                                  const Coordinate& coordinateTwo) const = 0;

	virtual size_t getDimension() const = 0;

};

template <typename FloatingPrecision>
class Topology2D : public Topology<FloatingPrecision> {
public:
	Topology2D() {}

	std::unique_ptr<Topology<FloatingPrecision>> clone() const override {
		return std::make_unique<Topology2D>(*this);
	}

	FloatingPrecision getDistance(const Coordinate& coordinateOne,
		                          const Coordinate& coordinateTwo) const override {
		try {
			// The two given coordinates are of type Coordinate2D
			const auto& coordinateOne2D = dynamic_cast<const Coordinate2D<FloatingPrecision>&>(coordinateOne);
			const auto& coordinateTwo2D = dynamic_cast<const Coordinate2D<FloatingPrecision>&>(coordinateTwo);
			return getDistance(coordinateOne2D, coordinateTwo2D);
		}
		catch (const std::bad_cast&) {
			try {
				// First coordinate is of type GrazingCoordinate, second is Coordinate2D
				const auto& coordinateOneGrazing = dynamic_cast<const GrazingCoordinate&>(coordinateOne);
				const auto& coordinateTwo2D      = dynamic_cast<const Coordinate2D<FloatingPrecision>&>(coordinateTwo);
				return getDistance(coordinateOneGrazing, coordinateTwo2D);
			}
			catch (const std::bad_cast&) {
				// First coordinate is of type Coordinate2D, second is GrazingCoordinate
				const auto& coordinateOne2D      = dynamic_cast<const Coordinate2D<FloatingPrecision>&>(coordinateOne);
				const auto& coordinateTwoGrazing = dynamic_cast<const GrazingCoordinate&>(coordinateTwo);
				return getDistance(coordinateOne2D, coordinateTwoGrazing);
			}
		}
	}

	FloatingPrecision getDistance(const Coordinate2D<FloatingPrecision>& coordinateOne,
								  const GrazingCoordinate&               coordinateTwoGrazing) const {
		return getDistance(coordinateTwoGrazing, coordinateOne);
	}

	FloatingPrecision getDistance(const GrazingCoordinate&               coordinateOneGrazing,
								  const Coordinate2D<FloatingPrecision>& coordinateTwo) const {
		if (coordinateOneGrazing.getHemisphere() != 1)
			LOG_CRITICAL("Grazing coordinate can be set only for hemisphere 1 in 2D topology"
				         ". Selected hemisphere is: ", coordinateOneGrazing.getHemisphere());

		FloatingPrecision theta = glm::half_pi<FloatingPrecision>();
		FloatingPrecision phi   = coordinateTwo.getPhi();

		Coordinate2D<FloatingPrecision> coordinateTwoProjection = Coordinate2D<FloatingPrecision>(theta, phi);

		return getDistance(coordinateTwoProjection, coordinateTwo);
	}

	FloatingPrecision getDistance(const Coordinate2D<FloatingPrecision>& coordinateOne,
		                          const Coordinate2D<FloatingPrecision>& coordinateTwo) const {
		return greatCircleDistance(coordinateOne.getTheta(), coordinateOne.getPhi(),
								   coordinateTwo.getTheta(), coordinateTwo.getPhi());
	}

	size_t getDimension() const override { return 2; }

};

template <typename FloatingPrecision>
class Topology2DSym : public Topology2D<FloatingPrecision> {
public:
	Topology2DSym() {}

	std::unique_ptr<Topology<FloatingPrecision>> clone() const override {
		return std::make_unique<Topology2DSym>(*this);
	}

	FloatingPrecision getDistance(const Coordinate& coordinateOne,
		                          const Coordinate& coordinateTwo) const override {
		try {
			// The two given coordinates are of type Coordinate2D
			const auto& coordinateOne2D = dynamic_cast<const Coordinate2D<FloatingPrecision>&>(coordinateOne);
			const auto& coordinateTwo2D = dynamic_cast<const Coordinate2D<FloatingPrecision>&>(coordinateTwo);
			return getDistance(coordinateOne2D, coordinateTwo2D);
		}
		catch (const std::bad_cast&) {
			try {
				// First coordinate is of type GrazingCoordinate, second is Coordinate2D
				const auto& coordinateOneGrazing = dynamic_cast<const GrazingCoordinate&>(coordinateOne);
				const auto& coordinateTwo2D      = dynamic_cast<const Coordinate2D<FloatingPrecision>&>(coordinateTwo);
				return getDistance(coordinateOneGrazing, coordinateTwo2D);
			}
			catch (const std::bad_cast&) {
				// First coordinate is of type Coordinate2D, second is GrazingCoordinate
				const auto& coordinateOne2D      = dynamic_cast<const Coordinate2D<FloatingPrecision>&>(coordinateOne);
				const auto& coordinateTwoGrazing = dynamic_cast<const GrazingCoordinate&>(coordinateTwo);
				return getDistance(coordinateOne2D, coordinateTwoGrazing);
			}
		}
	}

	FloatingPrecision getDistance(const Coordinate2D<FloatingPrecision>& coordinateOne,
		                          const GrazingCoordinate&               coordinateTwoGrazing) const {
		return getDistance(coordinateTwoGrazing, coordinateOne);
	}

	FloatingPrecision getDistance(const GrazingCoordinate&               coordinateOneGrazing,
		                          const Coordinate2D<FloatingPrecision>& coordinateTwo) const {
		if (coordinateOneGrazing.getHemisphere() != 1)
			LOG_CRITICAL("Grazing coordinate can be set only for hemisphere 1 in 2D symmetrical "
				         "topology. Selected hemisphere is: ", coordinateOneGrazing.getHemisphere());

		FloatingPrecision theta = glm::half_pi<FloatingPrecision>();
		FloatingPrecision phi   = coordinateTwo.getPhi();

		Coordinate2D<FloatingPrecision> coordinateTwoProjection = Coordinate2D<FloatingPrecision>(theta, phi);

		return getDistance(coordinateTwoProjection, coordinateTwo);
	}

	/*
	 * @details Even if the main part of the cost of computation comes from the execution of the
	 *          getDistance functions from Topology2D, an important part of the computation time
	 *          came from the way that path lengths were stored and how the minimum was returned
	 *          We were using a std::vector and std::min_element to get the minimum value.
	 *          For this function and the others equivalents, the following implementation using
	 *			FloatingPrecision and ternary operators is much faster, and should almost always
	 *			be preferred.
	 */
	FloatingPrecision getDistance(const Coordinate2D<FloatingPrecision>& coordinateOne,
								  const Coordinate2D<FloatingPrecision>& coordinateTwo) const {
		FloatingPrecision distanceOne, distanceTwo; // Contains all possible paths

		// Only two possibilities: others are the same by symmetry of the metric
		distanceOne = Topology2D<FloatingPrecision>::getDistance(coordinateOne, coordinateTwo);
		distanceTwo = Topology2D<FloatingPrecision>::getDistance(coordinateOne, coordinateTwo.getBilateralSymmetrical());

		// Riemannian distance is the geodesic, i.e. infimum of allpaths
		return (distanceOne < distanceTwo) ? distanceOne : distanceTwo;
	}
};

template <typename FloatingPrecision>
class Topology3DSph : public Topology<FloatingPrecision> {
public:
	Topology3DSph() {}

	std::unique_ptr<Topology<FloatingPrecision>> clone() const override {
		return std::make_unique<Topology3DSph>(*this);
	}

	/*
	 * @details Forcing inlining has given good results during profiling (approx. + 5% performance) with
		 /!\    a computation time cost negligeable, and a weight increase of 6 kB (< 1% of the complete
				binary weight) (Values obtained with MSVC 17.13.6). This has been tested on all distance
				functions, and better performance results only for this level.
	 */
	FORCE_INLINE FloatingPrecision getDistance(const Coordinate& coordinateOne,
											   const Coordinate& coordinateTwo) const {
		try {
			// The two given coordinates are of type Coordinate3DSpherical
			const auto& coordinateOne3DSph = dynamic_cast<const Coordinate3DSpherical<FloatingPrecision>&>(coordinateOne);
			const auto& coordinateTwo3DSph = dynamic_cast<const Coordinate3DSpherical<FloatingPrecision>&>(coordinateTwo);
			return getDistance(coordinateOne3DSph, coordinateTwo3DSph);
		}
		catch (const std::bad_cast&) {
			try {
				// First coordinate is of type GrazingCoordinate, second is Coordinate3DSpherical
				const auto& coordinateOneGrazing = dynamic_cast<const GrazingCoordinate&>(coordinateOne);
				const auto& coordinateTwo3DSph   = dynamic_cast<const Coordinate3DSpherical<FloatingPrecision>&>(coordinateTwo);
				return getDistance(coordinateOneGrazing, coordinateTwo3DSph);
			}
			catch (const std::bad_cast&) {
				// First coordinate is of type Coordinate3DSpherical, second is GrazingCoordinate
				const auto& coordinateOne3DSph   = dynamic_cast<const Coordinate3DSpherical<FloatingPrecision>&>(coordinateOne);
				const auto& coordinateTwoGrazing = dynamic_cast<const GrazingCoordinate&>(coordinateTwo);
				return getDistance(coordinateOne3DSph, coordinateTwoGrazing);
			}
		}
	}

	/*
	 * @details Forcing inlining has given good results during profiling (approx. + 5% performance) with
		 /!\    a computation time cost negligeable, and a weight increase of 6 kB (< 1% of the complete
				binary weight) (Values obtained with MSVC 17.13.6). This has been tested on all distance
				functions, and better performance results only for this level.
	 */
	FORCE_INLINE FloatingPrecision getDistance(const Coordinate3DSpherical<FloatingPrecision>& coordinateOne,
											   const GrazingCoordinate&                        coordinateTwoGrazing) const {
		return getDistance(coordinateTwoGrazing, coordinateOne);
	}

	/*
	 * @details Forcing inlining has given good results during profiling (approx. + 5% performance) with
		 /!\    a computation time cost negligeable, and a weight increase of 6 kB (< 1% of the complete
				binary weight) (Values obtained with MSVC 17.13.6). This has been tested on all distance
				functions, and better performance results only for this level.
	 */
	FORCE_INLINE FloatingPrecision getDistance(const GrazingCoordinate&                        coordinateOneGrazing,
											   const Coordinate3DSpherical<FloatingPrecision>& coordinateTwo) const {
		if (coordinateOneGrazing.getHemisphere() == 1) {
			FloatingPrecision thetaI   = glm::half_pi<FloatingPrecision>();
			FloatingPrecision thetaO   = coordinateTwo.getThetaO();
			FloatingPrecision deltaPhi = coordinateTwo.getDeltaPhi();

			Coordinate3DSpherical<FloatingPrecision> coordinateTwoProjection = Coordinate3DSpherical<FloatingPrecision>(thetaI, thetaO, deltaPhi);

			return getDistance(coordinateTwoProjection, coordinateTwo);
		}
		else if (coordinateOneGrazing.getHemisphere() == 2) {
			FloatingPrecision thetaI   = coordinateTwo.getThetaI();
			FloatingPrecision thetaO   = glm::half_pi<FloatingPrecision>();
			FloatingPrecision deltaPhi = coordinateTwo.getDeltaPhi();

			Coordinate3DSpherical<FloatingPrecision> coordinateTwoProjection = Coordinate3DSpherical<FloatingPrecision>(thetaI, thetaO, deltaPhi);

			return getDistance(coordinateTwoProjection, coordinateTwo);
		}
		else
			LOG_CRITICAL("Grazing coordinate can be set only for hemisphere 1 or 2 in 3D topology"
				         ". Selected hemisphere is: ", coordinateOneGrazing.getHemisphere());
	}

	/*
	 * @details Although very simple, this function and the ones called inside (Topology2D::getDistance)
				are being the most called in the whole program, and are responsible for more than > 50 %
				of the total computation time. This version has been profiled and optimizes computation.
				We strongly recommend to do a profiling comparison if any modification is made.
		 /!\    Forcing inlining has given good results during profiling (approx. + 5% performance) with
		        a compilation time cost negligeable, and a weight increase of 6 kB (< 1% of the complete
				binary weight) (Values obtained with MSVC 17.13.6). This has been tested on all distance
				functions, and better performance results only for this one.
	 */
	FORCE_INLINE FloatingPrecision getDistanceSquared(const Coordinate3DSpherical<FloatingPrecision>& coordinateOne,
													  const Coordinate3DSpherical<FloatingPrecision>& coordinateTwo) const {
		Topology2D<FloatingPrecision>   topology2D;
		Coordinate2D<FloatingPrecision> coordinateOneOmegaO = Coordinate2D<FloatingPrecision>(coordinateOne.getThetaO(), coordinateOne.getDeltaPhi());
		Coordinate2D<FloatingPrecision> coordinateTwoOmegaO = Coordinate2D<FloatingPrecision>(coordinateTwo.getThetaO(), coordinateTwo.getDeltaPhi());		

		// In 3D BRDF, the first hemisphere distance is just  the difference between the two thetaI
		FloatingPrecision distanceHemisphereOne = coordinateTwo.getThetaI() - coordinateOne.getThetaI();
		// The second hemisphere distance is the distance between the two coordinates in the 2D topology
		FloatingPrecision distanceHemisphereTwo = topology2D.getDistance(coordinateOneOmegaO, coordinateTwoOmegaO);

		// Since we are using here an Euclidean metric, the distances need to be squared
		distanceHemisphereOne *= distanceHemisphereOne;
		distanceHemisphereTwo *= distanceHemisphereTwo;

		// Even if the Euclidean distance uses a square root, we only work with squared distances
		// Since different distances will be compared and only the smallest will be kept, calling
		// glm::sqrt only at the end and working with squared distances is more efficient.
		return distanceHemisphereOne + distanceHemisphereTwo;
	}

	FORCE_INLINE FloatingPrecision getDistance(const Coordinate3DSpherical<FloatingPrecision>& coordinateOne,
											   const Coordinate3DSpherical<FloatingPrecision>& coordinateTwo) const {
		return glm::sqrt(getDistanceSquared(coordinateOne, coordinateTwo));
	}

	size_t getDimension() const override { return 3; }
};

template <typename FloatingPrecision>
class Topology3DSphRec : public Topology3DSph<FloatingPrecision> {
public:
	Topology3DSphRec() {}

	std::unique_ptr<Topology<FloatingPrecision>> clone() const override {
		return std::make_unique<Topology3DSphRec>(*this);
	}

	FloatingPrecision getDistance(const Coordinate& coordinateOne,
		                          const Coordinate& coordinateTwo) const {
		try {
			// The two given coordinates are of type Coordinate3DSpherical
			const auto& coordinateOne3DSph = dynamic_cast<const Coordinate3DSpherical<FloatingPrecision>&>(coordinateOne);
			const auto& coordinateTwo3DSph = dynamic_cast<const Coordinate3DSpherical<FloatingPrecision>&>(coordinateTwo);
			return getDistance(coordinateOne3DSph, coordinateTwo3DSph);
		}
		catch (const std::bad_cast&) {
			try {
				// First coordinate is of type GrazingCoordinate, second is Coordinate3DSpherical
				const auto& coordinateOneGrazing = dynamic_cast<const GrazingCoordinate&>(coordinateOne);
				const auto& coordinateTwo3DSph   = dynamic_cast<const Coordinate3DSpherical<FloatingPrecision>&>(coordinateTwo);
				return getDistance(coordinateOneGrazing, coordinateTwo3DSph);
			}
			catch (const std::bad_cast&) {
				// First coordinate is of type Coordinate3DSpherical, second is GrazingCoordinate
				const auto& coordinateOne3DSph   = dynamic_cast<const Coordinate3DSpherical<FloatingPrecision>&>(coordinateOne);
				const auto& coordinateTwoGrazing = dynamic_cast<const GrazingCoordinate&>(coordinateTwo);
				return getDistance(coordinateOne3DSph, coordinateTwoGrazing);
			}
		}
	}

	FloatingPrecision getDistance(const Coordinate3DSpherical<FloatingPrecision>& coordinateOne,
		                          const GrazingCoordinate&                        coordinateTwoGrazing) const {
		return getDistance(coordinateTwoGrazing, coordinateOne);
	}

	FloatingPrecision getDistance(const GrazingCoordinate&                        coordinateOneGrazing,
		                          const Coordinate3DSpherical<FloatingPrecision>& coordinateTwo) const {
		if (coordinateOneGrazing.getHemisphere() == 1) {
			FloatingPrecision thetaI   = glm::half_pi<FloatingPrecision>();
			FloatingPrecision thetaO   = coordinateTwo.getThetaO();
			FloatingPrecision deltaPhi = coordinateTwo.getDeltaPhi();

			Coordinate3DSpherical<FloatingPrecision> coordinateTwoProjection = Coordinate3DSpherical<FloatingPrecision>(thetaI, thetaO, deltaPhi);

			return getDistance(coordinateTwoProjection, coordinateTwo);
		}
		else if (coordinateOneGrazing.getHemisphere() == 2) {
			FloatingPrecision thetaI   = coordinateTwo.getThetaI();
			FloatingPrecision thetaO   = glm::half_pi<FloatingPrecision>();
			FloatingPrecision deltaPhi = coordinateTwo.getDeltaPhi();

			Coordinate3DSpherical<FloatingPrecision> coordinateTwoProjection = Coordinate3DSpherical<FloatingPrecision>(thetaI, thetaO, deltaPhi);

			return getDistance(coordinateTwoProjection, coordinateTwo);
		}
		else
			LOG_CRITICAL("Grazing coordinate can be set only for hemisphere 1 or 2 in 3D topology"
				         ". Selected hemisphere is: ", coordinateOneGrazing.getHemisphere());
	}

	/*
	 * @details Even if the main part of the cost of computation comes from the execution of the
	 *          getDistance functions from Topology3DSph, an important part (> 30%) of this cost
	 *          came from the way that path lengths were stored and how the minimum was returned
	 *          We were using a std::vector and std::min_element to get the minimum value. 
	 *          For this function and the others equivalents, the following implementation using
	 *			FloatingPrecision and ternary operators is much faster, and should almost always
	 *			be preferred.
	 */
	FloatingPrecision getDistanceSquared(const Coordinate3DSpherical<FloatingPrecision>& coordinateOne,
		                                 const Coordinate3DSpherical<FloatingPrecision>& coordinateTwo) const {
		FloatingPrecision distanceOneSquared, distanceTwoSquared, distanceThreeSquared, distanceFourSquared; // Contains all possible paths

		Coordinate3DSpherical<FloatingPrecision> coordinateOneReciprocal = coordinateOne.getReciprocal();
		Coordinate3DSpherical<FloatingPrecision> coordinateTwoReciprocal = coordinateTwo.getReciprocal();

		// In order to optimize the computation, it is useless to call several times the glm::sqrt function.
		// Only once we have obtained the minimum distance length, it makes sense to compute its square root
		distanceOneSquared   = Topology3DSph<FloatingPrecision>::getDistanceSquared(coordinateOne, coordinateTwo);
		distanceTwoSquared   = Topology3DSph<FloatingPrecision>::getDistanceSquared(coordinateOne, coordinateTwoReciprocal);
		distanceThreeSquared = Topology3DSph<FloatingPrecision>::getDistanceSquared(coordinateOneReciprocal, coordinateTwo);
		distanceFourSquared  = Topology3DSph<FloatingPrecision>::getDistanceSquared(coordinateOneReciprocal, coordinateTwoReciprocal);

		// Riemannian distance is the geodesic, i.e. infimum of all paths
		// NB : since this function is on the hot path of the program, it should be optimised as
		// much as possible. Using std::min with four arguments costs too much because lists are
		// initialised in background. Using std::min in cascade is better, and could achieve the
		// same result as the following, depending on the compiler optimisations. This solution,
		// however, is the fastest in any cases.
		FloatingPrecision minOne = (distanceOneSquared   < distanceTwoSquared ) ? distanceOneSquared   : distanceTwoSquared;
		FloatingPrecision minTwo = (distanceThreeSquared < distanceFourSquared) ? distanceThreeSquared : distanceFourSquared;
		return (minOne < minTwo) ? minOne : minTwo;
	}

	FloatingPrecision getDistance(const Coordinate3DSpherical<FloatingPrecision>& coordinateOne,
								  const Coordinate3DSpherical<FloatingPrecision>& coordinateTwo) const {
		return glm::sqrt(getDistanceSquared(coordinateOne, coordinateTwo));
	}

};

template <typename FloatingPrecision>
class Topology3DSphSym : public Topology3DSph<FloatingPrecision> {
public:
	Topology3DSphSym() {}

	std::unique_ptr<Topology<FloatingPrecision>> clone() const override {
		return std::make_unique<Topology3DSphSym>(*this);
	}

	FloatingPrecision getDistance(const Coordinate& coordinateOne,
		                          const Coordinate& coordinateTwo) const {
		try {
			// The two given coordinates are of type Coordinate3DSpherical
			const auto& coordinateOne3DSph = dynamic_cast<const Coordinate3DSpherical<FloatingPrecision>&>(coordinateOne);
			const auto& coordinateTwo3DSph = dynamic_cast<const Coordinate3DSpherical<FloatingPrecision>&>(coordinateTwo);
			return getDistance(coordinateOne3DSph, coordinateTwo3DSph);
		}
		catch (const std::bad_cast&) {
			try {
				// First coordinate is of type GrazingCoordinate, second is Coordinate3DSpherical
				const auto& coordinateOneGrazing = dynamic_cast<const GrazingCoordinate&>(coordinateOne);
				const auto& coordinateTwo3DSph   = dynamic_cast<const Coordinate3DSpherical<FloatingPrecision>&>(coordinateTwo);
				return getDistance(coordinateOneGrazing, coordinateTwo3DSph);
			}
			catch (const std::bad_cast&) {
				// First coordinate is of type Coordinate3DSpherical, second is GrazingCoordinate
				const auto& coordinateOne3DSph   = dynamic_cast<const Coordinate3DSpherical<FloatingPrecision>&>(coordinateOne);
				const auto& coordinateTwoGrazing = dynamic_cast<const GrazingCoordinate&>(coordinateTwo);
				return getDistance(coordinateOne3DSph, coordinateTwoGrazing);
			}
		}
	}

	FloatingPrecision getDistance(const Coordinate3DSpherical<FloatingPrecision>& coordinateOne,
		                          const GrazingCoordinate&                        coordinateTwoGrazing) const {
		return getDistance(coordinateTwoGrazing, coordinateOne);
	}

	FloatingPrecision getDistance(const GrazingCoordinate&                        coordinateOneGrazing,
		                          const Coordinate3DSpherical<FloatingPrecision>& coordinateTwo) const {
		if (coordinateOneGrazing.getHemisphere() == 1) {
			FloatingPrecision thetaI   = glm::half_pi<FloatingPrecision>();
			FloatingPrecision thetaO   = coordinateTwo.getThetaO();
			FloatingPrecision deltaPhi = coordinateTwo.getDeltaPhi();

			Coordinate3DSpherical<FloatingPrecision> coordinateTwoProjection = Coordinate3DSpherical<FloatingPrecision>(thetaI, thetaO, deltaPhi);

			return getDistance(coordinateTwoProjection, coordinateTwo);
		}
		else if (coordinateOneGrazing.getHemisphere() == 2) {
			FloatingPrecision thetaI   = coordinateTwo.getThetaI();
			FloatingPrecision thetaO   = glm::half_pi<FloatingPrecision>();
			FloatingPrecision deltaPhi = coordinateTwo.getDeltaPhi();

			Coordinate3DSpherical<FloatingPrecision> coordinateTwoProjection = Coordinate3DSpherical<FloatingPrecision>(thetaI, thetaO, deltaPhi);

			return getDistance(coordinateTwoProjection, coordinateTwo);
		}
		else
			LOG_CRITICAL("Grazing coordinate can be set only for hemisphere 1 or 2 in 3D topology"
				         ". Selected hemisphere is: ", coordinateOneGrazing.getHemisphere());
	}

	/*
	 * @details Even if the main part of the cost of computation comes from the execution of the
	 *          getDistance functions from Topology3DSph, an important part (> 40%) of this cost
	 *          came from the way that path lengths were stored and how the minimum was returned
	 *          We were using a std::vector and std::min_element to get the minimum value.
	 *          For this function and the others equivalents, the following implementation using
	 *			FloatingPrecision and ternary operators is much faster, and should almost always
	 *			be preferred.
	 */
	FloatingPrecision getDistance(const Coordinate3DSpherical<FloatingPrecision>& coordinateOne,
								  const Coordinate3DSpherical<FloatingPrecision>& coordinateTwo) const {
		FloatingPrecision distanceOneSquared, distanceTwoSquared; // Contains all possible paths

		// In order to optimize the computation, it is useless to call several times the glm::sqrt function.
		// Only once we have obtained the minimum distance length, it makes sense to compute its square root
		distanceOneSquared = Topology3DSph<FloatingPrecision>::getDistanceSquared(coordinateOne, coordinateTwo);
		distanceTwoSquared = Topology3DSph<FloatingPrecision>::getDistanceSquared(coordinateOne, coordinateTwo.getBilateralSymmetrical());

		// Riemannian distance is the geodesic, i.e. infimum of allpaths
		FloatingPrecision minimumDistanceSquared = (distanceOneSquared < distanceTwoSquared) ? distanceOneSquared : distanceTwoSquared;
		return glm::sqrt(minimumDistanceSquared);
	}

};

template <typename FloatingPrecision>
class Topology3DSphRecSym : public Topology3DSphRec<FloatingPrecision> {
public:
	Topology3DSphRecSym() {}

	std::unique_ptr<Topology<FloatingPrecision>> clone() const override {
		return std::make_unique<Topology3DSphRecSym>(*this);
	}

	FloatingPrecision getDistance(const Coordinate& coordinateOne,
		                          const Coordinate& coordinateTwo) const {
		try {
			// The two given coordinates are of type Coordinate3DSpherical
			const auto& coordinateOne3DSph = dynamic_cast<const Coordinate3DSpherical<FloatingPrecision>&>(coordinateOne);
			const auto& coordinateTwo3DSph = dynamic_cast<const Coordinate3DSpherical<FloatingPrecision>&>(coordinateTwo);
			return getDistance(coordinateOne3DSph, coordinateTwo3DSph);
		}
		catch (const std::bad_cast&) {
			try {
				// First coordinate is of type GrazingCoordinate, second is Coordinate3DSpherical
				const auto& coordinateOneGrazing = dynamic_cast<const GrazingCoordinate&>(coordinateOne);
				const auto& coordinateTwo3DSph   = dynamic_cast<const Coordinate3DSpherical<FloatingPrecision>&>(coordinateTwo);
				return getDistance(coordinateOneGrazing, coordinateTwo3DSph);
			}
			catch (const std::bad_cast&) {
				// First coordinate is of type Coordinate3DSpherical, second is GrazingCoordinate
				const auto& coordinateOne3DSph   = dynamic_cast<const Coordinate3DSpherical<FloatingPrecision>&>(coordinateOne);
				const auto& coordinateTwoGrazing = dynamic_cast<const GrazingCoordinate&>(coordinateTwo);
				return getDistance(coordinateOne3DSph, coordinateTwoGrazing);
			}
		}
	}

	FloatingPrecision getDistance(const Coordinate3DSpherical<FloatingPrecision>& coordinateOne,
		                          const GrazingCoordinate&                        coordinateTwoGrazing) const {
		return getDistance(coordinateTwoGrazing, coordinateOne);
	}

	FloatingPrecision getDistance(const GrazingCoordinate&                        coordinateOneGrazing,
		                          const Coordinate3DSpherical<FloatingPrecision>& coordinateTwo) const {
		if (coordinateOneGrazing.getHemisphere() == 1) {
			FloatingPrecision thetaI   = glm::half_pi<FloatingPrecision>();
			FloatingPrecision thetaO   = coordinateTwo.getThetaO();
			FloatingPrecision deltaPhi = coordinateTwo.getDeltaPhi();

			Coordinate3DSpherical<FloatingPrecision> coordinateTwoProjection = Coordinate3DSpherical<FloatingPrecision>(thetaI, thetaO, deltaPhi);

			return getDistance(coordinateTwoProjection, coordinateTwo);
		}
		else if (coordinateOneGrazing.getHemisphere() == 2) {
			FloatingPrecision thetaI   = coordinateTwo.getThetaI();
			FloatingPrecision thetaO   = glm::half_pi<FloatingPrecision>();
			FloatingPrecision deltaPhi = coordinateTwo.getDeltaPhi();

			Coordinate3DSpherical<FloatingPrecision> coordinateTwoProjection = Coordinate3DSpherical<FloatingPrecision>(thetaI, thetaO, deltaPhi);

			return getDistance(coordinateTwoProjection, coordinateTwo);
		}
		else
			LOG_CRITICAL("Grazing coordinate can be set only for hemisphere 1 or 2 in 3D topology"
				         ". Selected hemisphere is: ", coordinateOneGrazing.getHemisphere());
	}

	/*
	 * @details Even if the main part of the cost of computation comes from the execution of the
	 *          getDistance functions from Topology3DSph, an important part (> 30%) of this cost
	 *          came from the way that path lengths were stored and how the minimum was returned
	 *          We were using a std::vector and std::min_element to get the minimum value.
	 *          For this function and the others equivalents, the following implementation using
	 *			FloatingPrecision and ternary operators is much faster, and should almost always
	 *			be preferred.
	 */
	FloatingPrecision getDistance(const Coordinate3DSpherical<FloatingPrecision>& coordinateOne,
								  const Coordinate3DSpherical<FloatingPrecision>& coordinateTwo) const {
		FloatingPrecision distanceOneSquared, distanceTwoSquared; // Contains all possible paths

		// In order to optimize the computation, it is useless to call several times the glm::sqrt function.
		// Only once we have obtained the minimum distance length, it makes sense to compute its square root
		distanceOneSquared = Topology3DSphRec<FloatingPrecision>::getDistanceSquared(coordinateOne, coordinateTwo);
		distanceTwoSquared = Topology3DSphRec<FloatingPrecision>::getDistanceSquared(coordinateOne, coordinateTwo.getBilateralSymmetrical());

		// Riemannian distance is the geodesic, i.e. infimum of allpaths
		FloatingPrecision minimumDistanceSquared = (distanceOneSquared < distanceTwoSquared) ? distanceOneSquared : distanceTwoSquared;
		return glm::sqrt(minimumDistanceSquared);
	}

};