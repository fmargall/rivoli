#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>

#include "coordinate.hpp"
#include "logger.hpp"

/*
 * @brief Computes the haversine of an angle
 *        Haversine is the square of the sine of half the angle
 *
 * @tparam FloatingPrecision FP precision. Usually float or double
 *
 * @param arg Angle, given in radians
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
	virtual FloatingPrecision getDistance(const Coordinate& coordinateOne, 
		                                  const Coordinate& coordinateTwo) const = 0;

};

template <typename FloatingPrecision>
class Topology2D : public Topology<FloatingPrecision> {
public:
	Topology2D() {}

	FloatingPrecision getDistance(const Coordinate& coordinateOne,
								  const Coordinate& coordinateTwo) const override {
		const auto& coordinateOne2D = dynamic_cast<const Coordinate2D<FloatingPrecision>&>(coordinateOne);
		const auto& coordinateTwo2D = dynamic_cast<const Coordinate2D<FloatingPrecision>&>(coordinateTwo);
		return getDistance(coordinateOne2D, coordinateTwo2D);;
	}

	FloatingPrecision getDistance(const Coordinate2D<FloatingPrecision>& coordinateOne, 
		                          const Coordinate2D<FloatingPrecision>& coordinateTwo) const {
		return greatCircleDistance(coordinateOne.getTheta(), coordinateOne.getPhi(),
								   coordinateTwo.getTheta(), coordinateTwo.getPhi());
	}

};

template <typename FloatingPrecision>
class Topology2DSym : public Topology2D<FloatingPrecision> {
public:
	Topology2DSym() {}

	FloatingPrecision getDistance(const Coordinate& coordinateOne, 
		                          const Coordinate& coordinateTwo) const {
		const auto& coordinateOne2D = dynamic_cast<const Coordinate2D<FloatingPrecision>&>(coordinateOne);
		const auto& coordinateTwo2D = dynamic_cast<const Coordinate2D<FloatingPrecision>&>(coordinateTwo);
		return getDistance(coordinateOne2D, coordinateTwo2D);
	}

	FloatingPrecision getDistance(const Coordinate2D<FloatingPrecision>& coordinateOne,
								  const Coordinate2D<FloatingPrecision>& coordinateTwo) const {
		std::vector<FloatingPrecision> pathsLengths(2); // Contains all possible paths

		// Only two possibilities: others are the same by symmetry of the metric
		pathsLengths[0] = Topology2D<FloatingPrecision>::getDistance(coordinateOne, coordinateTwo);
		pathsLengths[1] = Topology2D<FloatingPrecision>::getDistance(coordinateOne, coordinateTwo.getBilateralSymmetrical());

		// Riemannian distance is the geodesic, i.e. infimum of allpaths
		return *std::min_element(pathsLengths.begin(), pathsLengths.end());
	}
};

template <typename FloatingPrecision>
class Topology3DSph : public Topology<FloatingPrecision> {
public:
	Topology3DSph() {}

	FloatingPrecision getDistance(const Coordinate& coordinateOne, 
		                          const Coordinate& coordinateTwo) const {
		const auto& coordinateOne3DSph = dynamic_cast<const Coordinate3DSpherical<FloatingPrecision>&>(coordinateOne);
		const auto& coordinateTwo3DSph = dynamic_cast<const Coordinate3DSpherical<FloatingPrecision>&>(coordinateTwo);
		return getDistance(coordinateOne3DSph, coordinateTwo3DSph);
	}

	FloatingPrecision getDistance(const Coordinate3DSpherical<FloatingPrecision>& coordinateOne,
							      const Coordinate3DSpherical<FloatingPrecision>& coordinateTwo) const {
		Topology2D<FloatingPrecision> topology2D;
		Coordinate2D<FloatingPrecision> coordinateOneOmegaO = Coordinate2D<FloatingPrecision>(coordinateOne.getThetaO(), coordinateOne.getDeltaPhi());
		Coordinate2D<FloatingPrecision> coordinateTwoOmegaO = Coordinate2D<FloatingPrecision>(coordinateTwo.getThetaO(), coordinateTwo.getDeltaPhi());		

		return glm::sqrt(glm::pow(coordinateTwo.getThetaI() - coordinateOne.getThetaI(), static_cast<FloatingPrecision>(2)) +
						 glm::pow(topology2D.getDistance(coordinateOneOmegaO,
											             coordinateTwoOmegaO), static_cast<FloatingPrecision>(2)));
	}
};

template <typename FloatingPrecision>
class Topology3DSphRec : public Topology3DSph<FloatingPrecision> {
public:
	Topology3DSphRec() {}

	FloatingPrecision getDistance(const Coordinate& coordinateOne,
								  const Coordinate& coordinateTwo) const {
		const auto& coordinateOne3DSph = dynamic_cast<const Coordinate3DSpherical<FloatingPrecision>&>(coordinateOne);
		const auto& coordinateTwo3DSph = dynamic_cast<const Coordinate3DSpherical<FloatingPrecision>&>(coordinateTwo);
		return getDistance(coordinateOne3DSph, coordinateTwo3DSph);
	}

	FloatingPrecision getDistance(const Coordinate3DSpherical<FloatingPrecision>& coordinateOne,
		                          const Coordinate3DSpherical<FloatingPrecision>& coordinateTwo) const {
		std::vector<FloatingPrecision> pathsLengths(4); // Contains all possible paths

		pathsLengths[0] = Topology3DSph<FloatingPrecision>::getDistance(coordinateOne, coordinateTwo);
		pathsLengths[1] = Topology3DSph<FloatingPrecision>::getDistance(coordinateOne, coordinateTwo.getReciprocal());
		pathsLengths[2] = Topology3DSph<FloatingPrecision>::getDistance(coordinateOne.getReciprocal(), coordinateTwo);
		pathsLengths[3] = Topology3DSph<FloatingPrecision>::getDistance(coordinateOne.getReciprocal(), coordinateTwo.getReciprocal());

		// Riemannian distance is the geodesic, i.e. infimum of allpaths
		return *std::min_element(pathsLengths.begin(), pathsLengths.end());
	}

};

template <typename FloatingPrecision>
class Topology3DSphSym : public Topology3DSph<FloatingPrecision> {
public:
	Topology3DSphSym() {}

	FloatingPrecision getDistance(const Coordinate& coordinateOne,
								  const Coordinate& coordinateTwo) const {
		const auto& coordinateOne3DSph = dynamic_cast<const Coordinate3DSpherical<FloatingPrecision>&>(coordinateOne);
		const auto& coordinateTwo3DSph = dynamic_cast<const Coordinate3DSpherical<FloatingPrecision>&>(coordinateTwo);
		return getDistance(coordinateOne3DSph, coordinateTwo3DSph);
	}

	FloatingPrecision getDistance(const Coordinate3DSpherical<FloatingPrecision>& coordinateOne,
								  const Coordinate3DSpherical<FloatingPrecision>& coordinateTwo) const {
		std::vector<FloatingPrecision> pathsLengths(2); // Contains all possible paths

		pathsLengths[0] = Topology3DSph<FloatingPrecision>::getDistance(coordinateOne, coordinateTwo);
		pathsLengths[1] = Topology3DSph<FloatingPrecision>::getDistance(coordinateOne, coordinateTwo.getBilateralSymmetrical());

		// Riemannian distance is the geodesic, i.e. infimum of allpaths
		return *std::min_element(pathsLengths.begin(), pathsLengths.end());
	}

};

template <typename FloatingPrecision>
class Topology3DSphRecSym : public Topology3DSphRec<FloatingPrecision> {
public:
	Topology3DSphRecSym() {}

	FloatingPrecision getDistance(const Coordinate& coordinateOne,
								  const Coordinate& coordinateTwo) const {
		const auto& coordinateOne3DSph = dynamic_cast<const Coordinate3DSpherical<FloatingPrecision>&>(coordinateOne);
		const auto& coordinateTwo3DSph = dynamic_cast<const Coordinate3DSpherical<FloatingPrecision>&>(coordinateTwo);
		return getDistance(coordinateOne3DSph, coordinateTwo3DSph);
	}

	FloatingPrecision getDistance(const Coordinate3DSpherical<FloatingPrecision>& coordinateOne,
								  const Coordinate3DSpherical<FloatingPrecision>& coordinateTwo) const {
		std::vector<FloatingPrecision> pathsLengths(2); // Contains all possible paths

		pathsLengths[0] = Topology3DSphRec<FloatingPrecision>::getDistance(coordinateOne, coordinateTwo);
		pathsLengths[1] = Topology3DSphRec<FloatingPrecision>::getDistance(coordinateOne, coordinateTwo.getBilateralSymmetrical());

		// Riemannian distance is the geodesic, i.e. infimum of allpaths
		return *std::min_element(pathsLengths.begin(), pathsLengths.end());
	}

};