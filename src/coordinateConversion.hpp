#pragma once

#include <tuple>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtx/rotate_vector.hpp>

#include "arrays.hpp"
#include "logger.hpp"


const glm::vec3 t(1.0, 0.0, 0.0);
const glm::vec3 b(0.0, 1.0, 0.0);
const glm::vec3 n(0.0, 0.0, 1.0);


template <typename T> std::tuple<T, T, T>    standardCoordToRusinkiewiczCoord(const std::tuple<T, T, T>& tupleStandardCoordinates);
template <typename T> std::tuple<T, T, T>    standardCoordToRusinkiewiczCoord(const T& thetaI, const T& thetaO, const T& deltaPhi);
template <typename T> std::tuple<T, T, T, T> standardCoordToRusinkiewiczCoord(const std::tuple<T, T, T, T>& tupleStandardCoordinates);
template <typename T> std::tuple<T, T, T, T> standardCoordToRusinkiewiczCoord(const T& thetaI, const T& phiI, const T& thetaO, const T& phiO);

template <typename T> std::tuple<T, T, T>    rusinkiewiczCoordToStandardCoord(const std::tuple<T, T, T>& tupleRusinkiewiczCoordinates);
template <typename T> std::tuple<T, T, T>    rusinkiewiczCoordToStandardCoord(const T& thetaH, const T& thetaD, const T& phiD);
template <typename T> std::tuple<T, T, T, T> rusinkiewiczCoordToStandardCoord(const std::tuple<T, T, T, T>& tupleRusinkiewiczCoordinates);
template <typename T> std::tuple<T, T, T, T> rusinkiewiczCoordToStandardCoord(const T& thetaH, const T& phiH, const T& thetaD, const T& phiD);


/*
 * @return std::tuple containing three values of type T: thetaH, thetaD, and phiD,
 *         which represent the Rusinkiewicz spherical coordinates
 */
template <typename T>
std::tuple<T, T, T> standardCoordToRusinkiewiczCoord(const std::tuple<T, T, T>& tupleStandardCoordinates) {
	std::tuple<T, T, T, T> rusinkiewiczCoordinates = standardCoordToRusinkiewiczCoord(std::get<0>(tupleStandardCoordinates), static_cast<T>(0),
		std::get<1>(tupleStandardCoordinates),
		std::get<2>(tupleStandardCoordinates));
	return { std::get<0>(rusinkiewiczCoordinates) , std::get<2>(rusinkiewiczCoordinates) , std::get<3>(rusinkiewiczCoordinates) };
}

template <typename T>
std::tuple<T, T, T> standardCoordToRusinkiewiczCoord(const T& thetaI, const T& thetaO, const T& deltaPhi) {
	std::tuple<T, T, T, T> rusinkiewiczCoordinates = standardCoordToRusinkiewiczCoord(thetaI, static_cast<T>(0), thetaO, deltaPhi);
	return { std::get<0>(rusinkiewiczCoordinates) , std::get<2>(rusinkiewiczCoordinates) , std::get<3>(rusinkiewiczCoordinates) };
}

template <typename T>
std::tuple<T, T, T, T> standardCoordToRusinkiewiczCoord(const std::tuple<T, T, T, T>& tupleStandardCoordinates) {
	return standardCoordToRusinkiewiczCoord(std::get<0>(tupleStandardCoordinates),
		std::get<1>(tupleStandardCoordinates),
		std::get<2>(tupleStandardCoordinates),
		std::get<3>(tupleStandardCoordinates));
}

/*
 * @brief Converts spherical coordinates to Rusinkiewicz coordinates
 *
 * @reference 10.1007/978-3-7091-6453-2_2
 *
 * @tparam T The data type of the input angles and the returned angles. Must support
 *           trigonometric operations like sin, cos, acos, and atan and be compatible
 *		     with glm::vec3
 *
 * @param thetaI The incident angle theta (between incident direction and macroscopic
 *               normal n to the surface) called incident zenithal angle in radians
 * @param phiI   The incident angle phi (between the projection of the incident direction
 *               on the macroscopic plane of the surface and the macroscopic tangent t to
 *				 the surface) called incident azimuthal angle in radians
 * @param thetaO The observed angle theta (between observed direction and macroscopic
 *               normal n to the surface) called observed zenithal angle in radians
 * @param phiO   The observed angle phi (between the projection of the observed direction
 *               on the macroscopic plane of the surface and the macroscopic tangent t to
 *				 the surface) called observed azimuthal angle in radians
 *
 * @return a std::tuple containing four values of type T: thetaH, phiH, thetaD, and phiD,
 *         which represent the Rusinkiewicz spherical coordinates
 *
 * @note For use with isotropic BRDFs in 3D dimensions only, only the difference between
 *       phiO and phiI matters: it is advisable to give phiI equal to 0 and phiO equal to
 *		 deltaPhi as arguments
 *
 * Dependencies:
 * - GLM (OpenGL Mathematics): A header-only C++ mathematics library for graphics software
 *   based on the OpenGL Shading Language (GLSL) specifications. Used for vector operations
 *   and transformations. Available at: https://github.com/g-truc/glm
 *
 * @author  François Margall
 * @contact francois.margall@inria.fr
 */
template <typename T>
std::tuple<T, T, T, T> standardCoordToRusinkiewiczCoord(const T& thetaI, const T& phiI, const T& thetaO, const T& phiO) {
	glm::vec3 omegaI(glm::sin(thetaI) * glm::cos(phiI), glm::sin(thetaI) * glm::sin(phiI), glm::cos(thetaI));
	glm::vec3 omegaO(glm::sin(thetaO) * glm::cos(phiO), glm::sin(thetaO) * glm::sin(phiO), glm::cos(thetaO));

	glm::vec3 omegaH = glm::normalize(omegaI + omegaO);

	T thetaH = glm::acos(omegaH.z);
	T phiH = glm::atan(omegaH.y, omegaH.x);

	glm::vec3 omegaT = glm::rotate(omegaI, static_cast<float>(-phiH), n); // Temporary result
	glm::vec3 omegaD = glm::rotate(omegaT, static_cast<float>(-thetaH), b);

	T thetaD = glm::acos(omegaD.z);
	T phiD = glm::atan(omegaD.y, omegaD.x);

	return std::tuple<T, T, T, T>(thetaH, phiH, thetaD, phiD);
}

template <typename T>
std::tuple<T, T, T> rusinkiewiczCoordToStandardCoord(const std::tuple<T, T, T>& tupleRusinkiewiczCoordinates) {
	std::tuple<T, T, T, T> standardCoordinates = rusinkiewiczCoordToStandardCoord(std::get<0>(tupleRusinkiewiczCoordinates), static_cast<T>(0),
		std::get<1>(tupleRusinkiewiczCoordinates),
		std::get<2>(tupleRusinkiewiczCoordinates));
	return { std::get<0>(standardCoordinates) , std::get<2>(standardCoordinates) , std::get<3>(standardCoordinates) - std::get<1>(standardCoordinates) };
}

/*
 * @return std::tuple containing three values of type T: thetaI, thetaO, and deltaPhi,
 *         deltaPhi being phiO - phiI, which represent standard spherical coordinates.
 */
template <typename T>
std::tuple<T, T, T> rusinkiewiczCoordToStandardCoord(const T& thetaH, const T& thetaD, const T& phiD) {
	std::tuple<T, T, T, T> standardCoordinates = rusinkiewiczCoordToStandardCoord(thetaH, static_cast<T>(0), thetaD, phiD);
	return { std::get<0>(standardCoordinates) , std::get<2>(standardCoordinates) , std::get<3>(standardCoordinates) - std::get<1>(standardCoordinates) };
}

template <typename T>
std::tuple<T, T, T, T> rusinkiewiczCoordToStandardCoord(const std::tuple<T, T, T, T>& tupleRusinkiewiczCoordinates) {
	return rusinkiewiczCoordToStandardCoord(std::get<0>(tupleRusinkiewiczCoordinates),
		std::get<1>(tupleRusinkiewiczCoordinates),
		std::get<2>(tupleRusinkiewiczCoordinates),
		std::get<3>(tupleRusinkiewiczCoordinates));
}

/*
 * @brief Converts Rusinkiewicz coordinates to spherical coordinates
 *
 * @reference 10.1007/978-3-7091-6453-2_2
 *
 * @tparam T The data type of the input angles and the returned angles. Must support
 *           trigonometric operations like sin, cos, acos, and atan and be compatible
 *		     with glm::vec3
 *
 * @param thetaH The zenithal contribution of the Rusinkiewicz half angle in radians
 * @param phiH   the azimuthal contribution of the Rusinkiewicz half angle in radians
 * @param thetaD The zenithal contribution of the Rusinkiewicz difference angle in radians
 * @param phiD   The azimuthal contribution of the Rusinkiewicz difference angle in radians
 *
 * @return a std::tuple containing four values of type T: thetaI, phiI, thetaO, and phiO,
 *         which represent the standard spherical coordinates
 *
 * @note For use with isotropic BRDFs in 3D dimensions only, phiH angle is useless, and
 *       thus can be defined to 0. Once the result computed, it is mandatory to define
 *       deltaPhi as phiO - phiI
 *
 * Dependencies:
 * - GLM (OpenGL Mathematics): A header-only C++ mathematics library for graphics software
 *   based on the OpenGL Shading Language (GLSL) specifications. Used for vector operations
 *   and transformations. Available at: https://github.com/g-truc/glm
 * - 'arrays.hpp': mandatory in order to use the method 'isClose'
 * - 'logger.hpp': (optional) useful only if one wants to see warning messages if NaN values
 *   appear
 *
 * @author  François Margall
 * @contact francois.margall@inria.fr
 */
template <typename T>
std::tuple<T, T, T, T> rusinkiewiczCoordToStandardCoord(const T& thetaH, const T& phiH, const T& thetaD, const T& phiD) {
	glm::vec3 omegaH(glm::sin(thetaH) * glm::cos(phiH), glm::sin(thetaH) * glm::sin(phiH), glm::cos(thetaH));
	glm::vec3 omegaD(glm::sin(thetaD) * glm::cos(phiD), glm::sin(thetaD) * glm::sin(phiD), glm::cos(thetaD));

	glm::vec3 omegaT = glm::rotate(omegaD, static_cast<float>(thetaH), b);
	glm::vec3 omegaI = glm::rotate(omegaT, static_cast<float>(phiH), n);

	// Carefuly handling rounding errors for arccosine arguments
	if (omegaI.z >= static_cast<decltype(omegaI.z)>(1.) && isClose(omegaI.z, static_cast<decltype(omegaI.z)>(1.)))
		omegaI.z = static_cast<decltype(omegaI.z)>(1.);

	T thetaI = glm::acos(omegaI.z);
	T phiI = glm::atan(omegaI.y, omegaI.x);

	glm::vec3 omegaO = 2 * glm::dot(omegaI, omegaH) * omegaH - omegaI;

	// Carefuly handling rounding errors for arccosine arguments
	if (omegaO.z >= static_cast<decltype(omegaO.z)>(1.) && isClose(omegaO.z, static_cast<decltype(omegaO.z)>(1.)))
		omegaO.z = static_cast<decltype(omegaO.z)>(1.);

	T thetaO = glm::acos(omegaO.z);
	T phiO = glm::atan(omegaO.y, omegaO.x);

	return std::tuple<T, T, T, T>(thetaI, phiI, thetaO, phiO);
}