#pragma once

#include <cmath>
#include <vector>

#include "logger.hpp"

/*
 * @brief Determines if two values are close to each other within a tolerance
 *
 * This function checks if the two input values are close to each other, within a specified
 * relative and absolute tolerance. The function returns true if the absolute difference
 * between argOne and argTwo is less than or equal to the maximum of the relative tolerance
 * times the maximum absolute values of argOne and argTwo, and the absolute tolerance
 *
 * @tparam FloatingPrecision The data type of the input values and tolerance parameters.
 *                           Must support operations such as subtraction and comparison,
 *                           and be compatible with std::fabs and std::max.
 *
 * @param argOne First input value to compare
 * @param argTwo Second input value to compare
 * @param rTol   The relative tolerance parameter, defaulting to 1e-5. It's a positive,
 *               typically very small number
 * @param aTol   The absolute tolerance parameter, defaulting to 1e-8. It's a positive,
 *               typically very small number
 *
 * @return A boolean indicating whether argOne and argTwo are close to each other within
 *         the specified tolerances
 *
 * @reference This implementation is inspired by numpy's `isclose` function, documented
 *            here: https://numpy.org/doc/stable/reference/generated/numpy.isclose.html
 *
 * @author  Francois Margall
 * @contact francois.margall@inria.fr
 */
template <typename FloatingPrecision>
bool isClose(
	const FloatingPrecision& argOne,
	const FloatingPrecision& argTwo,
	const FloatingPrecision& rTol = static_cast<FloatingPrecision>(1e-5),
	const FloatingPrecision& aTol = static_cast<FloatingPrecision>(1e-8))
{
	return std::fabs(argOne - argTwo) <= std::max(rTol * std::max(std::fabs(argOne), std::fabs(argTwo)), aTol);
}

template <typename FloatingPrecision>
std::vector<FloatingPrecision> linspace(
	const FloatingPrecision& start,
	const FloatingPrecision& end,
	const size_t& numPoints)
{
	std::vector<FloatingPrecision> result(numPoints);
	if (start >= end)
		LOG_CRITICAL("'start' should be inferior to 'end'");

	FloatingPrecision step = (end - start) / static_cast<FloatingPrecision>(numPoints - 1);
	for (size_t i = 0; i < numPoints; ++i)
		result[i] = start + i * step;

	return result;
}