#include <array>
#include <vector>

#include <gtest/gtest.h>

#include <rivoli/rivoli.hpp>

TEST(Interpolator, InterpolatorNonef)
{
	std::vector<float> thetaCoordinates = { 0.0, 0.0, 0.0, 0.0 };
	std::vector<float> phiCoordinates   = { 0.0, 0.0, 0.0, 0.0 };
	std::array<std::vector<float>, 2> coordinates  = { thetaCoordinates, phiCoordinates };
	std::vector<float>                coefficients = { 0.0, 0.0, 0.0, 0.0 };

	rivoli::KernelLinear<float, vectra::SIMDLevel::None> kernel;
	rivoli::Topology2S  <float, vectra::SIMDLevel::None> topology;

	rivoli::Interpolator<
		rivoli::KernelLinear<float, vectra::SIMDLevel::None>,
		rivoli::Topology2S  <float, vectra::SIMDLevel::None>> interpolator(coordinates, coefficients, kernel, topology);

	EXPECT_NEAR(interpolator.interpolate(0.0, 0.0), 0.0, 1e-6);
}