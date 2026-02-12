#include <gtest/gtest.h>

#include <rivoli/rivoli.hpp>

TEST(Topology2S, Topology2SNonef)
{
	rivoli::Topology2S<float, vectra::SIMDLevel::None> topology;

	using vct = vectra::Vectratype<float, vectra::SIMDLevel::None>;
	
	vct thetaOne{ 0.0 };
	vct phiOne  { 0.0 };
	vct thetaTwo{ 0.0 };
	vct phiTwo  { 0.0 };

	vct result = topology.getDistance(thetaOne, phiOne, thetaTwo, phiTwo);
	EXPECT_NEAR(result.value, 0.0, 1e-6);
}