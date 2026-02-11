#include <gtest/gtest.h>

#include <rivoli/rivoli.hpp>

TEST(VectratypeDistances, GreatCircleNonef)
{
	using vct = vectra::Vectratype<float, vectra::SIMDLevel::None>;
	
	vct thetaOne{0.0};
	vct phiOne  {0.0};
	vct thetaTwo{0.0};
	vct phiTwo  {0.0};
	
	vct result = rivoli::distanceGreatCircle(thetaOne, phiOne, thetaTwo, phiTwo);
	EXPECT_NEAR(result.value, 0.0, 1e-6);
}

TEST(BackendDistances, GreatCircleNonef)
{
	using backend     = vectra::ComputeBackend<float, vectra::SIMDLevel::None>;
	using backendType = typename backend::type;

	backendType thetaOne{ 0.0 };
	backendType phiOne  { 0.0 };
	backendType thetaTwo{ 0.0 };
	backendType phiTwo  { 0.0 };

	backendType result = rivoli::experimental::distanceGreatCircle<float, vectra::SIMDLevel::None>(thetaOne, phiOne, thetaTwo, phiTwo);
	EXPECT_NEAR(result, 0.0, 1e-6);
}