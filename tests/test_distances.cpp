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

TEST(VectratypeDistances, 1Rx2SEuclideanNonef)
{
	using vct = vectra::Vectratype<float, vectra::SIMDLevel::None>;
	
	vct thetaAOne{ 0.0 };
	vct thetaBOne{ 0.0 };
	vct phiBOne  { 0.0 };
	vct thetaATwo{ 0.0 };
	vct thetaBTwo{ 0.0 };
	vct phiBTwo  { 0.0 };
	
	vct result = rivoli::distance1Rx2SEuclidean(thetaAOne, thetaBOne, phiBOne, thetaATwo, thetaBTwo, phiBTwo);
	EXPECT_NEAR(result.value, 0.0, 1e-6);
}

TEST(VectratypeDistances, 1Rx2SBilateralEuclideanNonef)
{
	using vct = vectra::Vectratype<float, vectra::SIMDLevel::None>;

	vct thetaAOne{ 1.0 };
	vct thetaBOne{ 0.5 };
	vct phiBOne  { 0.283185307179586 };
	vct thetaATwo{ 1.0 };
	vct thetaBTwo{ 0.5 };
	vct phiBTwo  { 6.0 };

	vct result = rivoli::distance1Rx2SBilateralEuclidean(thetaAOne, thetaBOne, phiBOne, thetaATwo, thetaBTwo, phiBTwo);
	EXPECT_NEAR(result.value, 0.0, 1e-6);
}

TEST(BackendDistances, GreatCircleNonef)
{
	using backend = vectra::ComputeBackend<float, vectra::SIMDLevel::None>;
	using T = typename backend::type;

	T thetaOne{ 0.0 };
	T phiOne  { 0.0 };
	T thetaTwo{ 0.0 };
	T phiTwo  { 0.0 };

	T result = rivoli::experimental::distanceGreatCircle<float, vectra::SIMDLevel::None>(thetaOne, phiOne, thetaTwo, phiTwo);
	EXPECT_NEAR(result, 0.0, 1e-6);
}