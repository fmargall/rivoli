#include <gtest/gtest.h>

#include <rivoli/rivoli.hpp>

TEST(Kernel, LinearNonef)
{
	rivoli::KernelLinear<float, vectra::SIMDLevel::None> kernel;

	float r = 3.5f;
	float result = kernel(r);

	EXPECT_FLOAT_EQ(result, r);
}

TEST(KernelSelector, LinearNonef)
{
	using Kernel = rivoli::KernelSelector<rivoli::KernelType::Linear, float, vectra::SIMDLevel::None>;

	Kernel kernel;

	float r = 3.5f;
	float result = kernel(r);

	EXPECT_FLOAT_EQ(result, r);
}