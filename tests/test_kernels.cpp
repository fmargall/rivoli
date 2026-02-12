#include <gtest/gtest.h>

#include <rivoli/rivoli.hpp>

TEST(KernerLinear, Nonef)
{
	rivoli::KernelLinear<float, vectra::SIMDLevel::None> kernel;

	float r = 3.5f;
	float result = kernel(r);

	EXPECT_FLOAT_EQ(result, r);
}